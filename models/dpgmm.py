import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

class DPGMM:
    def __init__(self, max_components=10, alpha=0.01, learning_rate=0.01):
        """
	        Initialize the DP(Dirichlet Process)-GMM model.
	        Args:
	            max_components: Maximum number of mixture components.
	            alpha: Concentration parameter for the Dirichlet Process.
	            learning_rate: Learning rate for optimization.
        """
        self.max_components = max_components
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.dp_gmm = None

    def build_model(self):
        """
	        Build the DP-GMM model.
        """
        # Dirichlet Process Prior
        dirichlet = tfd.Dirichlet(concentration=tf.ones([self.max_components]) * self.alpha)
        categorical = tfd.Categorical(probs=dirichlet.sample())

        # Gaussian Mixture Components
        self.locs = tf.Variable(tf.random.normal([self.max_components]), name="locs")
        self.scales = tfp.util.TransformedVariable(
            tf.fill([self.max_components], value=1.0), bijector=tfb.Exp(), name="scales"
        )

        # Mixture Distribution
        self.logits = tf.Variable(tf.random.normal([self.max_components]), name="logits")
        self.dp_gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.logits),
            components_distribution=tfd.Normal(loc=self.locs, scale=self.scales),
        )

    def fit(self, data, num_steps=100):
        """
		    Train the DP-GMM model.
		    Args:
		        data: Input data (1D NumPy array).
		        num_steps: Number of optimization steps.
        """
        if self.dp_gmm is None:
            self.build_model()

        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        
        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                nll = -tf.reduce_mean(self.dp_gmm.log_prob(data))  # Negative log likelihood
            gradients = tape.gradient(nll, self.dp_gmm.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.dp_gmm.trainable_variables))
            return nll

        #for step in range(num_steps):
        #    nll = train_step()
        #    if step % 100 == 0:
        #        print(f"Step {step}, NLL: {nll.numpy()}")

    def get_weights(self):
        """
	        Extract the mixture weights.
	        Returns:
	            A NumPy array of component weights.
        """
        mixture_distribution = self.dp_gmm.mixture_distribution
        component_weights = tf.nn.softmax(mixture_distribution.logits)  # Convert logits to probabilities
        return component_weights.numpy()

    def predict_proba(self, data):
      """
	      Calculate the probabilities of data points belonging to each component.
	      Args:
	          data: Input data (1D NumPy array).
	      Returns:
	          A 2D NumPy array where each row corresponds to a data point and each column corresponds to a component.
      """
      data = tf.convert_to_tensor(data, dtype=tf.float32)
      responsibilities = self.dp_gmm.components_distribution.prob(data[:, None])  # Shape: (n_samples, n_clusters)

      # Retrieve mixture weights
      weights = self.get_weights()  # Shape: (n_clusters,)

      # Compute weighted probabilities
      prob_matrix = responsibilities * weights  # Shape: (n_samples, n_clusters)

      # Handle zero probabilities by normalizing across rows
      total_probs = tf.reduce_sum(prob_matrix, axis=1, keepdims=True)  # Shape: (n_samples, 1)
      total_probs = tf.where(total_probs == 0, tf.ones_like(total_probs), total_probs)  # Avoid division by zero

      # Normalize probabilities across components
      normalized_probs = prob_matrix / total_probs 

      # Ensure the output is always 2D
      normalized_probs = tf.reshape(normalized_probs, (-1, self.dp_gmm.components_distribution.batch_shape.num_elements()))

      return normalized_probs.numpy()

    
    def predict(self, data):
      """
        Predict the most likely component for each data point.
        Args:
            data: Input data (1D NumPy array).
        Returns:
            A 1D NumPy array where each element is the index of the most probable component for the corresponding data point.
      """
      prob_matrix = self.predict_proba(data)
      return np.argmax(prob_matrix,axis=1)
  
    def get_component_stats(self):
      """
        Compute the means and covariances for the Gaussian components.
        Returns:
            A tuple (means, covariances) where:
            - means: A 1D NumPy array of the means of the Gaussian components.
            - covariances: A 1D NumPy array of the covariances of the Gaussian components.
      """
      means = self.locs.numpy()
      covariances = np.square(self.scales.numpy())
      return means, covariances