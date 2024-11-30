import numpy as np
import cupy as cp
from models.dpgmm import DPGMM
class ContinuousDataTransformer():
	"""
		Transformer class responsible for processing continuous data to train the CTABGANSynthesizer model
		Args:
	        n_clusters (int): Number of clusters (maximum components) for the DPGMM.
	        eps (float): Minimum weight threshold for filtering components.

	    Attributes:
	    	n_clusters (int): Number of clusters for the DPGMM.
	        eps (float): Threshold for active components.
	        ordering (list): Stores the ordering of components for inverse transformation.
	        output_info (list): Stores information about the output structure.
	        output_dim (int): Dimensionality of the transformed data.
	        components (array-like): Boolean array indicating active components.
	        model (DPGMM): The DPGMM model instance.
	        data_mean (float): Mean of the input data for normalization.
	        data_std (float): Standard deviation of the input data for normalization.
	"""
	def __init__(self, n_clusters=10, eps=0.005):
	    self.n_clusters = n_clusters
	    self.eps = eps
	    self.ordering = []
	    self.output_info = []
	    self.output_dim = 0
	    self.components = []
	    self.filter_arr = []
	    self.model = None
	    self.data_mean = None
	    self.data_std = None

	def fit(self, column_data):
        """
	        Fit the DPGMM model to the input data and identify active components.

	        Args:
	            column_data (np.ndarray): 1D array of input data to fit the model.

	        Returns:
	            None
        """
        self.mean = np.mean(column_data)
        self.std = np.std(column_data)

        column_data = (column_data - self.mean) / self.std  # Normalize data
        self.model = DPGMM(max_components=self.n_clusters, alpha=0.005, learning_rate=0.01)
        self.model.fit(column_data, num_steps=100)

        # Identify active components based on weights and predictions
        old_comp = self.model.get_weights() > self.eps
        predicted_modes = self.model.predict(column_data)
        mode_freq = np.unique(predicted_modes)
        active_components = np.isin(np.arange(self.n_clusters), mode_freq) & old_comp

        self.components = active_components
        self.output_info += [(1, 'tanh'), (np.sum(active_components), 'softmax')]
        self.output_dim += 1 + np.sum(active_components)

    def transform(self, column_data):
        """
	        Transform input data into a latent space with selected features and one-hot encodings.

	        Args:
	            column_data (np.ndarray): 1D array of input data to transform.

	        Returns:
	            cp.ndarray: Transformed data with selected features and one-hot encodings.
        """
        column_data = cp.asarray(column_data)  # Move data to GPU
        column_data = (column_data - self.mean) / self.std  # Normalize data
        n_samples = len(column_data)

        # Compute means and standard deviations for Gaussian components
        means, covariances = self.model.get_component_stats()
        means = cp.asarray(means).reshape((1, self.n_clusters))
        stds = cp.sqrt(cp.asarray(covariances)).reshape((1, self.n_clusters))

        # Normalize the data
        features = (column_data[:, None] - means) / (4 * stds)

        # Filter active components
        active_components = cp.asarray(self.components, dtype=bool)
        n_active_components = cp.sum(active_components)

        # Compute probabilities
        probs = cp.asarray(self.model.predict_proba(cp.asnumpy(column_data)))
        probs = probs[:, active_components]
        probs += 1e-6  # Avoid zero probabilities
        probs /= cp.sum(probs, axis=1, keepdims=True)

        # Weighted sampling
        cdf = cp.cumsum(probs, axis=1)
        random_vals = cp.random.rand(n_samples).reshape(-1, 1)
        opt_sel = cp.sum(random_vals > cdf, axis=1)

        # Create one-hot encodings
        probs_onehot = cp.zeros_like(probs)
        probs_onehot[cp.arange(n_samples), opt_sel] = 1

        # Extract selected features
        features = features[:, active_components]
        selected_features = features[cp.arange(n_samples), opt_sel].reshape(-1, 1)
        selected_features = cp.clip(selected_features, -0.99, 0.99)

        # Reorder encodings based on frequency
        mode_frequencies = cp.sum(probs_onehot, axis=0)
        reordered_indices = cp.argsort(-mode_frequencies)
        reordered_probs_onehot = probs_onehot[:, reordered_indices]

        # Save ordering for inverse transformation
        self.ordering.append(cp.asnumpy(reordered_indices))

        # Combine features and one-hot encodings
        transformed_data = cp.hstack([selected_features, reordered_probs_onehot])

        return transformed_data

    def inverse_transform(self, data):
        """
	        Perform inverse transformation from the latent space back to the original data space.

	        Args:
	            data (cp.ndarray): Transformed data to inverse transform.

	        Returns:
	            np.ndarray: Original data reconstructed from the transformed data.
        """
        data = cp.asarray(data)
        n_samples = data.shape[0]
        data_t = cp.zeros((n_samples, 1))

        # Extract normalized values
        u = cp.clip(data[:, 0], -1, 1)

        # Extract and reorder one-hot encodings
        v = data[:, 1 : 1 + cp.sum(self.components)]
        order = self.ordering[0]
        v_re_ordered = cp.zeros_like(v)
        for id, val in enumerate(order):
            v_re_ordered[:, val] = v[:, id]
        v = v_re_ordered

        # Map encodings back to components
        v_t = cp.ones((n_samples, self.n_clusters)) * -100
        v_t[:, self.components] = v
        p_argmax = cp.argmax(v_t, axis=1)

        # Retrieve means and standard deviations for selected components
        means, covariances = self.model.get_component_stats()
        means = cp.asarray(means).flatten()
        stds = cp.sqrt(cp.asarray(covariances).flatten())
        selected_means = means[p_argmax]
        selected_stds = stds[p_argmax]

        # Reconstruct original values
        restored_values = u * 4 * selected_stds + selected_means
        data_t[:, 0] = restored_values * self.std + self.mean

        return cp.asnumpy(data_t)