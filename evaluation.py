import os
import numpy as np

def calculate_errors(actual_folder, result_folder, metric="rmse"):
    """
    Calculate the error between actual_data and result_data for all partitions.

    Args:
        actual_folder (str): Path to the folder containing actual_data NumPy files.
        result_folder (str): Path to the folder containing result_data NumPy files.
        metric (str): Metric for error calculation. Supported: "mse", "mae", "rmse".

    Returns:
        list: A list containing the calculated error for each partition.

    Raises:
        ValueError: If the number of files or their shapes mismatch between actual_data 
                    and result_data folders, or if an unsupported metric is provided.
    """
    # Retrieve and sort file lists to ensure correct correspondence
    actual_files = sorted(os.listdir(actual_folder))
    result_files = sorted(os.listdir(result_folder))
    
    # Check for file count mismatch
    if len(actual_files) != len(result_files):
        raise ValueError("Mismatch in the number of files between actual_data and result_data folders.")
    
    errors = []  # List to store error values for each partition

    for actual_file, result_file in zip(actual_files, result_files):
        # Load NumPy files
        actual_data = np.load(os.path.join(actual_folder, actual_file))
        result_data = np.load(os.path.join(result_folder, result_file))

        # Check for shape mismatch
        if actual_data.shape != result_data.shape:
            raise ValueError(f"Shape mismatch between {actual_file} and {result_file}")

        # Calculate the error based on the selected metric
        if metric == "mse":
            error = np.mean((actual_data - result_data) ** 2)
        elif metric == "mae":
            error = np.mean(np.abs(actual_data - result_data))
        elif metric == "rmse":
            error = np.sqrt(np.mean((actual_data - result_data) ** 2))
        else:
            raise ValueError(f"Unsupported metric: {metric}. Supported metrics: 'mse', 'mae', 'rmse'.")

        errors.append(error)

    return errors
