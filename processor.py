import os
import dask.dataframe as dd
from transformers.continuous_data_transformer import ContinuousDataTransformer
import numpy as np
def read_input_data(input_parquet_dir, column_name):
	"""
	    Reads input data from a directory containing Parquet files, selects the specified column,
	    and repartitions it for processing.

	    Args:
	        input_parquet_dir (str): Directory containing Parquet files.
	        column_name (str): Name of the column to read and process.

	    Returns:
	        dask.DataFrame: A Dask DataFrame containing the selected column, repartitioned.
    """
	ddf = dd.read_parquet(input_parquet_dir, columns=[column_name])
	ddf = ddf.repartition(partition_size="50MB")
	return ddf

def process_partition(partition, partition_idx):
	"""
		Processes a single partition of data: fits the DPGMM model, transforms data,
	    and performs inverse transformation. Saves actual, transformed, and result data
	    as NumPy arrays for later evaluation.

	    Args:
	        partition (DataFrame): The data partition to process.
	        partition_idx (int): Index of the partition for file naming.

	    Returns:
	        tuple: Paths to the saved actual, transformed, and result data files.
    """
	transformer = ContinuousDataTransformer(n_clusters=10, eps=0.005)
	partition_np = partition.to_numpy().astype(float)  # Ensure data type is float for processing
	transformer.fit(partition_np)

	# Transform and inverse transform the data
	transformed_data = transformer.transform(partition_np)
	inv_transform_data = transformer.inverse_transform(transformed_data)

	actual_path = os.path.join("actual_data", f"partition_{partition_idx}.npy")
	np.save(actual_path, partition_np)

	transformed_path = os.path.join("transformed_data", f"partition_{partition_idx}.npy")
	np.save(transformed_path, transformed_data)

	result_path = os.path.join("result_data", f"partition_{partition_idx}.npy")
	np.save(result_path, inv_transform_data)

	return actual_path, transformed_path, result_path
