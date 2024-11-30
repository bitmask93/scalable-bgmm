import pandas as pd

def generate_chunk(chunk_id):
     """
	    Generate a synthetic data chunk with replicated rows and added noise.
	    
	    Args:
	        chunk_id (int): ID of the chunk being generated.
	    
	    Returns:
	        pd.DataFrame: Synthetic data for the chunk.
    """
    noise = np.random.normal(0, 0.01, size=(current_count * replications_per_chunk,))
    synthetic_data = np.tile(price_data, replications_per_chunk) + noise
    return pd.DataFrame({'Amount': synthetic_data})

def generate_synthetic_data(existing_file_path, column_name, output_dir, target_count, num_chunks = 1000000):
	"""
		Generate synthetic data by replicating existing data and adding random noise, and save the result in chunks as Parquet files.

		Args:
		    existing_file_path (str): Path to the CSV file containing the original data.
		    column_name (str): Name of the column in the CSV file to use for data replication.
		    output_dir (str): Directory to save the generated synthetic data files.
		    target_count (int): Total number of rows to generate in the synthetic dataset.
		    num_chunks (int, optional): Number of chunks to split the generated data into. Defaults to 1,000,000.

		Returns:
    None: Saves the synthetic data chunks directly to the specified output directory.
	"""
	os.makedirs(output_dir, exist_ok=True)

	data = pd.read_csv(existing_file_path)[column_name].values
	current_count = len(data)

	replications_per_chunk = chunk_size // current_count

	# Generate and save data in chunks
	for i in range(target_count // chunk_size):
		print(f"Generating chunk {i + 1}...")
    	chunk_df = generate_chunk(i)

    	output_path = os.path.join(output_dir, f"synthetic_data_part_{i + 1}.parquet")
    	chunk_df.to_parquet(output_path, engine="pyarrow", index=False)