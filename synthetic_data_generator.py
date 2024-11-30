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