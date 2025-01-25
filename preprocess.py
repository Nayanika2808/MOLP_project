import pandas as pd

# Input and output file paths
input_file = 'data/raw_data.csv'
output_file = 'data/preprocessed_data.csv'

try:
    # Read raw data
    raw_data = pd.read_csv(input_file)

    # Rename columns if necessary (this ensures consistency)
    raw_data.columns = ['feature', 'target']

    # Drop rows with missing or malformed data
    processed_data = raw_data.dropna()

    # Save the cleaned data
    processed_data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")