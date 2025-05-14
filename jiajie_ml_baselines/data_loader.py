# jiajie_ml_baselines/data_loader.py
import pandas as pd
import os
import config

def load_data(file_path):
    """Loads training or test data from a CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        raise FileNotFoundError(f"File not found at {file_path}")
    try:
        df = pd.read_csv(file_path, index_col=0)
        print(f"Data loaded successfully from {file_path}.")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise

def load_training_data():
    """Loads the training dataset."""
    return load_data(config.TRAIN_CSV)

def load_test_data():
    """Loads the test dataset."""
    return load_data(config.TEST_CSV)

if __name__ == '__main__':
    try:
        train_df = load_training_data()
        print("Training Data Head:\n", train_df.head())
        print("\nTraining Data Info:\n", train_df.info())

        test_df = load_test_data()
        print("\nTest Data Head:\n", test_df.head())
        print("\nTest Data Info:\n", test_df.info())
    except Exception as e:
        print(f"Failed to load data: {e}")