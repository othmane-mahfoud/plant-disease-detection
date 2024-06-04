import os
import requests
import zipfile
import shutil
import random

from tqdm.auto import tqdm

def download_dataset(url, save_path):
    """
    Download dataset from the given URL and save it to the specified path.

    Args:
    - url (str): URL of the dataset to download.
    - save_path (str): Path where the dataset will be saved.

    Returns:
    - None
    """

    # Check if data folder already exists
    if os.path.exists(save_path):
        print("Data folder already exists. Skipping download.")
        return
    else:
        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        print(f"Dataset downloaded and saved to {save_path}")


# Function to extract the zip file
def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a specified directory.

    Args:
    - zip_path (str): Path to the zip file to be extracted.
    - extract_to (str): Directory where the zip file will be extracted.

    Returns:
    - None
    """
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Dataset extracted to {extract_to}")


def generate_dataset_splits(dataset_dir, splits):
    """
    Generate multiple dataset folders with different splits from the original dataset.

    Args:
    - dataset_dir (str): Path to the original dataset directory.
    - splits (dict): Dictionary with the expected output folder name as keys and
                     the train, test, and holdout splits as values.

    Returns:
    - None
    """
    # Iterate over each split configuration
    for split_name, split in splits.items():
        train_split, test_split, holdout_split = split
        
        # Create output directories for train, test, and holdout splits
        train_dir = os.path.join(dataset_dir, f"{split_name}/train")
        test_dir = os.path.join(dataset_dir, f"{split_name}/test")
        holdout_dir = os.path.join(dataset_dir, f"{split_name}/holdout")
        
        # Iterate over each class directory in the dataset
        for class_name in os.listdir(dataset_dir):
            class_dir = os.path.join(dataset_dir, class_name)
            
            # Skip if not a directory
            if not os.path.isdir(class_dir):
                continue
            
            # List all files in the class directory
            files = os.listdir(class_dir)
            random.shuffle(files)
                
            # Calculate the split index
            train_split_index = int(len(files) * train_split)
            test_split_index = train_split_index + int(len(files) * test_split)
            
            # Create class directories in train, test, and holdout directories
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            holdout_class_dir = os.path.join(holdout_dir, class_name)
            
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)
            os.makedirs(holdout_class_dir, exist_ok=True)
            
            # Move files to train directory
            for file in files[:train_split_index]:
                src_file = os.path.join(class_dir, file)
                dst_file = os.path.join(train_class_dir, file)
                shutil.move(src_file, dst_file)
            
            # Move files to test directory
            for file in files[train_split_index:test_split_index]:
                src_file = os.path.join(class_dir, file)
                dst_file = os.path.join(test_class_dir, file)
                shutil.move(src_file, dst_file)
            
            # Move files to holdout directory
            for file in files[test_split_index:]:
                src_file = os.path.join(class_dir, file)
                dst_file = os.path.join(holdout_class_dir, file)
                shutil.move(src_file, dst_file)
                
    print("Dataset splits generated successfully!")


def create_dataset(
        url: str, 
        zip_path: str,
        splits: dict = {'data_70_20_10': [0.7, 0.2, 0.1]}, 
        data_dir: str = "data",
        extracted_dir: str = ""
    ):
    
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the dataset
    download_dataset(url, zip_path)
    
    # Extract the downloaded zip file
    extract_zip(zip_path, data_dir)
    
    # Remove the zip file after extraction (optional)
    os.remove(zip_path)
    
    # Path to the root directory of the dataset
    dataset_dir = f"{data_dir}/{extracted_dir}"
    
    # Split the dataset according to the splits specified
    generate_dataset_splits(dataset_dir, splits)