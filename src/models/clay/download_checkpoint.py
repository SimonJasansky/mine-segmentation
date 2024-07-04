import os
import argparse
from tqdm import tqdm

r"""
This script downloads a checkpoint file from a given URL and saves it to a specified directory.
Example command to run the script from the command line:

python src/models/clay/download_checkpoint.py --url <URL> --directory <DIRECTORY>

Example:
python src/models/clay/download_checkpoint.py --url https://huggingface.co/made-with-clay/Clay/raw/main/clay-v1-base.ckpt --directory models

"""

import urllib.request

def download_checkpoint(url, directory):
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Extract the filename from the URL
    filename = url.split("/")[-1]

    # Get the file size
    file_size = int(urllib.request.urlopen(url).info().get("Content-Length", -1))

    # Download the checkpoint file with progress bar
    with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        def update_progress(block_num, block_size, total_size):
            pbar.update(block_size)

        urllib.request.urlretrieve(url, os.path.join(directory, filename), reporthook=update_progress)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download a checkpoint file from a given URL and save it to a specified directory.")
    parser.add_argument("--url", type=str, help="URL of the checkpoint file")
    parser.add_argument("--directory", type=str, help="Directory to save the checkpoint file")
    args = parser.parse_args()

    # Call the download_checkpoint function with the provided arguments
    download_checkpoint(args.url, args.directory)