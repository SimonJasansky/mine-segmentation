r"""
Data Processing Script
======================================

This script processes GeoTIFF files from the custom dataset to create
image chips for segmentation tasks.

    Run the script as follows:
    python preprocess_data.py <data_dir> <output_dir> <chip_size> [<must_contain_mining>]

    Examples:
    python src/data/make_chips.py data/processed/files data/processed/chips 512
    python src/data/make_chips.py data/processed/files data/processed/chips 512 --must_contain_mining
"""


import os
import sys
from pathlib import Path

import numpy as np
import rasterio as rio
import argparse


def read_and_chip(file_path, chip_size, output_dir):
    """
    Reads a GeoTIFF file, creates chips of specified size, and saves them as
    numpy arrays.

    Args:
        file_path (str or Path): Path to the GeoTIFF file.
        chip_size (int): Size of the square chips.
        output_dir (str or Path): Directory to save the chips.
    """
    os.makedirs(output_dir, exist_ok=True)

    with rio.open(file_path) as src:
        data = src.read()

        n_chips_x = src.width // chip_size
        n_chips_y = src.height // chip_size

        chip_number = 0
        for i in range(n_chips_x):
            for j in range(n_chips_y):
                x1, y1 = i * chip_size, j * chip_size
                x2, y2 = x1 + chip_size, y1 + chip_size

                chip = data[:, y1:y2, x1:x2]
                chip_path = os.path.join(
                    output_dir,
                    f"{Path(file_path).stem}_chip_{chip_number}.npy",
                )
                np.save(chip_path, chip)
                chip_number += 1


def process_files(file_paths, output_dir, chip_size):
    """
    Processes a list of files, creating chips and saving them.

    Args:
        file_paths (list of Path): List of paths to the GeoTIFF files.
        output_dir (str or Path): Directory to save the chips.
        chip_size (int): Size of the square chips.
        must_contain_mining (bool): Flag to indicate if chips must contain some mining area.
    """
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        read_and_chip(file_path, chip_size, output_dir)


def purge_chips(chips_dir, labels_dir):
    """
    Purges chips that do not contain mining area.

    Args:
        chips_dir (str or Path): Directory containing the chips.
        labels_dir (str or Path): Directory containing the labels.
    """
    print(f"Purging chips in {chips_dir} that do not contain mining area")
    for chip_path in chips_dir.glob("*.npy"):
        label_path = labels_dir / f"{chip_path.stem}.npy"
        label_path = str(label_path).replace("_img", "_mask")
        label = np.load(label_path)
        if np.sum(label) == 0:
            os.remove(chip_path)
            os.remove(label_path)

def main():
    """
    Main function to process files and create chips.
    Expects three command line arguments:
        - data_dir: Directory containing the input GeoTIFF files.
        - output_dir: Directory to save the output chips.
        - chip_size: Size of the square chips.
    """

    parser = argparse.ArgumentParser(description="Data Processing Script")
    parser.add_argument("data_dir", help="Directory containing the input GeoTIFF files")
    parser.add_argument("output_dir", help="Directory to save the output chips")
    parser.add_argument("chip_size", type=int, help="Size of the square chips")
    parser.add_argument("--must_contain_mining", action="store_true", help="Flag to indicate if chips must contain some mining area")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    chip_size = int(args.chip_size)
    must_contain_mining = args.must_contain_mining

    train_image_paths = list((data_dir / "train").glob("*_img.tif"))
    val_image_paths = list((data_dir / "val").glob("*_img.tif"))
    train_label_paths = list((data_dir / "train").glob("*_mask.tif"))
    val_label_paths = list((data_dir / "val").glob("*_mask.tif"))

    # check if chips already exist and remove them
    if output_dir.exists():
        print(f"Removing existing chips in {output_dir}")
        os.system(f"rm -r {output_dir}")
    
    process_files(train_image_paths, output_dir / "train/chips", chip_size)
    process_files(val_image_paths, output_dir / "val/chips", chip_size)
    process_files(train_label_paths, output_dir / "train/labels", chip_size)
    process_files(val_label_paths, output_dir / "val/labels", chip_size)

    if must_contain_mining:
        print("Purging chips that do not contain mining area")
        purge_chips(output_dir / "train/chips", output_dir / "train/labels")
        purge_chips(output_dir / "val/chips", output_dir / "val/labels")

if __name__ == "__main__":
    main()
