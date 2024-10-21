r"""
Data Processing Script
======================================

This script processes GeoTIFF files from the custom dataset to create
image chips for segmentation tasks.

    Run the script as follows:
    python preprocess_data.py <data_dir> <output_dir> <chip_size> <chip_format> [<must_contain_mining>] [<normalize>] [--split <split>]

    Examples:
    python src/data/06_make_chips.py data/processed/files data/processed/chips/npy/512 512 npy --split train
    python src/data/06_make_chips.py data/processed/files data/processed/chips/tif/1024 1024 tif --must_contain_mining --normalize --split val
"""


import os
import sys
from pathlib import Path
import argparse
import numpy as np
import rasterio as rio
from samgeo import split_raster

from src.utils import normalize_geotiff
from tqdm import tqdm

def read_and_chip(file_path, chip_size, output_dir, chip_format, use_first_channels=True):
    """
    Reads a GeoTIFF file, creates chips of specified size, and saves them as
    numpy arrays or TIFF files.

    Args:
        file_path (str or Path): Path to the GeoTIFF file.
        chip_size (int): Size of the square chips.
        output_dir (str or Path): Directory to save the chips.
        chip_format (str): Format to save the chips. Either 'npy' or 'tif'.
        use_first_channels (bool): Flag to indicate if only the first 3 channels should be used.
    """
    os.makedirs(output_dir, exist_ok=True)

    if chip_format not in ["npy", "tif"]:
        raise ValueError("Invalid chip format. Use 'npy' or 'tif'.")

    s2_image_name = Path(file_path).stem

    if chip_format == "npy":
        with rio.open(file_path) as src:
            data = src.read()

            if use_first_channels:
                data = data[:3, :, :]  # Use only the first 3 channels

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
                        f"{s2_image_name}_chip_{chip_number}.npy",
                    )
                    np.save(chip_path, chip)
                    chip_number += 1

    if chip_format == "tif":
        split_raster(
            filename=file_path,
            out_dir=output_dir,
            tile_size=chip_size,
            overlap=0,
        )

        # rename the chips
        chip_number = 0
        for chip_path in Path(output_dir).glob("*tile_*.tif"):
            new_chip_path = os.path.join(
                        output_dir,
                        f"{s2_image_name}_chip_{chip_number}.tif",
                    )
            chip_path.rename(new_chip_path)
            chip_number += 1


def process_files(file_paths, output_dir, chip_size, chip_format):
    """
    Processes a list of files, creating chips and saving them.

    Args:
        file_paths (list of Path): List of paths to the GeoTIFF files.
        output_dir (str or Path): Directory to save the chips.
        chip_size (int): Size of the square chips.
        chip_format (str): Format to save the chips. Either 'npy' or 'tif'.
        must_contain_mining (bool): Flag to indicate if chips must contain some mining area.
    """
    print(f"Processing files in {output_dir}")
    for file_path in tqdm(file_paths, desc="Processing files"):
        # print(f"Processing: {file_path}")
        read_and_chip(file_path, chip_size, output_dir, chip_format)


def purge_chips(chips_dir, labels_dir, chip_format, testset_mode=False):
    """
    Purges chips that do not contain mining area.

    Args:
        chips_dir (str or Path): Directory containing the chips.
        labels_dir (str or Path): Directory containing the labels.
        chip_format (str): Format of the chips.
        testset_mode (bool): Flag to indicate if the chips are from the test set. In case they are, add "nominearea" instead of removing them.
    """
    if type(chips_dir) == str:
        chips_dir = Path(chips_dir)
    if type(labels_dir) == str:
        labels_dir = Path(labels_dir)
    
    print(f"Purging chips in {chips_dir} that do not contain mining area")
    removed_count = 0
    renamed_count = 0
    chips = list(chips_dir.glob(f"*.{chip_format}"))
    for chip_path in tqdm(chips, desc="Purging chips"):
        label_path = labels_dir / f"{chip_path.stem}.{chip_format}"
        label_path = str(label_path).replace("_img", "_mask")
        label_path = Path(label_path)
        
        if chip_format == "npy":
            label = np.load(label_path)
            if np.sum(label) == 0:
                if testset_mode:
                    new_chip_path = chip_path.parent / f"{chip_path.stem}_nominearea.{chip_format}"
                    chip_path.rename(new_chip_path)
                    new_label_path = label_path.parent / f"{label_path.stem}_nominearea.{chip_format}"
                    label_path.rename(new_label_path)
                    renamed_count += 1
                else:
                    os.remove(chip_path)
                    os.remove(label_path)
                    removed_count += 1
                
        if chip_format == "tif":
            with rio.open(label_path) as src:
                label = src.read()
                if np.sum(label) == 0:
                    if testset_mode:
                        new_chip_path = chip_path.parent / f"{chip_path.stem}_nominearea.{chip_format}"
                        chip_path.rename(new_chip_path)
                        new_label_path = label_path.parent / f"{label_path.stem}_nominearea.{chip_format}"
                        label_path.rename(new_label_path)
                        renamed_count += 1
                    else:
                        os.remove(chip_path)
                        os.remove(label_path)
                        removed_count += 1
    
    print(f"Removed {removed_count} chips, renamed {renamed_count} chips without mining area")


def check_for_nan_values(chips_dir, labels_dir, chip_format):
    """
    Checks for nan values in chips and removes chip+label with nan values.

    Args:
        chips_dir (str or Path): Directory containing the chips.
        labels_dir (str or Path): Directory containing the labels.
        chip_format (str): Format of the chips.
    """
    print(f"Checking for nan values in chips in {chips_dir}")
    removed_count = 0
    chips = list(chips_dir.glob(f"*.{chip_format}"))
    for chip_path in tqdm(chips, desc="Checking for nan values"):
        label_path = labels_dir / f"{chip_path.stem}.{chip_format}"
        label_path = str(label_path).replace("_img", "_mask")
    
        if chip_format == "npy":
            chip = np.load(chip_path)
            label = np.load(label_path)
            if np.isnan(chip).any():
                os.remove(chip_path)
                os.remove(label_path)
                removed_count += 1
            if np.isnan(label).any():
                os.remove(chip_path)
                os.remove(label_path)
                removed_count += 1
            
        if chip_format == "tif":
            with rio.open(chip_path) as src:
                chip = src.read()
                label = np.load(label_path)
                if np.isnan(chip).any():
                    os.remove(chip_path)
                    os.remove(label_path)
                    removed_count += 1
                if np.isnan(label).any():
                    os.remove(chip_path)
                    os.remove(label_path)
                    removed_count += 1
    
    print(f"Removed {removed_count} chips with nan values")

def main():
    """
    Main function to process files and create chips.
    Expects three command line arguments:
        - data_dir: Directory containing the input GeoTIFF files.
        - output_dir: Directory to save the output chips.
        - chip_size: Size of the square chips.
        - chip_format: Format to save the chips. Either 'npy' or 'tif'.
        - must_contain_mining: Flag to indicate if chips must contain some mining area.
        - normalize: Flag to indicate if chips must be normalized. Can only be true with tif format.
        - split: Specify which split to persist. Options: 'all', 'train', 'val', 'test'

    Examples:
    python src/data/06_make_chips.py data/processed/files data/processed/npy/chips 512 npy --split train
    python src/data/06_make_chips.py data/processed/files data/processed/tif/chips 1024 tif --must_contain_mining --normalize --split val
    """

    parser = argparse.ArgumentParser(description="Data Processing Script")
    parser.add_argument("data_dir", help="Directory containing the input GeoTIFF files")
    parser.add_argument("output_dir", help="Directory to save the output chips")
    parser.add_argument("chip_size", type=int, help="Size of the square chips")
    parser.add_argument("chip_format", help="Format to save the chips. Either 'npy' or 'tif'")
    parser.add_argument("--must_contain_mining", action="store_true", help="Flag to indicate if chips must contain some mining area")
    parser.add_argument("--normalize", action="store_true", help="Flag to indicate if chips must be normalized. Can only be true with tif format")
    parser.add_argument("--split", default="all", help="Specify which split to persist. Options: 'all', 'train', 'val', 'test'")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    chip_size = int(args.chip_size)
    chip_format = args.chip_format
    must_contain_mining = args.must_contain_mining
    normalize = args.normalize
    split = args.split

    if chip_format not in ["npy", "tif"]:
        raise ValueError("Invalid chip format. Use 'npy' or 'tif'.")
    
    if normalize and chip_format != "tif":
        raise ValueError("Normalization can only be done with tif format")

    train_image_paths = list((data_dir / "train").glob("*_img.tif"))
    val_image_paths = list((data_dir / "val").glob("*_img.tif"))
    test_image_paths = list((data_dir / "test").glob("*_img.tif"))
    train_label_paths = list((data_dir / "train").glob("*_mask.tif"))
    val_label_paths = list((data_dir / "val").glob("*_mask.tif"))
    test_label_paths = list((data_dir / "test").glob("*_mask.tif"))

    # check if chips already exist and remove them
    if output_dir.exists():
        print(f"Removing existing chips in {output_dir}")
        if split == "train" or split == "all":
            os.system(f"rm -r {output_dir / 'train'}")
        if split == "val" or split == "all":
            os.system(f"rm -r {output_dir / 'val'}")
        if split == "test" or split == "all":
            os.system(f"rm -r {output_dir / 'test'}")
    
    if split == "train" or split == "all":
        process_files(train_image_paths, output_dir / "train/chips", chip_size, chip_format)
        process_files(train_label_paths, output_dir / "train/labels", chip_size, chip_format)
    if split == "val" or split == "all":
        process_files(val_image_paths, output_dir / "val/chips", chip_size, chip_format)
        process_files(val_label_paths, output_dir / "val/labels", chip_size, chip_format)
    if split == "test" or split == "all":
        process_files(test_image_paths, output_dir / "test/chips", chip_size, chip_format)
        process_files(test_label_paths, output_dir / "test/labels", chip_size, chip_format)

    if must_contain_mining:
        print("Purging chips that do not contain mining area")
        if split == "train" or split == "all":
            purge_chips(output_dir / "train/chips", output_dir / "train/labels", chip_format)
        if split == "val" or split == "all":
            purge_chips(output_dir / "val/chips", output_dir / "val/labels", chip_format)
    if split == "test" or split == "all":
        purge_chips(output_dir / "test/chips", output_dir / "test/labels", chip_format, testset_mode=True)
    
    # check for nan values in chips
    if chip_format == "npy":
        print("Checking for nan values in chips")
        if split == "train" or split == "all":
            check_for_nan_values(output_dir / "train/chips", output_dir / "train/labels", chip_format)
        if split == "val" or split == "all":
            check_for_nan_values(output_dir / "val/chips", output_dir / "val/labels", chip_format)
        if split == "test" or split == "all":
            check_for_nan_values(output_dir / "test/chips", output_dir / "test/labels", chip_format)

    if normalize and chip_format == "tif":
        print("Normalizing chips")
        if split == "train" or split == "all":
            for chip_path in (output_dir / "train" / "chips").glob(f"*.{chip_format}"):
                normalize_geotiff(chip_path, chip_path)
        if split == "val" or split == "all":
            for chip_path in (output_dir / "val" / "chips").glob(f"*.{chip_format}"):
                normalize_geotiff(chip_path, chip_path)
        if split == "test" or split == "all":
            for chip_path in (output_dir / "test" / "chips").glob(f"*.{chip_format}"):
                normalize_geotiff(chip_path, chip_path)

    print(f"Chips saved to {output_dir}")
    print(f"Nr. Train chips: {len(list((output_dir / 'train' / 'chips').glob(f'*.{chip_format}')))}")
    print(f"Nr. Val chips: {len(list((output_dir / 'val' / 'chips').glob(f'*.{chip_format}')))}")

    n_test_chips = len(list((output_dir / 'test' / 'chips').glob(f'*.{chip_format}')))
    n_test_chips_nominearea = len(list((output_dir / 'test' / 'chips').glob(f'*_nominearea.{chip_format}')))
    print(f"Nr. Test chips: {n_test_chips}")
    print(f".   of which without mining area: {n_test_chips_nominearea}")
    print(f".   of which with mining area: {n_test_chips - n_test_chips_nominearea}")

if __name__ == "__main__":
    main()
