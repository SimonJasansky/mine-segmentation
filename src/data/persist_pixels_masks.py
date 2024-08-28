r"""

Example:

```bash
python src/data/persist_pixels_masks.py data/processed/files preferred_polygons --limit 25 --train_ratio 0.8 --only_valid_surface_mines
```

"""

from typing import List
import geopandas as gpd
import pandas as pd
import os
import rasterio
import numpy as np
import argparse
from tqdm import tqdm
tqdm.pandas()

from src.data.get_satellite_images import ReadSTAC

DATASET_FILTERED = "data/processed/mining_tiles_with_masks_and_bounding_boxes_filtered.gpkg"

def process_row(
        row: pd.Series, 
        masks: pd.DataFrame, 
        stac_reader: ReadSTAC, 
        bands: List[str], 
        output_path: str
    ) -> None:
    """
    Process a row of data by retrieving the corresponding item from a STAC catalog,
    reading the stack of bands, persisting the stack as a GeoTIFF image, and saving
    the mask raster.

    Args:
        row (pd.Series): The row of data containing information about the tile.
        masks (pd.DataFrame): The DataFrame containing the masks.
        stac_reader (ReadSTAC): The STAC reader object used to retrieve STAC items.
        bands (List[str]): The list of bands to read from the stack.
        output_path (str): The path to save the output files.

    Returns:
        None
    """
    bounds = row.geometry.bounds
    lat = row.geometry.centroid.y
    lon = row.geometry.centroid.x
    tile_id = row.tile_id
    img_path = output_path + f"/{tile_id}_{row.s2_tile_id}_img.tif"
    mask_path = output_path + f"/{tile_id}_{row.s2_tile_id}_mask.tif"

    # check if the image already exists
    if os.path.exists(img_path) and os.path.exists(mask_path):
        print(f"Image and mask already exist for tile {row.tile_id}")
        return
    elif os.path.exists(img_path) and not os.path.exists(mask_path):
        raise FileNotFoundError(f"Image exists but mask does not for tile {row.tile_id}")
    elif not os.path.exists(img_path) and os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask exists but image does not for tile {row.tile_id}")
    
    # check if the mask is empty
    poly = masks[masks.tile_id == tile_id].geometry.values
    if (poly is None or len(poly) == 0 or poly[0] is None):
        print(f"No mask found for tile {tile_id}, skipping image")
        return

    item = stac_reader.get_item_by_name(row.s2_tile_id, bbox=bounds)

    # read as stack
    stack = stac_reader.get_stack(
        items=item, 
        bands=bands,
        crop_to_bounds=False, 
        squeeze_time_dim=True,
        custom_point_and_buffer=[lon, lat, 10240],
        chunk_size=512,
    )

    # save image
    stac_reader.save_stack_as_geotiff(stack, img_path)

    # convert mask to same crs as image
    poly = poly.to_crs(stack.crs)
    mask_raster = rasterio.features.rasterize(poly, out_shape=(2048,2048), transform=stack.rio.transform())

    # save mask
    with rasterio.open(mask_path, 'w', driver='GTiff', 
                       height=2048, width=2048, count=1, dtype=np.uint8, crs=stack.crs, transform=stack.rio.transform()) as dst:
        dst.write(mask_raster, 1)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str, help="Path to save the output images")
    parser.add_argument("--limit", type=int, help="Limit the number of rows to process")
    args = parser.parse_args()
    output_path = args.output_path
    limit = args.limit

    # Load the dataset
    tiles = gpd.read_file(DATASET_FILTERED, layer="tiles")
    masks = gpd.read_file(DATASET_FILTERED, layer="polygons")

    # Set the limit to the length of the tiles dataframe if not provided
    if limit is None:
        limit = len(tiles)

    # Limit the number of rows according to the limit
    tiles = tiles.head(limit)

    # Get the train/val ratio
    train_tiles = tiles[tiles.split == "train"]
    val_tiles = tiles[tiles.split == "val"]
    test_tiles = tiles[tiles.split == "test"]
    train_ratio = len(train_tiles) / (len(train_tiles) + len(val_tiles) + len(test_tiles))

    print(f"Processing {len(train_tiles)} rows for train set")
    print(f"Processing {len(val_tiles)} rows for val set")
    print(f"Processing {len(test_tiles)} rows for test set")
    print(f"Train/Val ratio: {train_ratio}")

    # make the directories for train and val
    os.makedirs(output_path + "/train", exist_ok=True)
    os.makedirs(output_path + "/val", exist_ok=True)

    # Initialize the STAC reader
    api_url="https://planetarycomputer.microsoft.com/api/stac/v1"
    stac_reader = ReadSTAC(api_url)

    # Apply the function to each row in the train set
    train_tiles.progress_apply(lambda row: process_row(
        row=row, 
        masks=masks, 
        stac_reader=stac_reader, 
        bands=["B04", "B03", "B02"],
        output_path=output_path + "/train"
    ), axis=1)

    # Print the number of rows processed for train set
    print(f"Processed {len(train_tiles)} rows for train set")

    # Apply the function to each row in the val set
    val_tiles.progress_apply(lambda row: process_row(
        row=row, 
        masks=masks, 
        stac_reader=stac_reader, 
        bands=["B04", "B03", "B02"],
        output_path=output_path + "/val"
    ), axis=1)

    # Print the number of rows processed for val set
    print(f"Processed {len(val_tiles)} rows for val set")

    # Apply the function to each row in the test set
    test_tiles.progress_apply(lambda row: process_row(
        row=row, 
        masks=masks, 
        stac_reader=stac_reader, 
        bands=["B04", "B03", "B02"],
        output_path=output_path + "/test"
    ), axis=1)

    # Print the number of rows processed for val set
    print(f"Processed {len(test_tiles)} rows for test set")

    print("Checking for duplicates and removing them if necessary...")

    train_files = os.listdir(output_path + "/train/")
    val_files = os.listdir(output_path + "/val/")
    test_files = os.listdir(output_path + "/test/")

    # check for duplicate files in the training and validation sets
    duplicate_files_train = set([x for x in train_files if train_files.count(x) > 1])
    duplicate_files_val = set([x for x in val_files if val_files.count(x) > 1])
    duplicate_files_test = set([x for x in test_files if test_files.count(x) > 1])  

    if len(duplicate_files_train) > 0:
        for file in duplicate_files_train:
            os.remove(output_path + "/train/" + file)
        print(f"Removed {len(duplicate_files_train)} duplicate files from training set")
    
    if len(duplicate_files_val) > 0:
        for file in duplicate_files_val:
            os.remove(output_path + "/val/" + file)
        print(f"Removed {len(duplicate_files_val)} duplicate files from validation set")

    # # check if there are chips in the validation set that are also in the training set
    # train_val_set_files = set(train_files).intersection(set(val_files))
    # if len(train_val_set_files) > 0:
    #     train_val_ratio = len(train_files) / (len(val_files) + len(train_files))

    #     # if the ratio is less than 0.8, remove the duplicates from the validation set
    #     if train_val_ratio < train_ratio:
    #         for file in train_val_set_files:
    #             os.remove(output_path + "/val/" + file)
    #         print(f"Removed {len(train_val_set_files)} duplicate files from validation set")
    #     # if the ratio is greater than or equal to 0.8, remove the duplicates from the training set
    #     elif train_val_ratio >= train_ratio:
    #         for file in train_val_set_files:
    #             os.remove(output_path + "/train/" + file)
    #         print(f"Removed {len(train_val_set_files)} duplicate files from training set")

    # # recalculate the train/val ratio
    # train_files = os.listdir(output_path + "/train/")
    # val_files = os.listdir(output_path + "/val/")
    # train_val_ratio = len(train_files) / (len(val_files) + len(train_files))

    print("Finished processing images and masks")
    print(f"Train set: {len(train_files)/2} images")
    print(f"Val set: {len(val_files)/2} images")
    print(f"Test set: {len(test_files)/2} images")
