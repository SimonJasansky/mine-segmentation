r"""

Example:

```bash
python src/data/persist_pixels_masks.py data/processed/files preferred_polygons --limit 25
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

DATASET_PROCESSED = "data/processed/mining_tiles_with_masks_and_bounding_boxes.gpkg"

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

    # persist to disk
    stac_reader.save_stack_as_geotiff(stack, output_path + f"/{row.tile_id}_{row.s2_tile_id}_img.tif")

    # read the mask 
    tile_id = row.tile_id
    poly = masks[masks.tile_id == tile_id].geometry.values

    # convert to same crs as stack
    poly = poly.to_crs(stack.crs)

    mask_raster = rasterio.features.rasterize(poly, out_shape=(2048,2048), transform=stack.rio.transform())

    # save the mask
    with rasterio.open(output_path + f"/{row.tile_id}_{row.s2_tile_id}_mask.tif", 'w', driver='GTiff', 
                       height=2048, width=2048, count=1, dtype=np.uint8, crs=stack.crs, transform=stack.rio.transform()) as dst:
        dst.write(mask_raster, 1)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str, help="Path to save the output images")
    parser.add_argument("polygon_layer", type=str, help="Name of the polygon layer in the dataset")
    parser.add_argument("--limit", type=int, help="Limit the number of rows to process")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training (default: 0.8)")
    args = parser.parse_args()
    output_path = args.output_path
    polygon_layer = args.polygon_layer
    limit = args.limit
    train_ratio = args.train_ratio

    # Load the dataset
    tiles = gpd.read_file(DATASET_PROCESSED, layer="tiles")
    masks = gpd.read_file(DATASET_PROCESSED, layer=polygon_layer)

    # Set the limit to the length of the tiles dataframe if not provided
    if limit is None:
        limit = len(tiles)

    # Limit the number of rows according to the limit
    tiles = tiles.head(limit)

    # Split the data into train and val
    train_size = int(train_ratio * len(tiles))
    train_tiles = tiles[:train_size]
    val_tiles = tiles[train_size:]

    print(f"Processing {len(train_tiles)} rows for train set")
    print(f"Processing {len(val_tiles)} rows for val set")

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

