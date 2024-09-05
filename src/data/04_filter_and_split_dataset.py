r"""

Example:

```bash
python src/data/filter_and_split_dataset.py preferred_polygons --val_ratio 0.2 --test_ratio 0.1 --only_valid_surface_mines
```

"""
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import mapping, box, shape
import numpy as np
import pandas as pd
import shapely
import json
import os
import argparse
import warnings; warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
from tqdm import tqdm
import networkx as nx

DATASET_IN = "data/processed/mining_tiles_with_masks_and_bounding_boxes.gpkg"
DATASET_OUT = "data/processed/mining_tiles_with_masks_and_bounding_boxes_filtered.gpkg"
def split_data(tiles, val_ratio=0.2, test_ratio=0.1):
    """
    Split the data into train, validation, and test sets based on the overlap of the polygons.

    Args:
        tiles (GeoDataFrame): The input data.
        val_ratio (float): The size of the validation set.
        test_ratio (float): The size of the test set.

    Returns:
        GeoDataFrame: The input data with an additional column 'split' that contains the split type.
    """
    print("Splitting valid surface tiles into train, validation, and test sets...")
    
    np.random.seed(42)  # Set the seed for reproducibility
    
    n_test = int(len(tiles) * test_ratio)

    # for each tile, check with how many other tiles it overlaps
    tiles["overlaps"] = tiles["geometry"].apply(lambda x: tiles["geometry"].apply(lambda y: x.overlaps(y)).sum())

    if len(tiles[tiles["overlaps"] == 0]) < n_test:
        raise ValueError(f"Number of tiles that do not overlap with any other tiles ({len(tiles[tiles['overlaps'] == 0])}) is less than the number of test tiles ({n_test}).")

    # assign the test split directly only to tiles that overlap with no other tiles
    test_tiles = tiles[tiles["overlaps"] == 0].sample(n_test)
    test_tiles["split"] = "test"
    print(f"Out of {len(tiles[tiles['overlaps'] == 0])} tiles that do not overlap with any other tiles, {len(test_tiles)} are assigned to the test set.")
    
    # remove the test tiles from the dataset
    tiles = tiles.drop(test_tiles.index)

    print(f"Creating graph for {len(tiles)} tiles...")

    # Step 1: Create a graph
    G = nx.Graph()

    # Step 2: Add nodes
    for idx, geom in tiles.iterrows():
        G.add_node(idx, geometry=geom.geometry)

    # Step 3: Add edges for overlapping or touching geometries
    for i, geom1 in tqdm(tiles.iterrows(), total=len(tiles), desc="Adding edges"):
        for j, geom2 in tiles.iterrows():
            if i != j and (geom1.geometry.overlaps(geom2.geometry) or geom1.geometry.touches(geom2.geometry)):
                G.add_edge(i, j)

    # Step 4: Find connected components
    connected_components = list(nx.connected_components(G))

    # Step 5: Assign group IDs
    group_id = 0
    tiles['overlap_group'] = -1
    for component in connected_components:
        for idx in component:
            tiles.at[idx, 'overlap_group'] = group_id
        group_id += 1

    print(f"Number of connected components in the graph: {len(connected_components)}")

    # Step 6: Split the data into train, val, and test sets
    unique_groups = tiles['overlap_group'].unique()
    np.random.shuffle(unique_groups)

    n_val = int((len(unique_groups) + n_test) * val_ratio)

    val_groups = unique_groups[:n_val]
    tiles['split'] = 'train'
    tiles.loc[tiles['overlap_group'].isin(val_groups), 'split'] = 'val'

    # combine the test and validation dataframes
    tiles = pd.concat([tiles, test_tiles])

    return tiles.drop(columns=["overlaps", "overlap_group"])


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("polygon_layer", type=str, help="Name of the polygon layer in the dataset")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of data to use for validation (default: 0.2)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of data to use for testing (default: 0.1)")
    parser.add_argument("--only_valid_surface_mines", action="store_true", help="Only process valid surface mines, and ignore rejected tiles, and and tiles with brine & evaporation ponds")
    args = parser.parse_args()
    polygon_layer = args.polygon_layer
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    only_valid_surface_mines = args.only_valid_surface_mines

    # Check if dataset exists
    if not os.path.exists(DATASET_IN):
        raise FileNotFoundError(f"Dataset not found at {DATASET_IN}")

    tiles = gpd.read_file(DATASET_IN, layer="tiles")
    masks = gpd.read_file(DATASET_IN, layer=polygon_layer)

    if only_valid_surface_mines:
        len_before = len(tiles)
        tiles = tiles[(tiles["source_dataset"] != "rejected") & (tiles["minetype1"].isin(["Surface", "Placer"]))]
        len_after = len(tiles)
        print(f"Filtered out {len_before - len_after} rejected tiles and non-surface mines")

    # filter the polygons according to the tile_ids in the filtered tiles
    tile_ids = tiles.tile_id.unique()
    masks = masks[masks.tile_id.isin(tile_ids)]

    # Split the data into train, validation, and test sets
    tiles = split_data(tiles, val_ratio=val_ratio, test_ratio=test_ratio)

    # make sure both tiles and polygons are in the same order
    tiles["tile_id"] = tiles["tile_id"].astype(int)
    masks["tile_id"] = masks["tile_id"].astype(int)
    tiles = tiles.sort_values("tile_id")
    masks = masks.sort_values("tile_id")

    # sanity checks
    assert np.all(tiles.tile_id == masks.tile_id)
    assert np.all(tiles.split.notna())
    assert len(tiles) == len(tiles.tile_id.unique())
    assert len(masks) == len(masks.tile_id.unique())

    # Save the output
    tiles.to_file(DATASET_OUT, layer="tiles", driver="GPKG")
    masks.to_file(DATASET_OUT, layer="polygons", driver="GPKG")

    # print the train/val/test split ratio
    train_tiles = tiles[tiles.split == "train"]
    val_tiles = tiles[tiles.split == "val"]
    test_tiles = tiles[tiles.split == "test"]
    print(f"Train/Val/Test split: {len(train_tiles)}/{len(val_tiles)}/{len(test_tiles)}")

    print(f"Saved filtered dataset, containinig {len(tiles)} records, to {DATASET_OUT}")
