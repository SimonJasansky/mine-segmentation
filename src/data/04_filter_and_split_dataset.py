r"""

Example:

```bash
python src/data/04_filter_and_split_dataset.py preferred_polygons --val_ratio 0.18 --test_ratio 0.07 --only_valid_surface_mines
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
from sklearn.utils import resample

DATASET_PROCESSED = "data/processed/mining_tiles_with_masks_and_bounding_boxes.gpkg"

def split_data(tiles, val_ratio=0.15, test_ratio=0.1):
    """
    Split the data into train, validation, and test sets based on the overlap of the polygons.

    Args:
        tiles (GeoDataFrame): The input data.
        val_ratio (float): The size of the validation set.
        test_ratio (float): The size of the test set.

    Returns:
        GeoDataFrame: The input data with an additional column 'split' that contains the split type.
    """
    np.random.seed(1234)  # Set the seed for reproducibility
    print("Splitting valid surface tiles into train, validation, and test sets...")
    
    # calculate the number of test tiles
    n_test = int(len(tiles) * test_ratio)

    # for each tile, check with how many other tiles it overlaps
    tiles = tiles.copy()
    tiles["overlaps"] = tiles["geometry"].apply(lambda x: tiles["geometry"].apply(lambda y: x.overlaps(y)).sum())

    if len(tiles[tiles["overlaps"] == 0]) < n_test:
        raise ValueError(f"Number of tiles that do not overlap with any other tiles, and that have both maus and tang polygons ({len(tiles[tiles['overlaps'] == 0])}) is less than the number of test tiles ({n_test}).")

    # add a column that combines the minetype1, minetype2, and source_dataset columns, that can be used for stratification
    strat_cols = ["minetype1", "minetype2", "preferred_dataset"]
    tiles['stratify_col'] = tiles[strat_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    # assign the test split directly only to tiles that overlap with no other tiles    
    test_tiles = tiles[tiles["overlaps"] == 0]
    print(f"{len(test_tiles)} tiles do not overlap with any other tiles.")

    # # and that have both maus and tang polygons (for validation purposes)
    # test_tiles = test_tiles[test_tiles["source_dataset"] == "both"]
    # print(f"{len(test_tiles)} tiles do not overlap with any other tiles and have both maus and tang polygons.")

    # take a stratified sample
    strat_dist = tiles['stratify_col'].value_counts(normalize=True)

    test_tiles_sample = pd.DataFrame()

    for strat_value, proportion in strat_dist.items():
        # print(f"Stratified sample for {strat_value}: {proportion} proportion")
        strat_sample = test_tiles[test_tiles['stratify_col'] == strat_value]
        n_samples = int(proportion * n_test)  # Number of samples to draw
        # print(f"Stratified sample for {strat_value}: {n_samples} samples")
        # print(f"length of strat_sample: {len(strat_sample)}")
        if n_samples > 0:
            # if possible, take tiles where both maus and tang polygons are available
            if len(strat_sample[strat_sample["source_dataset"] == "both"]) >= n_samples:
                strat_sample = strat_sample[strat_sample["source_dataset"] == "both"]

            if len(strat_sample) < n_samples:
                print(f"Stratified sample for {strat_value}: {len(strat_sample)} samples is less than the required number of samples {n_samples}.")
                print(f"Taking all the available samples for {strat_value} ({len(strat_sample)} samples).")
                test_tiles_sample = pd.concat([test_tiles_sample, strat_sample])
            else:
                strat_sample = strat_sample.sample(n=n_samples)
                test_tiles_sample = pd.concat([test_tiles_sample, strat_sample])

    # remove the test tiles from the remaining dataset, used for validation and training
    tiles = tiles.drop(test_tiles_sample.index)

    # reset index
    test_tiles_sample = test_tiles_sample.reset_index(drop=True)

    # assign the split to the test tiles
    test_tiles_sample["split"] = "test"
    print(f"Out of {len(test_tiles)} tiles that do not overlap with any other tiles and that have both maus and tang polygons, {len(test_tiles_sample)} are assigned to the test set.")

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
    tiles = pd.concat([tiles, test_tiles_sample])

    return tiles.drop(columns=["overlaps", "overlap_group", "stratify_col"])


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
    if not os.path.exists(DATASET_PROCESSED):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PROCESSED}")

    tiles_original = gpd.read_file(DATASET_PROCESSED, layer="tiles")
    tiles_original["tile_id"] = tiles_original["tile_id"].astype(int)

    # if the split column already exists, remove it
    if "split" in tiles_original.columns:
        tiles_original = tiles_original.drop(columns=["split"])

    if only_valid_surface_mines:
        len_before = len(tiles_original)
        tiles = tiles_original[(tiles_original["source_dataset"] != "rejected") & (tiles_original["minetype1"].isin(["Surface", "Placer"]))]
        len_after = len(tiles)
        print(f"Filtered out {len_before - len_after} rejected tiles and non-surface mines")
    else:
        tiles = tiles_original

    # filter the polygons according to the tile_ids in the filtered tiles
    tile_ids = tiles.tile_id.unique()

    # Split the data into train, validation, and test sets
    tiles = split_data(tiles, val_ratio=val_ratio, test_ratio=test_ratio)

    # make sure both tiles and polygons are in the same order
    tiles["tile_id"] = tiles["tile_id"].astype(int)
    tiles = tiles.sort_values("tile_id")

    # sanity checks
    assert np.all(tiles.split.notna())
    assert len(tiles) == len(tiles.tile_id.unique())

    # merge the split column to tiles_original
    tiles_original = tiles_original.merge(tiles[["tile_id", "split"]], on="tile_id", how="left")

    # Save the output
    tiles_original.to_file(DATASET_PROCESSED, layer="tiles", driver="GPKG")

    # print the train/val/test split ratio
    train_tiles = tiles[tiles.split == "train"]
    val_tiles = tiles[tiles.split == "val"]
    test_tiles = tiles[tiles.split == "test"]
    print(f"Train/Val/Test split: {len(train_tiles)}/{len(val_tiles)}/{len(test_tiles)}, in relative terms: {len(train_tiles) / len(tiles):.2f}/{len(val_tiles) / len(tiles):.2f}/{len(test_tiles) / len(tiles):.2f}")
    print("-" * 80)
    print("\nCheck the distribution of the stratified test set:\n")

    # check if the stratified test set has the same distribution as the original dataset
    print("Distribution of Stratified Test Set:")
    print(test_tiles.groupby(["minetype1", "minetype2", "preferred_dataset"]).size() / len(test_tiles) * 100)
    print("\nDistribution of Original Dataset:")
    print(tiles.groupby(["minetype1", "minetype2", "preferred_dataset"]).size() / len(tiles) * 100)

    # check for the single variables
    print("\nDistribution of minetype1 in Stratified Test Set:")
    print(test_tiles["minetype1"].value_counts(normalize=True) * 100)
    print("\nDistribution of minetype1 in Original Dataset:")
    print(tiles["minetype1"].value_counts(normalize=True) * 100)

    print("\nDistribution of minetype2 in Stratified Test Set:")
    print(test_tiles["minetype2"].value_counts(normalize=True) * 100)
    print("\nDistribution of minetype2 in Original Dataset:")
    print(tiles["minetype2"].value_counts(normalize=True) * 100)

    print("\nDistribution of preferred_dataset in Stratified Test Set:")
    print(test_tiles["preferred_dataset"].value_counts(normalize=True) * 100)
    print("\nDistribution of preferred_dataset in Original Dataset:")
    print(tiles["preferred_dataset"].value_counts(normalize=True) * 100)

    print(f"Saved dataset with the train/val/test split column, containinig {len(tiles)} valid records, to {DATASET_PROCESSED}")


