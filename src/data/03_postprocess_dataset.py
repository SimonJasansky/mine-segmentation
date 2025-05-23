# Script to postprocess the dataset and add bounding boxes for individual polygons
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import mapping, box, shape
import numpy as np
import shapely
import json
import os
import warnings; warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)

DATASET_RAW = "data/raw/mining_tiles_with_masks.gpkg"
DATASET_PROCESSED = "data/processed/mining_tiles_with_masks_and_bounding_boxes.gpkg"

def add_bounding_boxes(row):
    if row.geometry is None:
        return None
    
    # apply buffer
    polygons = row.geometry.buffer(0.0001)

    # create the unary union, i.e. merge touching polygons
    union = unary_union(polygons)

    if isinstance(union, shapely.geometry.multipolygon.MultiPolygon):
        # get list of bounding boxes
        bounding_boxes = [box(*geom.bounds) for geom in list(union.geoms)]
        
        # convert back to multipolygon
        bounding_boxes = shapely.geometry.MultiPolygon(bounding_boxes)
        
        return bounding_boxes
    else:
        # Create a box from the bounding box
        bounding_boxes = box(*union.bounds)

        # convert to polygon
        bounding_boxes = shapely.geometry.Polygon(bounding_boxes)

        return bounding_boxes
    
def remove_duplicates(tiles, maus, tang):

    len_before = len(tiles)
    duplicates_tile_id = tiles[tiles.tile_id.duplicated(keep="first")]
    duplicates_geom = tiles[tiles.geometry.duplicated(keep="first")]

    indices_to_remove = list(set((*duplicates_tile_id.index, *duplicates_geom.index)))
    tile_ids_to_remove = tiles.iloc[indices_to_remove].tile_id.unique().tolist()
    print(f"Found {len(indices_to_remove)} duplicates at indices {indices_to_remove}, having tile_ids {tile_ids_to_remove}. ")

    tiles = tiles.drop(indices_to_remove)
    maus = maus.drop(indices_to_remove)
    tang = tang.drop(indices_to_remove)

    # reset index
    tiles = tiles.reset_index(drop=True)
    maus = maus.reset_index(drop=True)
    tang = tang.reset_index(drop=True)

    len_after = len(tiles)
    print(f"Removed {len_before - len_after} duplicates")
    return tiles, maus, tang

if __name__ == '__main__':

    # Check if dataset exists
    if not os.path.exists(DATASET_RAW):
        raise FileNotFoundError(f"Dataset not found at {DATASET_RAW}")

    tiles = gpd.read_file(DATASET_RAW, layer="tiles")
    maus_polygons = gpd.read_file(DATASET_RAW, layer="maus_polygons")
    tang_polygons = gpd.read_file(DATASET_RAW, layer="tang_polygons")

    # convert tile_id to integer
    tiles["tile_id"] = tiles["tile_id"].astype(int)
    maus_polygons["tile_id"] = maus_polygons["tile_id"].astype(int)
    tang_polygons["tile_id"] = tang_polygons["tile_id"].astype(int)

    # Remove duplicates
    tiles, maus_polygons, tang_polygons = remove_duplicates(tiles, maus_polygons, tang_polygons)
    
    # Sanity checks
    assert len(tiles) == len(maus_polygons) == len(tang_polygons), "Number of tiles, maus and tang datasets must be equal"
    assert len(tiles) == len(tiles.tile_id.unique()), f"tile_id must be unique, with {len(tiles)} tiles, and {len(tiles.tile_id.unique())} unique tile_ids"
    assert tiles["tile_id"].equals(maus_polygons["tile_id"]), "tile_id must be the same in tiles and maus"
    assert tiles["tile_id"].equals(tang_polygons["tile_id"]), "tile_id must be the same in tiles and tang"

    # Add bounding boxes to the dataset
    maus_bboxes = maus_polygons.apply(add_bounding_boxes, axis=1)
    tang_bboxes = tang_polygons.apply(add_bounding_boxes, axis=1)

    # Create the geodataframes with the bounding boxes as geometry column 
    maus_bboxes_gdf = gpd.GeoDataFrame(geometry=maus_bboxes, crs=maus_polygons.crs)
    tang_bboxes_gdf = gpd.GeoDataFrame(geometry=tang_bboxes, crs=tang_polygons.crs)

    # add the tile id as a column in front of the geometry column
    maus_bboxes_gdf.insert(0, 'tile_id', maus_polygons['tile_id'])
    tang_bboxes_gdf.insert(0, 'tile_id', tang_polygons['tile_id'])

    # Create combined dataset based on preferred dataset.
    preferred_poly = []
    preferred_bbox = []

    for i in range(len(tiles)):
        if tiles.iloc[i,:]["preferred_dataset"] == "none":
            preferred_poly.append(None)
            preferred_bbox.append(None)
        elif tiles.iloc[i,:]["preferred_dataset"] == "maus":
            preferred_poly.append(maus_polygons.iloc[i,:].geometry)
            preferred_bbox.append(maus_bboxes_gdf.iloc[i,:].geometry)
        elif tiles.iloc[i,:]["preferred_dataset"] == "tang":
            preferred_poly.append(tang_polygons.iloc[i,:].geometry)
            preferred_bbox.append(tang_bboxes_gdf.iloc[i,:].geometry)
        else:
            pref_dataset = tiles.iloc[i,:]['preferred_dataset']
            raise ValueError(f"preferred_dataset must be either 'maus' or 'tang', or 'none', but got {pref_dataset}")
    
    preferred_polygons = gpd.GeoDataFrame(geometry=preferred_poly, crs=tiles.crs)
    preferred_bboxes = gpd.GeoDataFrame(geometry=preferred_bbox, crs=tiles.crs)

    # add the tile id as a column in front of the geometry column
    preferred_polygons.insert(0, 'tile_id', tiles['tile_id'])
    preferred_bboxes.insert(0, 'tile_id', tiles['tile_id'])

    # Add sanity checks before writing to file
    assert len(tiles) == len(maus_polygons) == len(tang_polygons) == len(maus_bboxes_gdf) == len(tang_bboxes_gdf) == len(preferred_polygons) == len(preferred_bboxes), "Number of rows must be equal"
    assert tiles["tile_id"].equals(maus_polygons["tile_id"]), "tile_id must be the same in tiles and maus"
    assert tiles["tile_id"].equals(tang_polygons["tile_id"]), "tile_id must be the same in tiles and tang"
    assert tiles["tile_id"].equals(maus_bboxes_gdf["tile_id"]), "tile_id must be the same in tiles and maus_bboxes_gdf"
    assert tiles["tile_id"].equals(tang_bboxes_gdf["tile_id"]), "tile_id must be the same in tiles and tang_bboxes_gdf"
    assert tiles["tile_id"].equals(preferred_polygons["tile_id"]), "tile_id must be the same in tiles and preferred_polygons"
    assert tiles["tile_id"].equals(preferred_bboxes["tile_id"]), "tile_id must be the same in tiles and preferred_bboxes"
    
    # Write the dataframes to geopackage with different layers
    tiles.to_file(DATASET_PROCESSED, layer="tiles", driver="GPKG")
    maus_polygons.to_file(DATASET_PROCESSED, layer="maus_polygons", driver="GPKG")
    tang_polygons.to_file(DATASET_PROCESSED, layer="tang_polygons", driver="GPKG")
    maus_bboxes_gdf.to_file(DATASET_PROCESSED, layer="maus_bboxes", driver="GPKG")
    tang_bboxes_gdf.to_file(DATASET_PROCESSED, layer="tang_bboxes", driver="GPKG")
    preferred_polygons.to_file(DATASET_PROCESSED, layer="preferred_polygons", driver="GPKG")
    preferred_bboxes.to_file(DATASET_PROCESSED, layer="preferred_bboxes", driver="GPKG")

    print(f"Data with {len(tiles)} records successfully written to {DATASET_PROCESSED}")