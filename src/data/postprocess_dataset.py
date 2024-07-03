# Script to postprocess the dataset and add bounding boxes for individual polygons
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import mapping, box, shape
import shapely
import json
import os

DATASET_RAW = "data/raw/mining_tiles_with_masks.gpkg"
DATASET_PROCESSED = "data/processed/mining_tiles_with_masks_and_bounding_boxes.gpkg"

def add_bounding_boxes(row):
    if row.geometry is None:
        return shapely.geometry.Polygon()
    
    # apply buffer 
    row.geometry = row.geometry.buffer(0.0001)

    # create the unary union, i.e. merge touching polygons
    union = unary_union(row.geometry)

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

if __name__ == '__main__':

    # Check if dataset exists
    if not os.path.exists(DATASET_RAW):
        raise FileNotFoundError(f"Dataset not found at {DATASET_RAW}")

    tiles = gpd.read_file(DATASET_RAW, layer="tiles")
    maus = gpd.read_file(DATASET_RAW, layer="maus_polygons")
    tang = gpd.read_file(DATASET_RAW, layer="tang_polygons")

    # Add bounding boxes to the dataset
    maus_bboxes = maus.apply(add_bounding_boxes, axis=1)
    tang_bboxes = tang.apply(add_bounding_boxes, axis=1)

    # Create the geodataframes with the bounding boxes as geometry column 
    maus_bboxes_gdf = gpd.GeoDataFrame(geometry=maus_bboxes, crs=maus.crs)
    tang_bboxes_gdf = gpd.GeoDataFrame(geometry=tang_bboxes, crs=tang.crs)

    # add the tile id as a column in front of the geometry column
    maus_bboxes_gdf.insert(0, 'tile_id', maus['tile_id'])
    tang_bboxes_gdf.insert(0, 'tile_id', tang['tile_id'])

    # copy Dataset_RAW to location of DATASET_PROCESSED, and rename it
    os.system(f"cp {DATASET_RAW} {DATASET_PROCESSED}")

    # Create combined dataset based on preferred dataset. 
    preferred_poly = []
    preferred_bbox = []

    for i in range(len(tiles)):
        if tiles.iloc[i,:]["preferred_dataset"] == "none":
            continue
        if tiles.iloc[i,:]["preferred_dataset"] == "maus": 
            preferred_poly.append(maus.iloc[i,:].geometry)
            preferred_bbox.append(maus_bboxes_gdf.iloc[i,:].geometry)
        elif tiles.iloc[i,:]["preferred_dataset"] == "tang":
            preferred_poly.append(tang.iloc[i,:].geometry)
            preferred_bbox.append(tang_bboxes_gdf.iloc[i,:].geometry)
        else:
            raise ValueError("preferred_dataset must be either 'maus' or 'tang', or 'none'")
        
    preferred_polygons = gpd.GeoDataFrame(geometry=preferred_poly, crs=tiles.crs)
    preferred_bboxes = gpd.GeoDataFrame(geometry=preferred_bbox, crs=tiles.crs)

    # add the tile id as a column in front of the geometry column
    preferred_polygons.insert(0, 'tile_id', tiles['tile_id'])
    preferred_bboxes.insert(0, 'tile_id', tiles['tile_id'])
    
    # Write the dataframes to geopackage with different layers
    maus_bboxes_gdf.to_file(DATASET_PROCESSED, layer="maus_bboxes", driver="GPKG")
    tang_bboxes_gdf.to_file(DATASET_PROCESSED, layer="tang_bboxes", driver="GPKG")
    preferred_polygons.to_file(DATASET_PROCESSED, layer="preferred_polygons", driver="GPKG")
    preferred_bboxes.to_file(DATASET_PROCESSED, layer="preferred_bboxes", driver="GPKG")

    print(f"Data successfully written to {DATASET_PROCESSED}")