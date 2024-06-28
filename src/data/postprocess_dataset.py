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
        return None
    
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

        return bounding_boxes

if __name__ == '__main__':

    # Check if dataset exists
    if not os.path.exists(DATASET_RAW):
        raise FileNotFoundError(f"Dataset not found at {DATASET_RAW}")

    maus = gpd.read_file(DATASET_RAW, layer="maus_polygons")
    tang = gpd.read_file(DATASET_RAW, layer="tang_polygons")

    # Add bounding boxes to the dataset
    maus_bboxes = maus.apply(add_bounding_boxes, axis=1)
    tang_bboxes = tang.apply(add_bounding_boxes, axis=1)

    # Create the geodataframes with the bounding boxes
    maus_bboxes_gdf = gpd.GeoDataFrame(maus_bboxes, geometry="geometry", crs=maus.crs)
    tang_bboxes_gdf = gpd.GeoDataFrame(tang_bboxes, geometry="geometry", crs=tang.crs)

    # Write the dataframes to geopackage with different layers
    maus_bboxes_gdf.to_file(DATASET_PROCESSED, layer="maus_bboxes", driver="GPKG")
    tang_bboxes_gdf.to_file(DATASET_PROCESSED, layer="tang_bboxes", driver="GPKG")

    print(f"Data successfully written to {DATASET_PROCESSED}")