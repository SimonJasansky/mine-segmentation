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
    # apply buffer 
    row.geometry = row.geometry.buffer(0.0001)

    # create the unary union, i.e. merge touching polygons
    union = unary_union(row.geometry)

    if isinstance(union, shapely.geometry.multipolygon.MultiPolygon):
        # get list of bounding boxes
        bounding_boxes = [box(*geom.bounds) for geom in list(union.geoms)]
        
        # convert back to multipolygon
        bounding_boxes = shapely.geometry.MultiPolygon(bounding_boxes)
        
        # convert to GeoJSON string
        geojson_str = json.dumps(mapping(bounding_boxes))

        return geojson_str
    else:
        # Create a box from the bounding box
        bounding_box = box(*union.bounds)

        # Convert the bounding box to a GeoJSON string
        geojson_str = json.dumps(mapping(bounding_box))

        return geojson_str

if __name__ == '__main__':

    # Check if dataset exists
    if not os.path.exists(DATASET_RAW):
        raise FileNotFoundError(f"Dataset not found at {DATASET_RAW}")

    data = gpd.read_file(DATASET_RAW)

    # Add bounding boxes to the dataset
    data["bounding_box"] = data.apply(add_bounding_boxes, axis=1)

    # Split the data set into three layers to be added to the geopackage file
    tiles = data.loc[:, ["tile_id", "sentinel_2_id", "source_dataset", "timestamp", "tile_bbox"]]
    masks = data.loc[:, ["tile_id", "geometry"]]
    bounding_boxes = data.loc[:, ["tile_id", "bounding_box"]]

    # convert columns to valid geometry
    tiles["tile_bbox"] = tiles["tile_bbox"].apply(lambda x: shape(json.loads(x)))
    bounding_boxes["bounding_box"] = bounding_boxes["bounding_box"].apply(lambda x: shape(json.loads(x)))

    # rename columns
    tiles = tiles.rename(columns={"tile_bbox": "geometry"})
    bounding_boxes = bounding_boxes.rename(columns={"bounding_box": "geometry"})

    # convert to geodataframes
    tiles = gpd.GeoDataFrame(tiles, geometry="geometry")
    bounding_boxes = gpd.GeoDataFrame(bounding_boxes, geometry="geometry")

    # Set the crs
    tiles.crs = masks.crs
    bounding_boxes.crs = masks.crs

    # Write the dataframes to geopackage with different layers
    tiles.to_file(DATASET_PROCESSED, layer="tiles", driver="GPKG")
    masks.to_file(DATASET_PROCESSED, layer="masks", driver="GPKG")
    bounding_boxes.to_file(DATASET_PROCESSED, layer="bounding_boxes", driver="GPKG")

    print(f"Data successfully written to {DATASET_PROCESSED}")