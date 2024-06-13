import rasterio
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm


MAUS_POLYGONS = "data/external/maus_mining_polygons.gpkg"
MAUS_AREA_RASTER = "data/external/maus_mining_raster.tif"
TANG_POLYGONS = "data/external/tang_mining_polygons/74548_mine_polygons/74548_projected.shp"

import numpy as np
from tqdm import tqdm
src = rasterio.open(MAUS_AREA_RASTER)
array = src.read(1)

# Get the transformation matrix
transform = src.transform

# Create an empty list to store the bounding boxes
bounding_boxes = []
mining_area = []

# Iterate over the pixels in the raster
# only record bounding box if they have over 0.5 square km of mining area (out of a total area per square of 78.41 sq.km)
for x in tqdm(range(src.width)):
    for y in range(src.height):
        if array[y, x] > 0.5:
            # Get the pixel's bounding box
            # The bounding box is defined by the pixel's top-left and bottom-right corners
            top_left = transform * (x, y)
            bottom_right = transform * (x + 1, y + 1)
            bounding_box = [top_left[0], bottom_right[1], bottom_right[0], top_left[1]]
            
            # Add the bounding box to the list
            bounding_boxes.append(bounding_box)

            # add the mining area to the list
            mining_area.append(array[y, x])


# Create a GeoDataFrame from the bounding boxes and the area
gdf = gpd.GeoDataFrame(geometry=[box(*bbox) for bbox in bounding_boxes], crs="EPSG:4326")
gdf["mining_area"] = mining_area

# save the bounding boxes as a geopackage file
gdf.to_file("/workspaces/mine-segmentation/data/interim/mining_areas.gpkg", driver="GPKG")