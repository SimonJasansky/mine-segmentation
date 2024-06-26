import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import rasterio
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import transform
import shapely
from tqdm import tqdm
tqdm.pandas()
import pyproj
from functools import partial

MAUS_POLYGONS = "data/external/maus_mining_polygons.gpkg"
MAUS_AREA_RASTER = "data/external/maus_mining_raster.tif"
MAUS_AREA_RASTER_DOWNSAMPLED = "data/external/maus_mining_raster_downsampled.tif"
TANG_POLYGONS = "data/external/tang_mining_polygons/74548_mine_polygons/74548_projected.shp"

# Create the downsampled raster
def resample_geotiff(source_path, dest_path, resampling_factor): 
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(source_path) as dataset:

        # resample data to target shape using upscale_factor
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * resampling_factor),
                int(dataset.width * resampling_factor)
            ),
            resampling=Resampling.average
        )

        print('Shape before resample:', dataset.shape)
        print('Shape after resample:', data.shape[1:])

        # scale image transform
        dst_transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        print('Transform before resample:\n', dataset.transform, '\n')
        print('Transform after resample:\n', dst_transform)

        # Write outputs
        # set properties for output
        dst_kwargs = dataset.meta.copy()
        dst_kwargs.update(
            {
                "crs": dataset.crs,
                "transform": dst_transform,
                "width": data.shape[-1],
                "height": data.shape[-2],
                "nodata": 0,  
            }
        )

        with rasterio.open(dest_path, "w", **dst_kwargs) as dst:
            # iterate through bands
            for i in range(data.shape[0]):
                dst.write(data[i], i+1)

resample_geotiff(MAUS_AREA_RASTER, MAUS_AREA_RASTER_DOWNSAMPLED, 1/2)

# Open the downsampled raster
src = rasterio.open(MAUS_AREA_RASTER_DOWNSAMPLED)
array = src.read(1)

# print the total number of tiles
print(f"Number of total tiles: {src.width * src.height}")
# Get the original number of tiles with mining area larger than 0
print(f"Number of original mining area tiles: {len(array[array > 0])}")

# Get the transformation matrix
transform_matrix = src.transform

# Create an empty list to store the bounding boxes
bounding_boxes = []
mining_area = []

# Iterate over the pixels in the raster
# only record bounding box if they have over 2 square km of mining area (out of a total area per square of 4 * 78.41 sq.km)
for x in tqdm(range(src.width)):
    for y in range(src.height):
        if array[y, x] > 2:
            # Get the pixel's bounding box
            # The bounding box is defined by the pixel's top-left and bottom-right corners
            top_left = transform_matrix * (x, y)
            bottom_right = transform_matrix * (x + 1, y + 1)
            bounding_box = [top_left[0], bottom_right[1], bottom_right[0], top_left[1]]
            
            # Add the bounding box to the list
            bounding_boxes.append(bounding_box)

            # add the mining area to the list
            mining_area.append(array[y, x])

# Create a GeoDataFrame from the bounding boxes and the area
gdf = gpd.GeoDataFrame(geometry=[box(*bbox) for bbox in bounding_boxes], crs="EPSG:4326")

# # for each tile, get the centroid
centroids = gdf["geometry"].to_crs("EPSG:4326").centroid
gdf["centroid"] = gdf.centroid

print(gdf.head())

def add_bbox(row): 
    point = row.centroid

    # Define the projection to UTM (Universal Transverse Mercator)
    # Find UTM zone for the centroid of the polygon for more accuracy
    utm_zone = int((point.x + 180) / 6) + 1
    crs_proj = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84', preserve_units=False)

    # Define transformations from WGS84 to UTM and back
    project_to_utm = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'), crs_proj)
    project_to_wgs84 = partial(pyproj.transform, crs_proj, pyproj.Proj(init='epsg:4326'))

    # Transform the polygon to the UTM projection
    point = transform(project_to_utm, point)

    # calculate the bbox using the buffer
    buffer = 10240
    bbox = (point.x - buffer, point.y - buffer, point.x + buffer, point.y + buffer)

    # convert bbox to polygon
    bbox = shapely.geometry.box(*bbox)

    # Transform the polygon back to WGS84
    bbox = transform(project_to_wgs84, bbox)

    return bbox

# add the bounding box to the geodataframe
gdf["bbox"] = gdf.progress_apply(add_bbox, axis=1)

# convert to string
gdf["centroid"] = gdf["centroid"].to_wkt()
gdf["bbox"] = gdf["bbox"].to_wkt()

# add the mining area to the geodataframe
gdf["mining_area"] = mining_area

print(gdf.head())
print(f"Number of mining area tiles: {len(gdf)}")

# save the bounding boxes as a geopackage file
gdf.to_file("data/interim/mining_areas.gpkg", driver="GPKG")

print("Successfully saved mining area tiles.")