import math
import requests
import subprocess
import numpy as np
import leafmap
from osgeo import gdal, osr
from pyproj import Transformer, CRS
from pathlib import Path
from pystac_client import Client
from shapely.geometry import Point, mapping, box
from pystac.extensions.eo import EOExtension as eo


class ReadSTAC():
    """Read STAC API and obtain a TIFF of a desired location within the desired timeframe"""
    
    def __init__(
        self, 
        api_url = "https://planetarycomputer.microsoft.com/api/stac/v1": str,
        collection = "sentinel-2-l2a": str,
        location = [-6.054643555224412, -50.18255000697929]: list,
        buffer = 10: int,
        timerange = "2020-06-01/2020-12-31": str,
        max_cloud_cover = 10: int
    ):
        self.api_url = api_url
        self.location = location
        self.timerange = timerange

        # Approximate degrees for 10 km buffer at the equator (1 degree of lat ~ 111 km at the equator)
        buffer_lat_deg = buffer / 111  # Buffer in degrees latitude, which is approximately constant
        
        # Approximate degrees for 10 km buffer in longitude (varies with latitude)
        # Use cosine of the latitude to adjust for the earth's curvature
        buffer_lon_deg = buffer / (111 * abs(math.cos(math.radians(point.y))))

        # Create a box around the point using the approximate buffer
        bbox = box(point.x - buffer_lon_deg, point.y - buffer_lat_deg,
                   point.x + buffer_lon_deg, point.y + buffer_lat_deg)

        self.bbox = bbox

    def read_stac():
        
        # Convert the buffered box to GeoJSON
        buffered_box_geojson = mapping(self.bbox)

        catalog = Client.open(
            self.api_url,
            modifier = (
                planetary_computer.sign_inplace 
                if self.api_url == "https://planetarycomputer.microsoft.com/api/stac/v1" 
                else None
            ),
        )
        
        search = catalog.search(
            collections=[self.collection],
            bbox=self.bbox,
            datetime=self.timerange,
            limit=50,
            query={"eo:cloud_cover": {"lt": 10}} # Less than 10% cloud cover
        )

        items = search.item_collection()
        
        # df = geopandas.GeoDataFrame.from_features(items.to_dict(), crs="epsg:4326")
        # print(f"{len(items)} Items found.")
        # df.head
        
        # least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)
        # print(
        #    f"Choosing {least_cloudy_item.id} from {least_cloudy_item.datetime.date()}"
        #    f" with {eo.ext(least_cloudy_item).cloud_cover}% cloud cover"
        # )



# Convert to list if not already (depends on the pystac_client version)
items = list(items)

# Dictionary to hold the most recent item per orbit point (tile)
most_recent_per_orbit = {}

# Iterate over the items to populate the dictionary
for item in items:
    # Each tile (orbit point) can be identified by its 'sentinel:tile_id' property
    tile_id = item.properties.get('sentinel:tile_id')

    # If this tile hasn't been seen before or this item is more recent, store it
    if tile_id not in most_recent_per_orbit or item.datetime > most_recent_per_orbit[tile_id].datetime:
        most_recent_per_orbit[tile_id] = item

# Now you have the most recent item per orbit point
most_recent_items = list(most_recent_per_orbit.values())

# Assuming you have a list of most recent items in most_recent_items
# Bands of interest (B02: Blue, B03: Green, B04: Red, B08: Near-Infrared)
bands = ['B02', 'B03', 'B04', 'B08']

# Directory to save the images
output_dir = Path("./images")
output_dir.mkdir(parents=True, exist_ok=True)

# Update band names based on the provided keys
bands_to_download = {
    'blue': 'blue',      # Corresponds to 'B02'
    'green': 'green',    # Corresponds to 'B03'
    'red': 'red',        # Corresponds to 'B04'
    'nir08': 'nir08',    # Corresponds to 'B08'
    # Add other bands you are interested in here
}

def download_band(item, band_name, band_key):
    # Generate the full path where the file will be saved
    file_path = output_dir / f"{item.id}_{band_name}.tif"

    # Check if the file already exists
    if file_path.exists():
        print(f"File {file_path} already exists, skipping download.")
        return

    # Get the asset/href for the specified band
    asset = item.assets[band_key]
    
    # Download the file
    response = requests.get(asset.href)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {file_path}")
    else:
        print(f"Failed to download {asset.href}")

# Loop through items and download the specified bands
for item in most_recent_items:
    for band_name, band_key in bands_to_download.items():
        download_band(item, band_name, band_key)

# Function to create and crop VRT for given bands
def create_vrt(item_id, band_files, vrt_suffix):
    # Create VRT for the bands
    vrt_filename = output_dir / f"{item_id}_{vrt_suffix}.vrt"
    gdal.BuildVRT(str(vrt_filename), band_files, separate=True, options=['COMPRESS=DEFLATE'])
    print(f"Created VRT: {vrt_filename}")

# Create VRTs for each set of items' bands
for item in most_recent_items:
    item_id = item.id
    # Paths to the band files
    red_band_path = output_dir / f"{item_id}_red.tif"
    green_band_path = output_dir / f"{item_id}_green.tif"
    blue_band_path = output_dir / f"{item_id}_blue.tif"
    nir_band_path = output_dir / f"{item_id}_nir08.tif"

    # Check if all relevant band files exist before creating the VRT
    if all(Path(band).exists() for band in [red_band_path, green_band_path, blue_band_path]):
        true_color_bands = [str(red_band_path), str(green_band_path), str(blue_band_path)]
        create_vrt(item_id, true_color_bands, 'true_color')

    if all(Path(band).exists() for band in [nir_band_path, green_band_path, blue_band_path]):
        false_color_bands = [str(nir_band_path), str(green_band_path), str(blue_band_path)]
        create_vrt(item_id, false_color_bands, 'false_color')

# enhance composites 
def stretch_contrast_with_subsampling(vrt_path, output_path, crop_bbox, subsample=10, compression='DEFLATE'):
    
    # Get the source dataset's projection from the first band file
    src_ds = gdal.Open(str(vrt_path))
    src_wkt = src_ds.GetProjection()
    src_srs = osr.SpatialReference(wkt=src_wkt)

    # Convert the osr SpatialReference to a pyproj CRS object
    crs_src = CRS.from_wkt(src_wkt)

    # Create transformer to project crop_bbox into source's CRS
    transformer_to_src_crs = Transformer.from_crs("epsg:4326", crs_src, always_xy=True)

    # Get bounds of crop_bbox in source's CRS
    minx, miny, maxx, maxy = crop_bbox.bounds
    minx, miny = transformer_to_src_crs.transform(minx, miny)
    maxx, maxy = transformer_to_src_crs.transform(maxx, maxy)

    # Crop VRT
    cropped_vrt_filename = vrt_path.parent / f"tmp.vrt"
    gdal.Warp(str(cropped_vrt_filename), str(vrt_path), outputBounds=(minx, miny, maxx, maxy), dstNodata=None)

    # Open cropped VRT
    src = gdal.Open(str(cropped_vrt_filename))
    driver = gdal.GetDriverByName('GTiff')

    # Define the creation options for compression
    co_options = ['COMPRESS=' + compression]

    dst_ds = None

    for bi in range(1, src.RasterCount + 1):
        band = src.GetRasterBand(bi)
        band_arr_subsample = band.ReadAsArray(buf_xsize=band.XSize // subsample, buf_ysize=band.YSize // subsample)
        b_min, b_max = np.percentile(band_arr_subsample, [2, 98])
        band_arr = band.ReadAsArray()
        band_arr = np.clip(255 * (band_arr - b_min) / (b_max - b_min), 0, 255).astype(np.uint8)

        if dst_ds is None:
            # Create the output dataset with compression options
            dst_ds = driver.Create(str(output_path), src.RasterXSize, src.RasterYSize, src.RasterCount, gdal.GDT_Byte, options=co_options)
            dst_ds.SetGeoTransform(src.GetGeoTransform())
            dst_ds.SetProjection(src.GetProjection())

        dst_ds.GetRasterBand(bi).WriteArray(band_arr)

    dst_ds = None  # Close and save the dataset
    cropped_vrt_filename.unlink()


# Approximate degrees for 20 km buffer at the equator (1 degree of lat ~ 111 km at the equator)
double_buffer_lat_deg = 20 / 111  # Buffer in degrees latitude, which is approximately constant

# Use cosine of the latitude to adjust for the earth's curvature for longitude
double_buffer_lon_deg = 20 / (111 * abs(math.cos(math.radians(point.y))))

# Create a box around the point using the approximate buffer
double_buffered_box = box(point.x - double_buffer_lon_deg, point.y - double_buffer_lat_deg,
                          point.x + double_buffer_lon_deg, point.y + double_buffer_lat_deg)


# Apply the stretch to each VRT file
for vrt_file in output_dir.glob("*.vrt"):
    output_file = output_dir / f"{vrt_file.stem}.tif"
    stretch_contrast_with_subsampling(vrt_file, output_file, double_buffered_box)

