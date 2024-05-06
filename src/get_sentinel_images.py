import math
import requests
import subprocess
import numpy as np
import leafmap
import planetary_computer
import geopandas
import rich.table


from osgeo import gdal, osr
from pyproj import Transformer, CRS
from pathlib import Path
from pystac_client import Client
from shapely.geometry import Point, mapping, box
from pystac.extensions.eo import EOExtension as eo
from IPython.display import Image, display
from PIL import Image
from rasterio import windows, features, warp


class ReadSTAC():
    """Read STAC API and obtain a TIFF of a desired location within the desired timeframe"""
    
    def __init__(
        self, 
        api_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        collection: str = "sentinel-2-l2a",
        location: list = [-6.054643555224412, -50.18255000697929],
        buffer: int = 10,
        timerange: str = "2020-06-01/2020-12-31",
        max_cloud_cover: int = 10
    ):
        """
        Initialize the ReadSTAC class.

        Parameters:
        - api_url (str): The URL of the STAC API. Default is "https://planetarycomputer.microsoft.com/api/stac/v1".
        - collection (str): The name of the collection. Default is "sentinel-2-l2a".
        - location (list): The coordinates of the desired location. Default is [-6.054643555224412, -50.18255000697929].
        - buffer (int): The buffer size in kilometers. Default is 10.
        - timerange (str): The desired time range in the format "start_date/end_date". Default is "2020-06-01/2020-12-31".
        - max_cloud_cover (int): The maximum cloud cover percentage. Default is 10.

        """
        # Approximate degrees for 10 km buffer at the equator (1 degree of lat ~ 111 km at the equator)
        buffer_lat_deg = buffer / 111  # Buffer in degrees latitude, which is approximately constant

        # Approximate degrees for 10 km buffer in longitude (varies with latitude)
        # Use cosine of the latitude to adjust for the earth's curvature
        buffer_lon_deg = buffer / (111 * abs(math.cos(math.radians(location.y))))

        # Create a box around the point using the approximate buffer
        bbox = box(location.x - buffer_lon_deg, location.y - buffer_lat_deg,
                   location.x + buffer_lon_deg, location.y + buffer_lat_deg)
        
        bbox_geojson = mapping(bbox)
        
        self.api_url = api_url
        self.collection = collection
        self.location = location
        self.timerange = timerange
        self.max_cloud_cover = max_cloud_cover
        self.bbox = bbox
        self.bbox_geojson = bbox_geojson

    def get_items(self):
        
        # Open the STAC API
        catalog = Client.open(
            self.api_url,
            modifier = (
                planetary_computer.sign_inplace 
                if self.api_url == "https://planetarycomputer.microsoft.com/api/stac/v1" 
                else None
            ),
        )
        
        # Search for items in the collection that intersect the buffered box
        search = catalog.search(
            collections=[self.collection],
            bbox=self.bbox,
            datetime=self.timerange,
            limit=50,
            query={"eo:cloud_cover": {"lt": 10}} # Less than 10% cloud cover
        )

        items = search.item_collection()
        print(f"{len(items)} Items found.")

        return items


    def load_item(self, desired_item: str):
        """
        Load the desired item from the STAC API.
        
        Parameters:
        - desired_item (str): either most recent item, or least cloud cover item. 
        """
        items = self.get_items()

        #TODO: Currently this only works for single item/tile/orbit point. Need to implement for multiple tiles
        if desired_item == "most_recent":
            item = max(items, key=lambda item: item.datetime)
        elif desired_item == "least_cloudy":
            item = min(items, key=lambda item: eo.ext(item).cloud_cover)
        else:
            item = None
            print("Please specify either 'most_recent' or 'least_cloudy' as the desired item.")
        
        print(
            f"Choosing {item.id} from {item.datetime.date()}"
            f" with {eo.ext(item).cloud_cover}% cloud cover"
        )

        return item
    

    def preview_item(self, item):
        """
        Preview the item on an interactive map.
        
        Parameters:
        - item (dict): The item to view, obtained from load_item().
        """

        table = rich.table.Table("Asset Key", "Description")
        for asset_key, asset in item.assets.items():
            table.add_row(asset_key, asset.title)

        # Rendering With Image Preview
        whole_image = Image(url=item.assets["rendered_preview"].href, width=500)

        with rasterio.open(item.assets["rendered_preview"].href) as ds:
            aoi_bounds = features.bounds(self.bbox)
            warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
            aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
            band_data = ds.read(window=aoi_window)

        img = Image.fromarray(np.transpose(band_data, axes=[1, 2, 0]))
        w = img.size[0]
        h = img.size[1]
        aspect = w / h
        target_w = 800
        target_h = (int)(target_w / aspect)
        cropped_image = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        display(table)
        display(whole_image)
        display(cropped_image)
    

    def download_item(self, item, bands: list = ['B02', 'B03', 'B04', 'B08']):
        """
        Download the desired bands from the item.
        
        Parameters:
        - item (dict): The item to download, obtained from load_item().
        - bands (list): The bands to download. Default is ['B02', 'B03', 'B04', 'B08'].
        """

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
        for band_name, band_key in bands_to_download.items():
            download_band(item, band_name, band_key)

        return output_dir
    
    def create_vrt(self, item_id, band_files, vrt_suffix):
        # Create VRT for the bands
        vrt_filename = output_dir / f"{item_id}_{vrt_suffix}.vrt"
        gdal.BuildVRT(str(vrt_filename), band_files, separate=True, options=['COMPRESS=DEFLATE'])
        print(f"Created VRT: {vrt_filename}")

    # enhance composites 
    def stretch_contrast_with_subsampling(self, vrt_path, output_path, crop_bbox, subsample=10, compression='DEFLATE'):
        
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

    
    def download_and_stretch(self, item, bands: list = ['B02', 'B03', 'B04', 'B08']):
        """
        Download the desired bands from the item and stretch the contrast.
        
        Parameters:
        - item (dict): The item to download, obtained from load_item().
        - bands (list): The bands to download. Default is ['B02', 'B03', 'B04', 'B08'].
        """
        output_dir = self.download_item(item, bands)
        band_files = list(output_dir.glob("*.tif"))

        # Create VRT for the bands
        self.create_vrt(item.id, band_files, "stacked")

        # Stretch contrast with subsampling
        for vrt_file in output_dir.glob("*.vrt"):
            output_file = output_dir / f"{vrt_file.stem}.tif"
            self.stretch_contrast_with_subsampling(vrt_file, output_file, self.bbox, subsample=10)
