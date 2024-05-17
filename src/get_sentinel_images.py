import math
import planetary_computer
import pystac_client
import stackstac
import xarray
import pyproj
import rioxarray
import rasterio
import requests
import os
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from osgeo import gdal, osr
from pyproj import Transformer, CRS
from shapely.geometry import Point, mapping, box
from pystac.extensions.eo import EOExtension as eo

class ReadSTAC():
    """
    Read STAC API and obtain a TIFF of a desired location within the desired timeframe
    Default settings are set to use Microsoft's Planetary Computer API, and Sentinel-2-L2A data
    """
    
    def __init__(
        self, 
        api_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        collection: str = "sentinel-2-l2a",
    ):
        """
        Initialize the ReadSTAC class.

        Parameters:
        - api_url (str): The URL of the STAC API. Default is "https://planetarycomputer.microsoft.com/api/stac/v1".
        - collection (str): The name of the collection. Default is "sentinel-2-l2a".
        """
        self.api_url = api_url
        self.collection = collection

        # Set up a temporary directory to store the downloaded files
        script_dir = os.path.dirname(os.path.realpath(__file__))
        temp_dir = os.path.join(script_dir, 'data/temp')
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir

    ########################
    ### Helper Functions ###
    ########################

    def transform_bbox(
        self, 
        bbox: list, 
        crs_src: str, 
        crs_dst: str,
    ) -> list:
        """
        Transform a bounding box to a desired CRS.

        Parameters:
        - crs_src (str): The source CRS.
        - crs_dst (str): The destination CRS.

        Returns:
        - bbox_transformed (list): The transformed bounding box.
        """
        transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
        bbox_transformed = [transformer.transform(x, y) for x, y in zip(bbox[::2], bbox[1::2])]

        # flatten the list
        bbox_transformed = [item for sublist in bbox_transformed for item in sublist]

        return bbox_transformed
    


    ############################
    ### Reading STAC Methods ###
    ############################

    def get_items(
        self,
        location: list, 
        buffer: int,
        timerange: str,
        max_cloud_cover: int = 100,
    ) -> dict:
        """
        Query the STAC API. 

        Parameters: 
        - location (list): Coordinates of desired location. Format [longitude, latitude], using WGS84.
        - buffer (int): The buffer size in kilometers.
        - timerange (str): The desired time range in the format "start_date/end_date".
        - max_cloud_cover (int): The maximum cloud cover percentage.
        
        Returns: 
        - items (dict): Dictionary containing all the found items meeting specified criteria. 
        """
        location = Point(location)
        
        # Approximate degrees for 10 km buffer at the equator (1 degree of lat ~ 111 km at the equator)
        buffer_lat_deg = buffer / 111  # Buffer in degrees latitude, which is approximately constant

        # Approximate degrees for 10 km buffer in longitude (varies with latitude)
        # Use cosine of the latitude to adjust for the earth's curvature
        buffer_lon_deg = buffer / (111 * abs(math.cos(math.radians(location.y))))
        
        # Create a box around the point using the approximate buffer
        bbox = [location.x - buffer_lon_deg, location.y - buffer_lat_deg,
                location.x + buffer_lon_deg, location.y + buffer_lat_deg]
        bbox_geojson = mapping(box(bbox[0], bbox[1], bbox[2], bbox[3]))
        
        self.location = location
        self.timerange = timerange
        self.max_cloud_cover = max_cloud_cover
        self.bbox = bbox

        self.bbox_geojson = bbox_geojson

        # Open the STAC API
        catalog = pystac_client.Client.open(
            self.api_url,
            modifier = (
                planetary_computer.sign_inplace 
                if self.api_url == "https://planetarycomputer.microsoft.com/api/stac/v1" 
                else None
            ),
        )
        
        # Search for items in the collection that intersect the bounding box
        search = catalog.search(
            collections=[self.collection],
            bbox=self.bbox,
            datetime=self.timerange,
            query={"eo:cloud_cover": {"lt": self.max_cloud_cover}}
        )

        items = search.item_collection()
        print(f"{len(items)} Items found.")

        return items
    

    def filter_item(
        self, 
        items: dict, 
        filter_by: str,
    ) -> dict:
        """
        Filter for a specific item from the STAC API. Currently supports loadinge eithe the most recent or the least cloudy item. 
        
        Parameters:
        - desired_item (str): one of ['most_recent', 'least_cloudy']. 

        Returns: 
        - item (dict): the metadata dictionary for the most recent or least cloudy item. 
        """
        #TODO: Currently this only works for single item/tile/orbit point. Need to implement for multiple tiles
        if filter_by == "most_recent":
            item = max(items, key=lambda item: item.datetime)
        elif filter_by == "least_cloudy":
            item = min(items, key=lambda item: eo.ext(item).cloud_cover)
        else:
            "No filter selected, returning all items."
            return items

        print(
            f"Choosing {item.id} from {item.datetime.date()}"
            f" with {eo.ext(item).cloud_cover}% cloud cover"
        )

        return item
    

    def preview_item(
        self, 
        item: dict,
    ) -> tuple:
        """
        Preview an item on an interactive map.
        
        Parameters:
        - item (dict): The item to view, obtained from load_item().

        Returns: 
        - Table: Table containing all the channels of the item
        - Cropped Image: preview of the image, cropped to the area of interest only
        """
        # Create dataframe with the channels and metadata
        rows = []
        for asset_key, asset in item.assets.items():
            rows.append((asset_key, asset.title))
        table = pd.DataFrame(rows, columns=["Asset Key", "Description"])

        # Rendering With Image Preview
        with rasterio.open(item.assets["visual"].href) as ds:
            aoi_bounds = rasterio.features.bounds(self.bbox_geojson)
            warped_aoi_bounds = rasterio.warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
            band_data = ds.read(window=rasterio.windows.from_bounds(transform=ds.transform, *warped_aoi_bounds))

        img = Image.fromarray(np.transpose(band_data, axes=[1, 2, 0]))
        aspect = img.size[0] / img.size[1]
        cropped_image = img.resize((800, int(800 / aspect)), Image.Resampling.BILINEAR)

        # Downsample image to decrease size of the image
        cropped_image.thumbnail((800, 800))

        return table, cropped_image
    
    ############################################
    ### Processing as xarray using stackstac ###
    ############################################

    def get_stack(
        self, 
        items: dict, 
        filter_by: str = None,
        resolution: int = 10,
        bands: list = ['B02', 'B03', 'B04'],
    ) -> xarray.DataArray:
        """
        Load a specific item from the STAC API using stackstac. 
        Currently supports loading eithe the most recent or the least cloudy item. 
        
        Parameters:
        - items (dict): Dictionary containing all the found items.
        - filter_by (str, optional): one of ['most_recent', 'least_cloudy'].
        - resolution (int, optional): The resolution of the image. Default is 10.
        - bands (list, optional): The bands to load. Default is ['B02', 'B03', 'B04'].
        - bounds (list, optional): The bounds of the image. Default is None. 

        Returns: 
        """
        # Filter the items
        item = self.filter_item(items, filter_by)

        # slice the x and y dimensions to the specified bounds
        # transform the bounds to the desired CRS
        bounds = self.transform_bbox(self.bbox, "epsg:4326", item.properties["proj:epsg"])

        # Load the item
        stack = stackstac.stack(item, resolution=resolution, assets=bands, bounds=bounds)

        # optionally removing the time dimension
        stack = stack.squeeze()

        return stack
    
    def stretch_contrast(
        self, 
        stack: xarray.Dataset,
    ) -> xarray.Dataset:
        """
        Stretch the contrast of a stackstac.stack object.

        Parameters:
        - stack (xarray.Dataset): The stackstac.stack object to stretch.

        Returns:
        - stack (xarray.Dataset): The contrast-stretched stackstac.stack object.
        """
        # Loop over each band in the stack
        for band in range(len(stack)):

            # Calculate the 2nd and 98th percentiles of the band data
            min_val = int(stack[band].min().values)
            max_val = int(stack[band].max().values)

            # Stretch the contrast and scale the values to the range 0-255
            stack[band] = 255 * (stack[band] - min_val) / (max_val - min_val)

            # Clip the values to the range 0-255 and convert to uint8
            stack[band] = stack[band].clip(0, 255).astype('uint8')

        return stack

    def get_stretched_stack(
        self, 
        items, 
        filter_by: str = None,
        resolution: int = 10,
        bands: list = ['B02', 'B03', 'B04'],
    ) -> xarray.Dataset:
        """
        Get a stackstac.stack object, reproject it to a given CRS, and perform contrast stretching.

        Parameters:
        - items (dict): Dictionary containing all the found items.
        - filter_by (str): One of ['most_recent', 'least_cloudy'].
        - resolution (int): The resolution of the image.
        - bands (list): The bands to load.
        - bounds (list): The bounds of the image.
        - target_crs (str): The target CRS.
        - subsample (int, optional): The subsampling factor for contrast stretching. Default is 10.

        Returns:
        - stack_stretched (xarray.Dataset): The reprojected and contrast-stretched stackstac.stack object.
        """
        # Get the stack item
        stack = self.get_stack(items, filter_by, resolution, bands)

        # Perform contrast stretching
        stack_stretched = self.stretch_contrast(stack)

        return stack_stretched
    
    def save_stack_as_geotiff(
        self, 
        stack: xarray.Dataset, 
        output_path: str,
        compression = "zstd",
    ) -> None:
        """
        Save the stackstac.stack object as a GeoTIFF.

        Parameters:
        - stack (xarray.Dataset): The stackstac.stack object to save.
        - output_path (str): The output path.

        Returns:
        - None
        """
        # Save the stack as a GeoTIFF
        stack.rio.to_raster(output_path, compress=compression)


    def display_stack_as_image(
        self, 
        stack: xarray.Dataset,
    ) -> None:
        """
        Display the stackstac.stack object.

        Parameters:
        - stack (xarray.Dataset): The stackstac.stack object to display.

        Returns:
        - None
        """
        # Plot the stack
        stack.plot.imshow(robust=True, figsize=(10, 10))

    #############################################
    ### Downloading Whole Files from STAC API ###
    #############################################

    def download_band(
        self,
        item: dict, 
        band_name: str, 
        band_key: str,
    ) -> str:
        """
        Download the specified band from the item to the specified path, in TIF format. 

        Parameters:
        - item (dict): The item to download, obtained from load_item().
        - band_name (str): The name of the band.
        - band_key (str): The key of the band.

        Returns:
        - file_path (str): The path to the downloaded file.
        """

        # Generate the full path where the file will be saved
        file_path = os.path.join(self.temp_dir, f"{item.id}_{band_name}.tif")

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"File {file_path} already exists, skipping download.")
            return file_path
        else: 
            print("Downloading", file_path)

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

        return file_path


    def tif_to_vrt(
        self,
        item_id: str,
        band_paths: list, 
    ) -> str:
        """
        Create a VRT file for the specified bands.

        Parameters:
        - item_id (str): The ID of the item.
        - band_paths (list): The paths to the band files.        
        
        Returns:
        - vrt_filepath (str): The path to the VRT file.
        """
        # Create VRT for the bands
        vrt_filepath = f"{self.temp_dir}/{item_id}.vrt"
        options = gdal.BuildVRTOptions(separate=True, options=['COMPRESS=DEFLATE'])
        gdal.BuildVRT(vrt_filepath, band_paths)
        print(f"Created VRT: {vrt_filepath}")
        return vrt_filepath


    def stretch_contrast_with_subsampling(
        self, 
        vrt_path, 
        output_path, 
        crop_bbox, 
        subsample=10, 
        compression='DEFLATE'
    ) -> None:
        
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


    def download_item_as_vrt(
        self, 
        item: dict,
        bands_to_download: dict = {'B02': 'blue', 'B03': 'green', 'B04': 'red'},
    ) -> str:
        """
        Download the bands of an item, enhance the contrasts, and save as a VRT file.

        Parameters:
        - item (dict): The item to download.
        - bands_to_download (dict): The bands to download.

        Returns:
        - vrt_filepath (str): The path to the VRT file.
        """
        item_id = item.id

        # Download the bands
        band_paths = []
        for band_key, band_name in bands_to_download.items():
            band_path = self.download_band(item, band_name, band_key)
            band_paths.append(band_path)

        # Create a VRT file for the downloaded bands
        vrt_filepath = self.tif_to_vrt(item.id, band_paths)
        vrt_filepath = Path(vrt_filepath)

        # Stretch the contrast of the VRT file
        output_path = f"{self.temp_dir}/{item_id}_stretched.tif"
        self.stretch_contrast_with_subsampling(vrt_filepath, output_path, box(*self.bbox))

        return output_path
    
    