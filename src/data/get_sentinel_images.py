import math
import planetary_computer
import pystac.item_collection
import pystac_client
import stackstac
import xarray as xr
import dask.array as da
import pyproj
import rioxarray
import rasterio
import requests
import os
import numpy as np
import pandas as pd
import pystac
from skimage import exposure


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
        - bbox (list): The bounding box in the form [minx, miny, maxx, maxy].
        - crs_src (str): The source CRS.
        - crs_dst (str): The destination CRS.

        Returns:
        - bbox_transformed (list): The transformed bounding box in the form [minx, miny, maxx, maxy].
        """
        transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
        bbox_transformed = [transformer.transform(x, y) for x, y in zip(bbox[::2], bbox[1::2])]

        # Flatten the list
        bbox_transformed = [item for sublist in bbox_transformed for item in sublist]

        return bbox_transformed
    

    #####################
    ### Read STAC API ###
    #####################

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
        items: pystac.item_collection.ItemCollection, 
        filter_by: str,
    ) -> pystac.item_collection.ItemCollection | pystac.item.Item:
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
        item: pystac.item.Item,
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
    
    ###################################################
    ### Processing xarray.DataArray using stackstac ###
    ###################################################

    def get_stack(
        self, 
        items: pystac.item.Item | pystac.item_collection.ItemCollection, 
        bands: list,
        filter_by: str = None,
        resolution: int = 10,
    ) -> xr.DataArray:
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
        - stack (xr.DataArray): The stackstac.stack object.
        """
        print("Loading stack...")

        # Filter the items
        item = self.filter_item(items=items, filter_by=filter_by)

        # Get Item CRS in case it must be set manually
        if isinstance(item, pystac.item_collection.ItemCollection):
            # If the item is a collection, get the CRS from the first item
            item_crs = item[0].properties["proj:epsg"]
        else:
            # If the item is not a collection, get the CRS from the item
            item_crs = item.properties["proj:epsg"]

        # slice the x and y dimensions to the original bounding box

        # For this, transform the bounding box to the CRS of the item
        bounds = self.transform_bbox(self.bbox, "epsg:4326", item_crs)

        # Load the item
        stack = stackstac.stack(
            item, 
            resolution=(resolution, resolution), 
            assets=bands, 
            bounds=bounds, 
            epsg=item_crs
            )

        # optionally removing the time dimension, in case there is only one item
        stack = stack.squeeze()

        return stack
        

    def stretch_contrast_stack(
        self, 
        stack: xr.Dataset,
        upper_percentile: float = .98,
        lower_percentile: float = .02,
    ) -> xr.Dataset:
        """
        Perform contrast stretching.
        To compute the percentiles, the stack object must be loaded in memory.

        Parameters:
        - stack (xr.Dataset): The stackstac.stack object to stretch.
        - upper_percentile (float): The upper percentile for contrast stretching. Default is .98.
        - lower_percentile (float): The lower percentile for contrast stretching. Default is .02.

        Returns:
        - stack_stretched (xr.Dataset): The reprojected and contrast-stretched stackstac.stack object.
        """
        print("Stretching contrast...")

        # Load the stack object in memory
        # This is necessary to compute the quantiles
        stack = stack.compute()

        # Perform contrast stretching for each band
        for band_name in stack.band.values:
            band = stack.sel(band=band_name)

            # Calculate the percentiles for the band
            min_val = band.quantile(lower_percentile).values
            max_val = band.quantile(upper_percentile).values

            # Rescale the intensity of the image to cover the range 0-255
            stretched_band = (band - min_val) / (max_val - min_val) * 255

            # clip the value to the range 0-255
            stretched_band = np.clip(stretched_band, 0, 255).astype(np.uint8)

            # Update the band in the stack
            stack.loc[dict(band=band_name)] = stretched_band

        # change datatype of output stack from float to int
        stack = stack.astype(np.uint8)

        return stack
    
    
    def save_stack_as_geotiff(
        self, 
        stack: xr.Dataset, 
        compression = "zstd",
    ) -> str:
        """
        Save the stackstac.stack object as a GeoTIFF.

        Parameters:
        - stack (xr.Dataset): The stackstac.stack object to save.


        Returns:
        - output_path (str): The path to the saved GeoTIFF file.
        """
        # Get the item ID
        item_id = str(stack.id.values)
        output_path = f"{self.temp_dir}/{item_id}.tif"

        # Save the stack as a GeoTIFF
        print(f"Saving stack as GeoTIFF under: {output_path}")
        stack.rio.to_raster(output_path, compress=compression)
        return(output_path)


    def display_stack_as_image(
        self, 
        stack: xr.Dataset,
    ) -> None:
        """
        Display the stackstac.stack object.

        Parameters:
        - stack (xr.Dataset): The stackstac.stack object to display.

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
    
    