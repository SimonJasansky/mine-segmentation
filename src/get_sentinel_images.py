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
from PIL import Image
from osgeo import gdal


from pyproj import Transformer
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
            aoi_bounds = features.bounds(self.bbox_geojson)
            warped_aoi_bounds = rasterio.warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
            band_data = ds.read(window=rasterio.windows.from_bounds(transform=ds.transform, *warped_aoi_bounds))

        img = Image.fromarray(np.transpose(band_data, axes=[1, 2, 0]))
        aspect = img.size[0] / img.size[1]
        cropped_image = img.resize((800, int(800 / aspect)), Image.Resampling.BILINEAR)

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
        stack.rio.to_raster(output_path, compress="zstd")


    def show_stack(
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
    ) -> None:
        """
        Download the specified band from the item to the specified path, in TIF format. 

        Parameters:
        - item (dict): The item to download, obtained from load_item().
        - band_name (str): The name of the band.
        - band_key (str): The key of the band.
        - output_dir (str): The output directory, relative to the working directory /src.

        Returns:
        - None
        """
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Set the output directory to 'src/data' relative to the script directory
        output_dir = os.path.join(script_dir, 'data/temp')

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate the full path where the file will be saved
        file_path = os.path.join(output_dir, f"{item.id}_{band_name}.tif")

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"File {file_path} already exists, skipping download.")
            return
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


    def tif_to_vrt(
            self,
            item: dict, 
            bands_to_download: dict,
            output_dir: str,
        ) -> None:
            """
            Create a VRT file for the specified bands.

            Parameters:
            - item (dict): The item to download, obtained from load_item().
            - bands_to_download (dict): The bands to download, with band names as keys, and band keys as items.
            - output_dir (str): The output directory.           
            
            Returns:
            - None
            """
            item_id = item.id

            # Loop through items and download the specified bands
            for band_name, band_key in bands_to_download.items():
                self.download_band(item, band_name, band_key, output_dir)

            # Paths to the band files
            band_paths = [str(f"{output_dir}/{item_id}_{band_name}.tif") for band_name in bands_to_download.keys()]
            
            vrt_filename = output_dir / f"{item_id}.vrt"
            gdal.BuildVRT(str(vrt_filename), band_paths, separate=True, options=['COMPRESS=DEFLATE'])
            print(f"Created VRT: {vrt_filename}")


    def download_item(
        self, 
        item: dict,
        output_dir: str,
        bands_to_download: dict = {'B02': 'blue', 'B03': 'green', 'B04': 'red'},
    ) -> str:
        """
        Download the desired bands.
        
        Parameters:
        - item (dict): The item to download, obtained from load_item().
        - bands_to_download (dict): The bands to download, with band names as keys, and band keys as items.

        Returns: 
        - Path to the VRT 
        """

        # Download the bands
        create_vrt(item, bands_to_download, output_dir)
    
    