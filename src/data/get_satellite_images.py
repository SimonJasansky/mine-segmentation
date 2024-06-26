import math
import planetary_computer
import pystac
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
import geopandas as gpd
import leafmap.leafmap as leafmap
import shapely
import warnings

from skimage import exposure
from pathlib import Path
from PIL import Image
from pyproj import Transformer, CRS
from shapely.geometry import Point, mapping, box, shape
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
        data_dir: str = "data/interim",
    ):
        """
        Initialize the ReadSTAC class.

        Parameters:
        - api_url (str): The URL of the STAC API. Default is "https://planetarycomputer.microsoft.com/api/stac/v1".
        - collection (str): The name of the collection. Default is "sentinel-2-l2a".
        - data_dir (str): The directory where the data will be saved. Default is "/workspaces/mine-segmentation/data/interim".
        """
        self.api_url = api_url
        self.collection = collection
        self.data_dir = data_dir


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

    def select_item_w_largest_overlap(self, items): 
        largest_overlap_area = 0
        selected_item = None

        for item in items:
            item_geometry = shapely.geometry.shape(item.geometry)  
            bbox_geometry = shapely.geometry.box(*self.bbox)  

            # Calculate the intersection area
            intersection_area = item_geometry.intersection(bbox_geometry).area
            
            # Check if this item has the largest overlap so far
            if intersection_area > largest_overlap_area:
                largest_overlap_area = intersection_area
                selected_item = item

        return selected_item

    def select_items_w_full_overlap(self, items): 
        """
        Select the items that fully overlap with the bounding box.

        Parameters:
        - items (): The items to select from.

        Returns:
        - selected_item (pystac.item.Item | ): The selected items
        """
        selected_item = None
        bbox_geometry = shapely.geometry.box(*self.bbox)  

        for item in items:
            item_geometry = shapely.geometry.shape(item.geometry)  

            # Calculate the intersection area
            intersection_area = item_geometry.intersection(bbox_geometry).area
            
            # Check if this item has the largest overlap so far
            if intersection_area > largest_overlap_area:
                largest_overlap_area = intersection_area
                selected_item = item

        return selected_item

    #####################
    ### Read STAC API ###
    #####################

    def get_item_by_name(
        self,
        item_name: str, 
        location: list = None,
        timerange: str = None,
        max_cloud_cover: int = None,
        bbox: list = None,       
    ) -> pystac.item.Item: 
        """
        Query the STAC API for a specific item by name.

        Parameters:
        - item_name (str): The name of the item to query. Example: S2B_MSIL2A_20240610T141719_R010_T21QZB_20240610T171009
        - location (list, optional): Coordinates of desired location. Format [longitude, latitude], using WGS84.
        - timerange (str, optional): The desired time range in the format "start_date/end_date".
        - max_cloud_cover (int, optional): The maximum cloud cover percentage.
        - bbox (list, optional): The bounding box in the form [minx, miny, maxx, maxy].

        Returns:
        - item (pystac.item.Item): The queried item.
        """
        item_url = pystac.read_file(f"{self.api_url}/collections/{self.collection}/items/{item_name}")
        signed_item = planetary_computer.sign(item_url)  # these assets can be accessed

        # Optionally set properties
        self.location = location
        self.timerange = timerange
        self.max_cloud_cover = max_cloud_cover
        self.bbox = bbox
        if self.bbox is not None:
            self.bbox_geojson = mapping(box(bbox[0], bbox[1], bbox[2], bbox[3]))

        return signed_item
    
    def get_items(
        self,
        timerange: str,
        bbox: list | tuple = None,
        location: list = None, 
        buffer: int = None,
        max_cloud_cover: int = 100,
    ) -> dict:
        """
        Query the STAC API. 

        Parameters: 
        - location (list): Coordinates of desired location. Format [longitude, latitude], using WGS84.
        - buffer (int): The buffer size in kilometers.
        - timerange (str): The desired time range in the format "start_date/end_date".
        - bbox (list): The bounding box in the form [minx, miny, maxx, maxy]. Default is None.
        - max_cloud_cover (int): The maximum cloud cover percentage.
        
        Returns: 
        - items (dict): Dictionary containing all the found items meeting specified criteria. 
        """
        if bbox is not None:
            # Use the provided bounding box
            bbox_geojson = mapping(box(bbox[0], bbox[1], bbox[2], bbox[3]))
        else:
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

        # add unique tile identifier
        items = self.harmonize_properties(items)

        return items


    def harmonize_properties(
            self, 
            items: pystac.item_collection.ItemCollection,     
        ) -> pystac.item_collection.ItemCollection:
        """
        Assigns a unique tile identifier to each item in the collection.

        Parameters:
        - items (pystac.item_collection.ItemCollection): The items to harmonize.

        Returns:
        - items (pystac.item_collection.ItemCollection): The harmonized items with an additional property "unique_tile_identifier".
        """
        # Set a unique tile ID 
        if "sentinel" in self.collection:
            if "microsoft" in self.api_url:
                unique_tile_identifier = "s2:mgrs_tile"
            elif "aws" in self.api_url:
                unique_tile_identifier = "grid:code"
            else:
                raise ValueError("Unknown API provider. Currently only Microsoft's Planetary computer and AWS are supported.")
            
            # add the unique ID
            for item in items:
                unique_tile_id = item.properties[unique_tile_identifier]
                item.properties["unique_tile_identifier"] = unique_tile_id

        elif "landsat" in self.collection:           
            # add the unique ID
            for item in items:
                unique_tile_id = f'{item.properties["landsat:wrs_path"]}_{item.properties["landsat:wrs_row"]}'
                item.properties["unique_tile_identifier"] = unique_tile_id
        else:
            raise ValueError("Unknown collection. Currently only Landsat and Sentinel collections are supported.")
        
        return items
        

    def filter_item(
        self, 
        items: pystac.item_collection.ItemCollection, 
        filter_by: str,
        full_overlap: bool = False,
        take_best_n: int = 1,
    ) -> pystac.item_collection.ItemCollection | pystac.item.Item:
        """
        Filter for the most recent or the least cloudy items.
        Filter by unique orbit point/MGRS tile. This means that multiple items (ItemCollection) 
        can be returned, even if take_best_n = 1. 
        
        Parameters:
        - items (pystac.item_collection.ItemCollection): The items to filter.
        - filter_by (str): one of ['most_recent', 'least_cloudy']. 
        - full_overlap (bool): Whether to only keep items that fully overlap with the bounding box. Default is False.
        - take_best_n (int): The number of best items to take. Default is 1.

        Returns:
        - item (pystac.item_collection.ItemCollection | pystac.item.Item): The filtered items.
        """
        # Optionally filter for items that fully overlap with the bounding box
        if full_overlap:
            items = [item for item in items if shapely.geometry.shape(item.geometry).contains(shapely.geometry.box(*self.bbox))]

        # Filter items by unique tile/orbit points
        unique_tile_ids = set([item.properties["unique_tile_identifier"] for item in items])
        print(f"Found {len(unique_tile_ids)} unique tile ids.")

        filtered_items = []

        # Apply Filter for each unique tile/orbit points
        for tile_id in unique_tile_ids:

            # Filter items by unique_tile_id
            items_tile = [item for item in items if item.properties["unique_tile_identifier"] == tile_id]
            
            if filter_by == "most_recent":
                items_sorted = sorted(items_tile, key=lambda item: item.datetime, reverse=True)
                item = items_sorted[0:take_best_n]
            elif filter_by == "least_cloudy":
                items_sorted = sorted(items_tile, key=lambda item: eo.ext(item).cloud_cover)
                item = items_sorted[0:take_best_n]
            else:
                raise ValueError("Unknown filter_by value. Please specify either 'most_recent' or 'least_cloudy'.")
            
            item_ids = [item.id for item in item]
            item_dates = [item.datetime.date() for item in item]
            item_cloud_covers = [eo.ext(item).cloud_cover for item in item]
            
            print(
                f"Choosing the best {take_best_n} items."
                f"For unique tile {tile_id}, choosing {item_ids} from {item_dates}"
                f" with {item_cloud_covers}% cloud cover"
            )

            filtered_items.append(item)

        # Flatten the list, as it might be a list of lists
        filtered_items = [item for sublist in filtered_items for item in sublist]

        if len(filtered_items) == 1:
            return filtered_items[0]
        else:
            return pystac.item_collection.ItemCollection(filtered_items)


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

    def preview_tile_outlines(
        self,
        items: pystac.item_collection.ItemCollection,
    ):
        """
        Preview the outlines of the tiles on an interactive map.

        Parameters:
        - items (pystac.item_collection.ItemCollection): The items to preview.

        Returns:
        - m (leafmap.Map): The interactive map.
        """
        geojson = items.to_dict()

        # Create a map
        m = leafmap.Map(center=(self.location.y, self.location.x), zoom=10)

        # Add the tile outlines
        m.add_geojson(geojson, layer_name="Tiles")
        return m
    
    ###################################################
    ### Processing xarray.DataArray using stackstac ###
    ###################################################

    def get_stack(
        self, 
        items: pystac.item.Item | pystac.item_collection.ItemCollection, 
        bands: list,
        filter_by: str = None,
        take_best_n: int = 1,
        allow_mosaic: bool = False,
        resolution: int = 10,
        crop_to_bounds: bool = True,
        custom_point_and_buffer: list = None,
        squeeze_time_dim: bool = True,
        chunk_size: int = 1024,
    ) -> xr.DataArray:
        """
        Load a specific item from the STAC API using stackstac. 
        Currently supports loading eithe the most recent or the least cloudy item. 
        
        Parameters:
        - items (dict): Dictionary containing all the found items.
        - bands (list): The bands to load. 
        - filter_by (str, optional): one of ['most_recent', 'least_cloudy'].
        - take_best_n (int, optional): The number of best items to take. Default is 1.
        - allow_mosaic (bool, optional): Whether to allow mosaicing of multiple items. Default is false
        - resolution (int, optional): The resolution of the image. Default is 10.
        - crop_to_bounds (bool, optional): Whether to crop the image to the bounding box. Default is True.
        - custom_point_and_buffer (list, optional): Custom centroid point and buffer [meters] to use for the bounding box, in the form of [longitude, latitude, buffer]. Default is None.
        - squeeze_time_dim (bool, optional): Whether to squeeze the time dimension. Default is True.
        - chunk_size (int, optional): The chunk size. Default is 1024.

        Returns: 
        - stack (xr.DataArray): The stackstac.stack object.
        """
        if take_best_n > 1 and allow_mosaic == False: 
            print("""Warning: Got take_best_n > 1 and allow_mosaic = False.
                This will not return a mosaic, but only one single pystac.item.Item!""")

        print("Loading stack...")

        # Filter the items
        if filter_by is not None:
            items = self.filter_item(items=items, filter_by=filter_by, take_best_n=take_best_n)
        
        # Get Item CRS in case it must be set manually
        if isinstance(items, pystac.item_collection.ItemCollection):
            # If the item is a collection, get the CRS from the first item
            item_crs = items[0].properties["proj:epsg"]

            # if mosaicing is not allowed, take the item with the largest overlap with the bounds
            if not allow_mosaic:
                items = self.select_item_w_largest_overlap(items)

        else:
            # If the item is not a collection, get the CRS from the item
            item_crs = items.properties["proj:epsg"]

        if crop_to_bounds:
            # Slice the x and y dimensions to the original bounding box 
            # For this, transform the bounding box to the CRS of the item
            bounds = self.transform_bbox(self.bbox, "epsg:4326", item_crs)
        else:
            bounds = None

        # Load the item
        stack = stackstac.stack(
            items, 
            resolution=(resolution, resolution), 
            dtype="float32",
            assets=bands, 
            bounds=bounds, 
            epsg=item_crs,
            chunksize=chunk_size,
            )

        if isinstance(items, pystac.item.Item):
            # add the S2 tile id
            stack.attrs["s2_tile_id"] = items.id
            stack.coords["s2_tile_id"] = items.id
            print(f"Returning stack from single S2 image with ID: {items.id}")
        else:
            print("Returning Mosaic of mutliple S2 Images!")

        if custom_point_and_buffer is not None:
            lon, lat, buffer = custom_point_and_buffer

            x_utm, y_utm = pyproj.Proj(stack.crs)(lon, lat)
            stack = stack.loc[..., y_utm+buffer:y_utm-buffer, x_utm-buffer:x_utm+buffer]

        if squeeze_time_dim:
            # remove the time dimension (would be 1 in case it's a pystac.item.Item)
            stack = stackstac.mosaic(stack, dim='time').squeeze()

        return stack
        

    def stretch_contrast_stack(
        self, 
        stack: xr.Dataset,
        upper_percentile: float = 1.0,
        lower_percentile: float = 0.0,
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
        filename: str,
    ) -> str:
        """
        Save the stackstac.stack object as a GeoTIFF.

        Parameters:
        - stack (xr.Dataset): The stackstac.stack object to save.

        Returns:
        - output_path (str): The path to the saved GeoTIFF file.
        """

        # check if filename ends with .tif, if not add it 
        if not filename.endswith(".tif"):
            filename = f"{filename}.tif"
            
        output_path = f"{self.data_dir}/{filename}"

        # Save the stack as a GeoTIFF
        print(f"Saving stack as GeoTIFF under: {output_path}")
        stack.rio.to_raster(output_path)
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
    
    