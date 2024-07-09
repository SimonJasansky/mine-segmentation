
import pyproj
from functools import partial
from shapely.ops import transform
import rasterio
import numpy as np
from PIL import Image
import rasterio
from rasterio.merge import merge
import glob
import os

def calculate_dimensions_km(polygon):
    """
    Calculate the dimensions (width, height) in kilometers of a given polygon.
    
    Parameters:
    - polygon: A shapely Polygon object.
    
    Returns:
    - A tuple (width_km, height_km) representing the dimensions in kilometers.
    """
    # Define the projection to UTM (Universal Transverse Mercator)
    # Find UTM zone for the centroid of the polygon for more accuracy
    utm_zone = int((polygon.centroid.x + 180) / 6) + 1
    crs_proj = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84', preserve_units=False)
    
    # Define transformations from WGS84 to UTM and back
    project_to_utm = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'), crs_proj)
    project_to_wgs84 = partial(pyproj.transform, crs_proj, pyproj.Proj(init='epsg:4326'))
    
    # Transform the polygon to the UTM projection
    polygon_utm = transform(project_to_utm, polygon)
    
    # Calculate bounds in UTM
    minx, miny, maxx, maxy = polygon_utm.bounds
    
    # Calculate width and height in meters
    width_m = maxx - minx
    height_m = maxy - miny
    
    # Convert meters to kilometers
    width_km = width_m / 1000
    height_km = height_m / 1000
    
    return (width_km, height_km)


def geotiff_to_PIL(image_path):
    """
    Convert a GeoTIFF image to a PIL image.

    Parameters:
    - image_path (str): The path to the GeoTIFF image.

    Returns:
    - img (PIL.Image): The PIL image.
    """


    with rasterio.open(image_path) as src:
        # Read the red, green, and blue bands from the GeoTIFF
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)

        # Stack the R, G, and B bands to create an RGB image
        rgb = np.dstack((r, g, b))

    # Normalize the RGB image
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    img = Image.fromarray((rgb * 255).astype(np.uint8))
    return img


def normalize_geotiff(file_path, output_path):
    """
    Normalize a GeoTIFF image and save it to the output path.

    Parameters:
    - file_path (str): The path to the GeoTIFF image.
    - output_path (str): The path to save the normalized image.
    """

    with rasterio.open(file_path) as src:
        # Read the red, green, and blue bands from the GeoTIFF
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)

        # Stack the R, G, and B bands to create an RGB image
        rgb = np.dstack((r, g, b))

        # Normalize the RGB image
        rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        # Convert to uint8
        rgb_uint8 = (rgb_normalized * 255).astype(np.uint8)

        # Prepare metadata for the output file
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": rgb_uint8.shape[0],
            "width": rgb_uint8.shape[1],
            "count": 3,
            "dtype": np.uint8
        })

    # Save the normalized image
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        # Write each band separately
        dst.write(rgb_uint8[:, :, 0], 1)  # Red band
        dst.write(rgb_uint8[:, :, 1], 2)  # Green band
        dst.write(rgb_uint8[:, :, 2], 3)  # Blue band


def merge_geotiffs(tiles_dir, output_path):
    """
    Merge multiple GeoTIFF files in a directory and save the merged file.

    Parameters:
    - tiles_dir (str): The directory containing the GeoTIFF files.
    - output_path (str): The path to save the merged file.
    """

    # List of GeoTIFF files to merge
    geotiff_files = glob.glob(os.path.join(tiles_dir, "*.tif"))

    # Open the GeoTIFF files
    src_files_to_mosaic = []
    for fp in geotiff_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    # Merge function
    mosaic, out_trans = merge(src_files_to_mosaic)

    # Copy the metadata
    out_meta = src.meta.copy()

    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": src.crs
                     })

    # Write the mosaic raster to disk
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close the files
    for src in src_files_to_mosaic:
        src.close()