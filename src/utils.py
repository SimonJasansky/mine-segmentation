
import pyproj
from functools import partial
from shapely.ops import transform


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