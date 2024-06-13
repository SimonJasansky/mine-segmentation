import streamlit as st
import geopandas as gpd
import random
import shapely
import leafmap.foliumap as leafmap
import os

from src.data.get_satellite_images import ReadSTAC

MINING_AREAS = "/workspaces/mine-segmentation/data/interim/mining_areas.gpkg"
MAUS_POLYGONS = "/workspaces/mine-segmentation/data/external/maus_mining_polygons.gpkg"
TANG_POLYGONS = "/workspaces/mine-segmentation/data/external/tang_mining_polygons/74548_mine_polygons/74548_projected.shp"
DATASET = "/workspaces/mine-segmentation/data/interim/mining_tiles_with_masks.gpkg"

# Load data
@st.cache_data
def load_data():
    mining_area_tiles = gpd.read_file(MINING_AREAS)

    # Load Maus dataset
    maus_gdf = gpd.read_file(MAUS_POLYGONS)

    # Load Tang dataset
    tang_gdf = gpd.read_file(TANG_POLYGONS)
    tang_gdf = tang_gdf.to_crs(epsg=4326)
    tang_gdf.geometry = shapely.wkb.loads(shapely.wkb.dumps(tang_gdf.geometry, output_dimension=2))

    # Check if the dataset exists, and if not, create it
    if os.path.exists(DATASET):
        # Load the dataset
        dataset = gpd.read_file(DATASET)
    else:
        # Create the dataset
        columns = ["tile_id", "tile_bbox", "sentinel_2_id", "geometry"]
        dataset = gpd.GeoDataFrame(columns=columns)

    # Initialize the STAC reader
    api_url="https://planetarycomputer.microsoft.com/api/stac/v1"
    stac_reader = ReadSTAC(
        api_url=api_url, 
        collection = "sentinel-2-l2a",
        data_dir="/workspaces/mine-segmentation/streamlit_app/data"
        )

    return mining_area_tiles, maus_gdf, tang_gdf, stac_reader, dataset


def get_random_tile(mining_area_tiles):
    # Sample a random mining tile
    random_tile = mining_area_tiles.sample(n=1)
    return random_tile


def visualize_tile(tile, maus_gdf, tang_gdf, stac_reader, year):
    
    # Get the geometry of the random tile
    tile_geometry = tile['geometry'].values[0]

    bbox = tile_geometry.bounds
    # get the least cloudy sentinel image for the tile
    bands = ['B04', 'B03', 'B02']
    items = stac_reader.get_items(
        bbox=bbox,
        timerange=f'{year}-01-01/{year}-12-31',
        max_cloud_cover=10
    )

    stack = stac_reader.get_stack(items, filter_by="least_cloudy", bands=bands, resolution=10)    
    s2_tile_id = stack.attrs['s2_tile_id']
    stack_stretched = stac_reader.stretch_contrast_stack(stack, upper_percentile=0.99, lower_percentile=0.01)
    image = stac_reader.save_stack_as_geotiff(stack_stretched, filename="sentinel_image.tif")

    # Create a Map
    m = leafmap.Map(center=[tile_geometry.centroid.y, tile_geometry.centroid.x], zoom=2)

    # add the image
    m.add_raster(image)

    # Filter the polygons that are included 
    maus_gdf_filtered = maus_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    tang_gdf_filtered = tang_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    style = {
        "stroke": True,
        "color": "red",
        "weight": 2,
        "opacity": 1,
        "fill": True,
        "fillColor": "red",
        "fillOpacity": 0.1,
    }

    # Display the polygons
    if not maus_gdf_filtered.empty:
        m.add_gdf(maus_gdf_filtered, layer_name="maus_gdf")
    if not tang_gdf_filtered.empty:
        m.add_gdf(tang_gdf_filtered, layer_name="tang_gdf", style=style)

    m.to_streamlit()

    return maus_gdf_filtered, tang_gdf_filtered, s2_tile_id


def accept_polygons(tile, accepted_polygons, S2_tile_name, dataset):
    # Get the tile id
    tile_id = tile.index[0]

    # Get the tile bbox
    tile_bbox = tile['geometry'].values[0].bounds

    # Get the sentinel 2 id
    sentinel_2_id = S2_tile_name

    # Add the polygons to the dataset
    for index, row in accepted_polygons.iterrows():
        dataset = dataset.append({
            "tile_id": tile_id,
            "tile_bbox": tile_bbox,
            "sentinel_2_id": sentinel_2_id,
            "geometry": row['geometry']
        }, ignore_index=True)

    return dataset


def main():
    st.title('Mine Segmentation App')
    st.text("""
        This app allows you to accept or reject polygons for training a mine segmentation model. 
        Data Sources:
        - Maus et al: https://doi.pangaea.de/10.1594/PANGAEA.942325
        - Tang et al: https://zenodo.org/doi/10.5281/zenodo.6806816 
    """)

    # Load data
    mining_area_tiles, maus_gdf, tang_gdf, stac_reader, dataset = load_data()

    # Add a streamlit radio button for selecting the year
    year = st.radio("Select Year", list(range(2016, 2023)), index=3)

    # Get a random tile
    tile = get_random_tile(mining_area_tiles)

    # Visualize the tile
    maus_gdf_filtered, tang_gdf_filtered, s2_tile_id = visualize_tile(tile, maus_gdf, tang_gdf, stac_reader, year)

    # Add buttons for accepting maus and tang polygons
    if st.button("Accept Maus", key="maus"):
        dataset = accept_polygons(tile, accepted_polygons=maus_gdf_filtered)
    if st.button("Accept Tang", key="tang"):
        dataset = accept_polygons(tile, accepted_polygons=tang_gdf_filtered)

    # Save the dataset
    if st.button("Save Dataset"):
        dataset.to_file(DATASET, driver="GPKG")


if __name__ == "__main__":
    main()