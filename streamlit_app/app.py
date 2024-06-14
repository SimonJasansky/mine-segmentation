import streamlit as st
import geopandas as gpd
import pandas as pd
import random
import shapely
import os
import pystac
import sys
from src.data.get_satellite_images import ReadSTAC

import leafmap.foliumap as leafmap

sys.path.append("..")


MINING_AREAS = "data/interim/mining_areas.gpkg"
MAUS_POLYGONS = "data/external/maus_mining_polygons.gpkg"
TANG_POLYGONS = "data/external/tang_mining_polygons/74548_mine_polygons/74548_projected.shp"
DATASET = "data/interim/mining_tiles_with_masks.gpkg"

# Load data
@st.cache_data
def load_data():
    """
    Load the required data for the application.

    Returns:
        Tuple: A tuple containing the loaded data.
            - mining_area_tiles (GeoDataFrame): Mining area tiles.
            - maus_gdf (GeoDataFrame): Maus dataset.
            - tang_gdf (GeoDataFrame): Tang dataset.
            - stac_reader (ReadSTAC): STAC reader.
            - dataset (GeoDataFrame): Dataset.
    """
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
        columns = ["tile_id", "tile_bbox", "sentinel_2_id", "geometry", "source_dataset"]
        dataset = gpd.GeoDataFrame(columns=columns)

    # Initialize the STAC reader
    api_url="https://planetarycomputer.microsoft.com/api/stac/v1"
    stac_reader = ReadSTAC(
        api_url=api_url, 
        collection = "sentinel-2-l2a",
        data_dir="streamlit_app/data"
    )

    return mining_area_tiles, maus_gdf, tang_gdf, stac_reader, dataset


def get_random_tile(mining_area_tiles):
    """
    Get a random mining tile.

    Args:
        mining_area_tiles (GeoDataFrame): Mining area tiles.

    Returns:
        GeoDataFrame: Random mining tile.
    """
    # Sample a random mining tile
    random_tile = mining_area_tiles.sample(n=1)
    return random_tile


def visualize_tile(tile, maus_gdf, tang_gdf, stac_reader, year):
    """
    Visualize a tile.

    Args:
        tile (GeoDataFrame): Tile to visualize.
        maus_gdf (GeoDataFrame): Maus dataset.
        tang_gdf (GeoDataFrame): Tang dataset.
        stac_reader (ReadSTAC): STAC reader.
        year (int): Year to filter the images.

    Returns:
        Tuple: A tuple containing the filtered Maus dataset, filtered Tang dataset, and the Sentinel-2 tile ID.
    """
    # Get the geometry of the random tile
    tile_geometry = tile['geometry'].values[0]

    bbox = tile_geometry.bounds
    # get the least cloudy sentinel image for the tile
    items = stac_reader.get_items(
        bbox=bbox,
        timerange=f'{year}-01-01/{year}-12-31',
        max_cloud_cover=10
    )

    if len(items) < 1: 
        st.error("No S2 images found for this tile. Please refresh the tile.")

    least_cloudy_item = stac_reader.filter_item(items, "least_cloudy")

    if isinstance(least_cloudy_item, pystac.ItemCollection):
        least_cloudy_item = least_cloudy_item[0]

    url = least_cloudy_item.assets["visual"].href
    s2_tile_id = least_cloudy_item.id
    
    # Create a Map
    m = leafmap.Map(center=[tile_geometry.centroid.y, tile_geometry.centroid.x], zoom=2)

    m.add_cog_layer(url)
    
    # visualize the tile boundaries
    m.add_gdf(tile, layer_name="tile", fill_color="blue", fill_opacity=0.1)

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


def accept_polygons(tile, accepted_polygons, accepted_source_dataset, S2_tile_name, dataset):
    """
    Accept polygons for training a mine segmentation model.

    Args:
        tile (GeoDataFrame): Tile containing the polygons.
        accepted_polygons (GeoDataFrame): Accepted polygons.
        accepted_source_dataset (str): Source dataset of the accepted polygons.
        S2_tile_name (str): Sentinel-2 tile name.
        dataset (GeoDataFrame): Dataset to update.

    Returns:
        GeoDataFrame: Updated dataset.
    """
    # Get the tile id
    tile_id = tile.index[0]

    # Get the tile bbox
    tile_bbox = tile['geometry'].values[0].bounds

    # Get the sentinel 2 id
    sentinel_2_id = S2_tile_name

    # convert accepted polygons to a multipolygon
    accepted_multipolygon = shapely.geometry.MultiPolygon(accepted_polygons['geometry'].values)

    accepted_polygons_gdf = gpd.GeoDataFrame([{
        "tile_id": tile_id,
        "tile_bbox": tile_bbox,
        "sentinel_2_id": sentinel_2_id,
        "geometry": accepted_multipolygon, 
        "source_dataset": accepted_source_dataset
    }], crs="EPSG:4326")

    # Concatenate the dataset with the new row
    dataset = pd.concat([dataset, accepted_polygons_gdf], ignore_index=True)

    return dataset


def main():
    """
    Main function to run the Mine Segmentation App.
    """
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

    # Get a random tile if not already selected
    if "tile" not in st.session_state:
        st.session_state.tile = get_random_tile(mining_area_tiles)

    # Visualize the tile
    maus_gdf_filtered, tang_gdf_filtered, s2_tile_id = visualize_tile(st.session_state.tile, maus_gdf, tang_gdf, stac_reader, year)

    # Create a layout with 4 columns
    col1, col2, col3, col4 = st.columns(4)

    # Add a button to refresh the tile in the first column
    with col1:
        if st.button("Refresh Tile", key="refresh"):
            # Get a random tile
            new_tile = get_random_tile(mining_area_tiles)
            st.session_state.tile = new_tile
                    
    # Add buttons for accepting maus and tang polygons in the second and third columns
    with col2:
        if st.button(":blue-background[Accept Maus]", key="maus"):
            dataset = accept_polygons(st.session_state.tile, maus_gdf_filtered, "maus", s2_tile_id, dataset)
            st.success("Polygons by Maus (blue) accepted successfully")
    with col3:
        if st.button(":red-background[Accept Tang]", key="tang"):
            dataset = accept_polygons(st.session_state.tile, tang_gdf_filtered, "tang", s2_tile_id, dataset)
            st.success("Polygons by Tang (red) accepted successfully")

    # Save the dataset
    with col4:
        if st.button("Save Dataset", key="save"):
            dataset.to_file(DATASET, driver="GPKG")
            st.success("Dataset saved successfully")

    # Undo button deleting the last row
    if st.button("Undo", key="undo"):
        dataset = dataset.iloc[:-1]
        st.warning("Last row deleted")
        
    # Display the dataset
    st.write(dataset)



if __name__ == "__main__":
    main()