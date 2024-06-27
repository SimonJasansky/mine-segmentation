import streamlit as st
import geopandas as gpd
import pandas as pd
import shapely
import os
import pystac

from src.data.get_satellite_images import ReadSTAC
from src.utils import calculate_dimensions_km

import leafmap.foliumap as leafmap

TILES = "data/interim/tiles.gpkg"
MAUS_POLYGONS = "data/external/maus_mining_polygons.gpkg"
TANG_POLYGONS = "data/external/tang_mining_polygons/74548_mine_polygons/74548_projected.shp"
DATASET = "data/raw/mining_tiles_with_masks.gpkg"

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
            - dataset (GeoDataFrame): Final Dataset containing the accepted polygons.
    """
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
        columns = ["tile_id", "tile_bbox", "sentinel_2_id", "geometry", "source_dataset", "timestamp"]
        dataset = gpd.GeoDataFrame(columns=columns)

        # write to file
        dataset.to_file(DATASET, driver="GPKG")

    # Initialize the STAC reader
    api_url="https://planetarycomputer.microsoft.com/api/stac/v1"
    stac_reader = ReadSTAC(
        api_url=api_url, 
        collection = "sentinel-2-l2a",
        data_dir="streamlit_app/data"
    )

    mining_area_tiles = gpd.read_file(TILES, layer="mining_areas_square")

    return maus_gdf, tang_gdf, stac_reader, dataset, mining_area_tiles


def set_random_tile():
    """
    Get a random mining tile.

    Returns:
        GeoDataFrame: Random mining tile.
    """
    # Refresh the tile
    mining_area_tiles = gpd.read_file(TILES)
    dataset = gpd.read_file(DATASET)

    # take only tiles that are not yet in the dataset
    mining_area_tiles = mining_area_tiles[~mining_area_tiles.index.isin(dataset["tile_id"].values)]
    random_tile = mining_area_tiles.sample(n=1)

    st.session_state.tile = random_tile

    # Reset the year to 2019
    st.session_state.year = 2019


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

    # create three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        # Display the tile dataframe
        st.dataframe(pd.DataFrame(tile.drop(columns="geometry")))

    with col2:
        # project the tile, and calculate the extent
        width_km, height_km = calculate_dimensions_km(tile_geometry)
        st.write(f"Tile extents: {width_km:.2f} km x {height_km:.2f} km")


    # get the least cloudy sentinel image for the tile
    items = stac_reader.get_items(
        bbox=bbox,
        timerange=f'{year}-01-01/{year}-12-31',
        max_cloud_cover=15
    )

    if len(items) < 1: 
        st.error("No S2 images found for this tile. Please refresh the tile or change to another year.")

    # Get the least cloudy images
    least_cloudy_item = stac_reader.filter_item(items, "least_cloudy", full_overlap = True)

    if isinstance(least_cloudy_item, pystac.ItemCollection):
        least_cloudy_item = least_cloudy_item[0]

    with col3:
        # Display the cloud coverage
        st.write(f"Cloud coverage: {least_cloudy_item.properties['eo:cloud_cover']}%")
    
    url = least_cloudy_item.assets["visual"].href
    s2_tile_id = least_cloudy_item.id
    
    # Create a Map
    m = leafmap.Map(center=[tile_geometry.centroid.y, tile_geometry.centroid.x], zoom=10)

    # optionally add high resolution satellite imagery and toggle it off by default
    with st.sidebar:
        if st.checkbox("Add Google High-Res Satellite Imagery", key="satellite", value=False):
            m.add_basemap("SATELLITE")

    m.add_cog_layer(url, name="Sentinel-2")
    
    # visualize the tile boundaries
    style_tile = {
        "stroke": True,
        "color": "orange",
        "weight": 2,
        "opacity": 1,
        "fill": False,
    }
    m.add_gdf(tile, layer_name="tile_bbox", style=style_tile)

    # Filter the polygons that are included 
    maus_gdf_filtered = maus_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    tang_gdf_filtered = tang_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    # Crop the multipolygon to the tile bbox
    maus_gdf_filtered["geometry"] = maus_gdf_filtered["geometry"].apply(lambda x: x.intersection(tile_geometry))
    tang_gdf_filtered["geometry"] = tang_gdf_filtered["geometry"].apply(lambda x: x.intersection(tile_geometry))

    # Check that all are of type polygon and not multipolygon
    maus_gdf_filtered = maus_gdf_filtered[maus_gdf_filtered["geometry"].apply(lambda x: x.geom_type == "Polygon")]
    tang_gdf_filtered = tang_gdf_filtered[tang_gdf_filtered["geometry"].apply(lambda x: x.geom_type == "Polygon")]

    style_maus = {
        "stroke": True,
        "color": "blue",
        "weight": 2,
        "opacity": 1,
        "fill": True,
        "fillColor": "blue",
        "fillOpacity": 0.1,
    }

    style_tang = {
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
        m.add_gdf(maus_gdf_filtered, layer_name="maus_gdf", style=style_maus)
    if not tang_gdf_filtered.empty:
        m.add_gdf(tang_gdf_filtered, layer_name="tang_gdf", style=style_tang)

    m.to_streamlit()

    return maus_gdf_filtered, tang_gdf_filtered, s2_tile_id


def accept_polygons(accepted_polygons, accepted_source_dataset, S2_tile_name):
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
    tile = st.session_state.tile 

    # Get the tile id
    tile_id = tile.index[0]

    # Get the tile bbox as geojson
    tile_geojson = shapely.geometry.mapping(tile['geometry'].values[0])

    # Get the sentinel 2 id
    sentinel_2_id = S2_tile_name

    # Fix any potential issues with the polygons (e.g. self-intersections)
    # accepted_polygons["geometry"] = accepted_polygons.buffer(0)

    # convert accepted polygons to a multipolygon
    accepted_multipolygon = shapely.geometry.MultiPolygon(accepted_polygons['geometry'].values)

    accepted_polygons_gdf = gpd.GeoDataFrame([{
        "tile_id": tile_id,
        "tile_bbox": tile_geojson,
        "sentinel_2_id": sentinel_2_id,
        "geometry": accepted_multipolygon, 
        "source_dataset": accepted_source_dataset,
        "timestamp": pd.Timestamp.now()
    }], crs="EPSG:4326")

    dataset = gpd.read_file(DATASET)

    # Concatenate the dataset with the new row
    dataset = pd.concat([dataset, accepted_polygons_gdf], ignore_index=True)

    return dataset


##################
# Main interface #
##################


def main():
    """
    Main function to run the Mine Area Mask Selection App.
    """
    st.title('Mine Area Mask Selection App')
    st.text("""
        This app allows you to accept or reject polygons for training a mine segmentation model. 
        Data Sources:
        - Maus et al: https://doi.pangaea.de/10.1594/PANGAEA.942325
        - Tang et al: https://zenodo.org/doi/10.5281/zenodo.6806816 
    """)


    # Load data
    maus_gdf, tang_gdf, stac_reader, dataset, mining_area_tiles = load_data()

    # Get a random tile if not already selected
    if "tile" not in st.session_state:
        set_random_tile()

    # Add a streamlit radio button for selecting the year
    with st.sidebar:
        st.radio("Select Year", list(range(2016, 2023)), index=3, key="year")

    st.button("Refresh Tile", on_click=set_random_tile)

    # Visualize the tile
    maus_gdf_filtered, tang_gdf_filtered, s2_tile_id = visualize_tile(st.session_state.tile, maus_gdf, tang_gdf, stac_reader, st.session_state.year)
    
    # # Get the custom polygon from the leafmap 
    # m.save_draw_features("streamlit_app/data/custom_features.geojson")

    # Create layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(":blue-background[Accept Maus]", key="maus"):
            dataset = accept_polygons(maus_gdf_filtered, "maus", s2_tile_id)
            # save dataset to file
            dataset.to_file(DATASET, driver="GPKG")
            st.success("Polygons by Maus (blue) accepted successfully")

    with col2:
        if st.button(":red-background[Accept Tang]", key="tang"):
            dataset = accept_polygons(tang_gdf_filtered, "tang", s2_tile_id)
            # save dataset to file
            dataset.to_file(DATASET, driver="GPKG")
            st.success("Polygons by Tang (red) accepted successfully")

    with col3:
        # Reject the tile and the polygons
        if st.button(":x: Reject Tile", key="reject"):
            # combine maus and tang polygons
            all_polygons = gpd.GeoDataFrame(pd.concat([maus_gdf_filtered, tang_gdf_filtered], ignore_index=True))

            dataset = accept_polygons(all_polygons, "rejected", s2_tile_id)
            # save dataset to file
            dataset.to_file(DATASET, driver="GPKG")
            st.success("Tile and polygons rejected successfully")

    # with col4:
    #     if st.button("Accept both", key="custom"):
    #         # Get the custom polygon from the leafmap 
    #         m.save_draw_features("streamlit_app/data/custom_features.geojson")
            
    #         # read the file
    #         gdf = gpd.read_file("streamlit_app/data/custom_features.geojson")

    #         # Ensure that the custom polygon is in the same CRS as the tile
    #         custom_gdf = gdf.to_crs(st.session_state.tile.crs)

    #         # Ensure that the custom polygon is within the tile by taking the intersection
    #         custom_gdf["geometry"] = custom_gdf["geometry"].apply(lambda x: x.intersection(st.session_state.tile['geometry'].values[0]))

    #         dataset = accept_polygons(st.session_state.tile, custom_gdf, "custom", s2_tile_id)
    #         # save dataset to file
    #         dataset.to_file(DATASET, driver="GPKG")
    #         st.success("Custom polygons accepted successfully")

    # Add section separator
    st.write("---")

    # Undo button deleting the last row
    if st.button("Undo: Delete last Row", key="undo"):
        dataset_copy = gpd.read_file(DATASET)
        dataset_copy = dataset_copy.iloc[:-1]
        dataset_copy.to_file(DATASET, driver="GPKG")
        st.warning("Last row deleted")

    # Display the last 10 rows of the dataset
    dataset_copy = gpd.read_file(DATASET)
    dataset_copy['str_geom'] = dataset_copy['geometry'].apply(shapely.wkt.dumps)
    st.dataframe(dataset_copy.drop(columns="geometry"))

    # Add progress bar for the dataset
    n_tiles_reviewed = len(dataset_copy)
    n_tiles_to_review = len(mining_area_tiles)
    st.write(f"Progress: {n_tiles_reviewed}/{n_tiles_to_review} tiles reviewed.",
            f"{n_tiles_reviewed / n_tiles_to_review:.2%} completed.")
    st.progress(n_tiles_reviewed / n_tiles_to_review)

if __name__ == "__main__":
    main()