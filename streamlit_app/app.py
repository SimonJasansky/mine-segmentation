import streamlit as st
st.set_page_config(layout="wide")
import geopandas as gpd
import pandas as pd
import shapely
import os

from src.data.get_satellite_images import ReadSTAC
from src.utils import calculate_dimensions_km

import leafmap.foliumap as leafmap
# from streamlit_folium import st_folium

TILES = "data/interim/tiles.gpkg"
MAUS_POLYGONS = "data/external/maus_mining_polygons.gpkg"
TANG_POLYGONS = "data/external/tang_mining_polygons/74548_mine_polygons/74548_projected.shp"
DATASET = "data/raw/mining_tiles_with_masks.gpkg"
ERRONEOUS_TILES = "data/interim/erroneous_tiles.gpkg"

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
    if not os.path.exists(DATASET):
        # Create the dataset
        columns_tiles = ["tile_id", "s2_tile_id", "source_dataset", "preferred_dataset", "minetype1", "minetype2", "comment", "timestamp", "geometry"]
        columns_maus = ["tile_id", "geometry"]
        columns_tang = ["tile_id", "geometry"]
        tiles = gpd.GeoDataFrame(columns=columns_tiles, crs="EPSG:4326", geometry="geometry")
        maus = gpd.GeoDataFrame(columns=columns_maus, crs="EPSG:4326", geometry="geometry")
        tang = gpd.GeoDataFrame(columns=columns_tang, crs="EPSG:4326", geometry="geometry")

        # write to file
        tiles.to_file(DATASET, driver="GPKG", layer="tiles")
        maus.to_file(DATASET, driver="GPKG", layer="maus_polygons")
        tang.to_file(DATASET, driver="GPKG", layer="tang_polygons")

    # Check if the erroneous tiles file exists, and if not, create it
    if not os.path.exists(ERRONEOUS_TILES):
        # Create the dataset
        columns_tiles = ["tile_id", "timestamp", "geometry"]
        tiles = gpd.GeoDataFrame(columns=columns_tiles, crs="EPSG:4326", geometry="geometry")
        tiles.to_file(ERRONEOUS_TILES, driver="GPKG", layer="tiles")

    # Initialize the STAC reader
    api_url="https://planetarycomputer.microsoft.com/api/stac/v1"
    
    stac_reader = ReadSTAC(
        api_url=api_url, 
        collection = "sentinel-2-l2a",
        data_dir="streamlit_app/data"
    )

    mining_area_tiles = gpd.read_file(TILES, layer="mining_areas_square")

    return maus_gdf, tang_gdf, stac_reader, mining_area_tiles


def set_random_tile(refresh_counter=0):
    """
    Get a random mining tile.

    Returns:
        GeoDataFrame: Random mining tile.
    """
    # Refresh the tile
    mining_area_tiles = gpd.read_file(TILES)
    dataset = gpd.read_file(DATASET)
    erroneous_tiles = gpd.read_file(ERRONEOUS_TILES)

    # take only tiles that are not yet in the dataset or in the erroneous tiles
    forbidden_indices = dataset["tile_id"].astype(int).values.tolist() + erroneous_tiles["tile_id"].astype(int).values.tolist()
    mining_area_tiles = mining_area_tiles[~mining_area_tiles.index.isin(forbidden_indices)]
    random_tile = mining_area_tiles.sample(n=1)

    # check if the tile overlaps with an already annotated & accepted tile 
    dataset = dataset[dataset["preferred_dataset"].isin(["maus", "tang"])]
    tiles_multipolygon = dataset["geometry"].unary_union
    if random_tile["geometry"].values[0].intersects(tiles_multipolygon):
        refresh_counter +=1
        if refresh_counter > 10:
            st.warning("More than 10 refreshes needed to find a non-overlapping tile. Please check the dataset soon. ")
            refresh_counter = 0
        set_random_tile(refresh_counter=refresh_counter)
    else:
        st.session_state.tile = random_tile

        # Reset the year to 2019
        st.session_state.year = 2019
        st.session_state.comment = ""
        st.session_state.minetype1 = "Surface"
        st.session_state.minetype2 = None
        st.session_state.preferred_dataset = None
        st.session_state.satellite = False

@st.cache_data
def load_least_cloudy_item(_stac_reader, bbox, year):
    
    # get the least cloudy sentinel image for the tile
    items = _stac_reader.get_items(
        bbox=bbox,
        timerange=f'{year}-01-01/{year}-12-31',
        max_cloud_cover=15
    )

    if len(items) < 1: 
        st.error("No S2 images found for this tile. ")

    try:
        # Get the least cloudy images
        least_cloudy_item = _stac_reader.filter_item(items, "least_cloudy", full_overlap=True)
    except ValueError:
        st.warning("A ValueError occurred. No S2 images that fully overlap with this tile are found. Rejected tile. Please refresh.")

        # add to erroneous tiles
        add_erroneous_tile(st.session_state.tile.index[0])

    return least_cloudy_item


def visualize_tile(tile, maus_gdf, tang_gdf, least_cloudy_item):
    """
    Visualize a tile.

    Args:
        tile (GeoDataFrame): Tile to visualize.
        maus_gdf (GeoDataFrame): Maus dataset.
        tang_gdf (GeoDataFrame): Tang dataset.
        least_cloudy_item (Item): Least cloudy Sentinel-2 item.
        edit_mode (bool): Whether to enable edit mode.

    Returns:
        Tuple: A tuple containing the filtered Maus dataset, filtered Tang dataset, and the Sentinel-2 tile ID.
    """
    # Get the geometry of the random tile
    tile_geometry = tile['geometry'].values[0]

    bbox = tile_geometry.bounds
    try:
        url = least_cloudy_item.assets["visual"].href
    except Exception as e:
        st.warning("An error occurred. Error message: {}".format(str(e)))

        # Add to erroneous tiles
        add_erroneous_tile(tile.index[0])

    s2_tile_id = least_cloudy_item.id
    
    # Create a Map
    m = leafmap.Map(center=[tile_geometry.centroid.y, tile_geometry.centroid.x], zoom=10)

    # optionally add high resolution satellite imagery and toggle it off by default
    with st.sidebar:
        if st.checkbox("Add Google High-Res Satellite Imagery", key="satellite", value=False):
            m.add_basemap("SATELLITE")

        if st.checkbox("Edit Mode", key="edit_mode", value=False):
            edit_mode = True
        else:
            edit_mode = False
    
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

    # extract country name
    if not maus_gdf_filtered.empty:
        country = maus_gdf_filtered["COUNTRY_NAME"].values[0]

    # make them valid
    tang_gdf_filtered["geometry"] = tang_gdf_filtered["geometry"].apply(lambda x: shapely.make_valid(x))
    maus_gdf_filtered["geometry"] = maus_gdf_filtered["geometry"].apply(lambda x: shapely.make_valid(x))

    # create new gdf with only one row that holds the combined multipolygon
    maus_multipolygon = maus_gdf_filtered["geometry"].unary_union
    tang_multipolygon = tang_gdf_filtered["geometry"].unary_union

    maus_gdf_filtered = gpd.GeoDataFrame(geometry=[maus_multipolygon], crs="EPSG:4326")
    tang_gdf_filtered = gpd.GeoDataFrame(geometry=[tang_multipolygon], crs="EPSG:4326")

    # get intersection of multipolygon with tile bbox
    maus_gdf_filtered["geometry"] = maus_gdf_filtered["geometry"].apply(lambda x: x.intersection(tile_geometry))
    tang_gdf_filtered["geometry"] = tang_gdf_filtered["geometry"].apply(lambda x: x.intersection(tile_geometry))

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

    if edit_mode:
        st_data = m.to_streamlit(width=100, height=800, bidirectional=True)
    else:
        # with Screen
        # m.to_streamlit(width=1200, height=900, bidirectional=False)
        # on Laptop
        m.to_streamlit(width=970, height=900, bidirectional=False, add_layer_control=True)
        st_data = None

    # create three columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Display the tile dataframe
        st.dataframe(pd.DataFrame(tile.drop(columns="geometry")))

    with col2:
        # project the tile, and calculate the extent
        width_km, height_km = calculate_dimensions_km(tile_geometry)
        st.write(f"Tile extents: {width_km:.2f} km x {height_km:.2f} km")

    with col3:
        # Display the cloud coverage
        st.write(f"Cloud coverage: {least_cloudy_item.properties['eo:cloud_cover']}%")

    with col4:
        # Display the country
        if country:
            st.write(f"Country: {country}")

    st.write(f"Sentinel-2 Tile ID: {s2_tile_id}")

    return maus_gdf_filtered, tang_gdf_filtered, s2_tile_id, st_data


def accept_polygons(
        maus_gdf_filtered: gpd.GeoDataFrame,
        tang_gdf_filtered: gpd.GeoDataFrame,
        accepted_source_dataset: str,
        s2_tile_id: str,
    ):
    """
    Accept polygons and update the dataset.

    Args:
        tile (GeoDataFrame): Tile containing the polygons.
        accepted_polygons (GeoDataFrame): Accepted polygons.
        accepted_source_dataset (str): Source dataset of the accepted polygons.
        s2_tile_id (str): Sentinel-2 tile name.
        dataset (GeoDataFrame): Dataset to update.

    Returns:
        GeoDataFrame: Updated dataset.
    """
    tile = st.session_state.tile 

    if st.session_state.preferred_dataset == ":large_blue_circle: :blue-background[Maus]":
        preferred_dataset = "maus"
    elif st.session_state.preferred_dataset == ":red_circle: :red-background[Tang]":
        preferred_dataset = "tang"
    elif st.session_state.preferred_dataset == "None":
        preferred_dataset = "none"
    else:
        st.error("Please select a preferred dataset")

    # check if minetype2 is selected
    if st.session_state.minetype2 is None:
        st.error("Please select a mine type 2 (Artisanal or Industrial)")

    # check if preferred and accepted datasets are mutually exclusive
    if accepted_source_dataset == "maus" and preferred_dataset == "tang":
        st.error("Maus dataset cannot be preferred if Tang dataset is accepted")
    elif accepted_source_dataset == "tang" and preferred_dataset == "maus":
        st.error("Tang dataset cannot be preferred if Maus dataset is accepted")

    tiles_dict = {
        "tile_id": tile.index[0],
        "s2_tile_id": s2_tile_id,
        "source_dataset": accepted_source_dataset,
        "preferred_dataset": preferred_dataset,
        "minetype1": st.session_state.minetype1,
        "minetype2": st.session_state.minetype2,
        "comment": st.session_state.comment,
        "timestamp": pd.Timestamp.now(),
        "geometry": tile["geometry"].values[0]
    }

    maus_dict = {
        "tile_id": tile.index[0],
        "geometry": None
    }

    tang_dict = {
        "tile_id": tile.index[0],
        "geometry": None
    }
    
    # assert that the gdf_filtered have only one row containing the multipolygon
    if len(maus_gdf_filtered) > 1:
        raise ValueError("Maus gdf_filtered contains more than one row")
    if len(tang_gdf_filtered) > 1:
        raise ValueError("Tang gdf_filtered contains more than one row")
    

    if accepted_source_dataset == "maus":
        # Create a Multipolygon from the polygons with unary_union
        maus_dict["geometry"] = maus_gdf_filtered["geometry"][0]
    elif accepted_source_dataset == "tang":
        tang_dict["geometry"] = tang_gdf_filtered["geometry"][0]
    elif accepted_source_dataset == "both":
        maus_dict["geometry"] = maus_gdf_filtered["geometry"][0]
        tang_dict["geometry"] = tang_gdf_filtered["geometry"][0]
    elif accepted_source_dataset == "rejected":
        pass
    else:
        raise ValueError("Invalid source dataset")
    
    accepted_tiles = gpd.GeoDataFrame([tiles_dict], crs="EPSG:4326")
    accepted_maus = gpd.GeoDataFrame([maus_dict], crs="EPSG:4326")
    accepted_tang = gpd.GeoDataFrame([tang_dict], crs="EPSG:4326")

    # Load the existing dataset
    tiles = gpd.read_file(DATASET, layer="tiles")
    maus = gpd.read_file(DATASET, layer="maus_polygons")
    tang = gpd.read_file(DATASET, layer="tang_polygons")
    
    # Concatenate the dataset with the new row
    tiles = pd.concat([tiles, accepted_tiles], ignore_index=True)
    maus = pd.concat([maus, accepted_maus], ignore_index=True)
    tang = pd.concat([tang, accepted_tang], ignore_index=True)

    # Write the dataset to the file
    tiles.to_file(DATASET, driver="GPKG", layer="tiles")
    maus.to_file(DATASET, driver="GPKG", layer="maus_polygons")
    tang.to_file(DATASET, driver="GPKG", layer="tang_polygons")

    
def add_erroneous_tile(tile_id):
    """
    Add an erroneous tile to the dataset.

    Args:
        tile_id (int): Tile ID to add.
    """
    # Load the erroneous tiles
    erroneous_tiles = gpd.read_file(ERRONEOUS_TILES)

    # Add the tile to the erroneous tiles
    error_tile_dict = {
        "tile_id": tile_id,
        "timestamp": pd.Timestamp.now(),
        "geometry": st.session_state.tile["geometry"].values[0]
    }

    # Create a GeoDataFrame from the dictionary
    error_tile_gdf = gpd.GeoDataFrame([error_tile_dict], crs="EPSG:4326")

    # Concatenate the dataset with the new row
    erroneous_tiles = pd.concat([erroneous_tiles, error_tile_gdf], ignore_index=True)

    # Write the dataset to the file
    erroneous_tiles.to_file(ERRONEOUS_TILES, driver="GPKG", layer="tiles")

    st.warning("Tile added to the erroneous tiles")

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
    maus_gdf, tang_gdf, stac_reader, mining_area_tiles = load_data()

    # Get a random tile if not already selected
    if "tile" not in st.session_state:
        set_random_tile()

    with st.sidebar:
        
        st.radio("Select Year", list(range(2016, 2022)), index=3, key="year")
        
    st.button("Refresh Tile", on_click=set_random_tile)

    col1, col2 = st.columns(spec=[0.75, 0.25])

    with col1:
        # Visualize the tile
        least_cloudy_item = load_least_cloudy_item(stac_reader, st.session_state.tile["geometry"].values[0].bounds, st.session_state.year)
        maus_gdf_filtered, tang_gdf_filtered, s2_tile_id, st_data = visualize_tile(st.session_state.tile, maus_gdf, tang_gdf, least_cloudy_item)

    with col2:
        
        # Add horizontal radio button for preferred source dataset
        st.radio("Preferred Source Dataset", 
                [":large_blue_circle: :blue-background[Maus]", 
                    ":red_circle: :red-background[Tang]", 
                    "None"], 
                index=None, key="preferred_dataset")

        # Add horizontal radio button for mine type 1
        st.radio("Mine Type 1", ["Surface", "Placer", "Brine & Evaporation Pond", "Underground"], index=0, key="minetype1")

        # Add horizontal radio button for mine type 2
        st.radio("Mine Type 2", ["Industrial", "Artisanal"], index=None, key="minetype2")

        # Add a text input for comments
        st.text_input("Comment", key="comment", placeholder="Comment")
    
        # Add section separator
        st.write("---")

        if st.button(":large_blue_circle: :blue-background[Accept Maus]", key="maus"):
            accept_polygons(maus_gdf_filtered, tang_gdf_filtered, accepted_source_dataset="maus", s2_tile_id=s2_tile_id)
            st.success("Polygons by Maus (blue) accepted successfully")

        if st.button(":red_circle: :red-background[Accept Tang]", key="tang"):
            accept_polygons(maus_gdf_filtered, tang_gdf_filtered, accepted_source_dataset="tang", s2_tile_id=s2_tile_id)
            st.success("Polygons by Tang (red) accepted successfully")

        # Accept both Maus' and Tang's polygons
        if st.button(":white_check_mark: Accept both", key="both"):
            accept_polygons(maus_gdf_filtered, tang_gdf_filtered, accepted_source_dataset="both", s2_tile_id=s2_tile_id)
            st.success("Both polygons (Maus & Tang) accepted successfully")

        # Reject the tile and the polygons
        if st.button(":x: Reject Tile", key="rejected"):
            accept_polygons(maus_gdf_filtered, tang_gdf_filtered, accepted_source_dataset="rejected", s2_tile_id=s2_tile_id)
            st.success("Tile and polygons rejected successfully")

    # Add section separator
    st.write("---")

    # Undo button deleting the last row
    if st.button("Undo: Delete last Row", key="undo"):
        tiles_copy = gpd.read_file(DATASET, layer="tiles")
        maus_copy = gpd.read_file(DATASET, layer="maus_polygons")
        tang_copy = gpd.read_file(DATASET, layer="tang_polygons")

        # Delete the last row
        tiles_copy = tiles_copy.iloc[:-1]
        maus_copy = maus_copy.iloc[:-1]
        tang_copy = tang_copy.iloc[:-1]

        # Write the dataset to the file
        tiles_copy.to_file(DATASET, driver="GPKG", layer="tiles")
        maus_copy.to_file(DATASET, driver="GPKG", layer="maus_polygons")
        tang_copy.to_file(DATASET, driver="GPKG", layer="tang_polygons")

        st.warning("Last row deleted")

    # Display the last 10 rows of the dataset
    tiles_copy = gpd.read_file(DATASET, layer="tiles")
    maus_copy = gpd.read_file(DATASET, layer="maus_polygons")
    tang_copy = gpd.read_file(DATASET, layer="tang_polygons")
    erroneous_tiles = gpd.read_file(ERRONEOUS_TILES)

    # Convert the geometry to WKT
    tiles_copy['geometry_wkt'] = tiles_copy['geometry'].apply(lambda x: shapely.wkt.dumps(x)).astype(str)
    maus_copy['geometry_wkt'] = maus_copy['geometry'].apply(lambda x: shapely.wkt.dumps(x)).astype(str)
    tang_copy['geometry_wkt'] = tang_copy['geometry'].apply(lambda x: shapely.wkt.dumps(x)).astype(str)
    erroneous_tiles['geometry_wkt'] = erroneous_tiles['geometry'].apply(lambda x: shapely.wkt.dumps(x)).astype(str)

    # Drop the geometry column
    tiles_copy = tiles_copy.drop(columns="geometry")
    maus_copy = maus_copy.drop(columns="geometry")
    tang_copy = tang_copy.drop(columns="geometry")
    erroneous_tiles = erroneous_tiles.drop(columns="geometry")

    # Add progress bar for the dataset
    n_tiles_reviewed = len(tang_copy)
    n_erroneous_tiles = len(erroneous_tiles)
    n_tiles_to_review = len(mining_area_tiles)

    # Progress without erroneous tiles
    st.write(f"Progress: {n_tiles_reviewed}/{n_tiles_to_review} tiles reviewed.",
            f"{n_tiles_reviewed / n_tiles_to_review:.2%} completed.")
    st.progress(n_tiles_reviewed / n_tiles_to_review)

    # Progress with erroneous tiles
    st.write(f"Progress (including erroneous tiles): {n_tiles_reviewed + n_erroneous_tiles}/{n_tiles_to_review} tiles reviewed.",
            f"{(n_tiles_reviewed + n_erroneous_tiles) / n_tiles_to_review:.2%} reviewed.")
    st.progress((n_tiles_reviewed + n_erroneous_tiles) / n_tiles_to_review)

    # Display the dataset
    st.dataframe(tiles_copy)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Maus Polygons")
        st.dataframe(maus_copy)
    with col2:
        st.write("Tang Polygons")
        st.dataframe(tang_copy)

if __name__ == "__main__":
    main()