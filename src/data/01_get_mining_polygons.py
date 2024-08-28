import requests
import os
import subprocess
import shutil

# Script to download mining polygons from Maus et al. (2022) and Tang et al (2023)

URL_MAUS = "https://download.pangaea.de/dataset/942325/files/global_mining_polygons_v2.gpkg"
URL_MAUS_RASTER = "https://download.pangaea.de/dataset/942325/files/global_miningarea_v2_5arcminute.tif"
URL_TANG = "https://zenodo.org/records/6806817/files/Supplementary%201%EF%BC%9Amine%20area%20polygons.rar?download=1"
URL_MGRS = "https://mgrs-data.org/data/mgrs_index_ftp_link.zip"

maus_file_path = "data/external/maus_mining_polygons.gpkg"
maus_raster_file_path = "data/external/maus_mining_raster.tif"
tang_file_path = "data/external/tang_mining_polygons.rar"
mgrs_file_path = "data/external/mgrs_index_ftp_link.zip"


def download_file(url, file_path):
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Downloading file from {url} to {file_path}")
        with open(file_path, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully.")
    else:
        print("Failed to download the file.")

def extract_rar(file_path, extract_path):
    try:
        subprocess.run(["unrar", "x", "-o+", file_path, extract_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while extracting {file_path}: {str(e)}")

def extract_zip(file_path, extract_path):
    try:
        shutil.unpack_archive(file_path, extract_path)
    except Exception as e:
        print(f"Error occurred while extracting {file_path}: {str(e)}")

def download_and_extract(url, file_path, extract_path = None):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist yet, starting download...")
        download_file(url, file_path)
        if file_path.endswith(".rar"):
            extract_rar(file_path, extract_path)
        elif file_path.endswith(".zip"):
            extract_zip(file_path, extract_path)
    else:
        print(f"File {file_path} already exists.")


if __name__ == "__main__":

    download_and_extract(URL_MAUS, maus_file_path)

    download_and_extract(URL_MAUS_RASTER, maus_raster_file_path)

    download_and_extract(URL_TANG, tang_file_path, "data/external")

    download_and_extract(URL_MGRS, mgrs_file_path, "data/external")
    
    #########################
    # Post-processing steps #
    #########################

    # Fix the folder names in Tang's data
    parent_dir = "data/external/tang_mining_polygons"
    os.rename("data/external/Supplementary 1：mine area polygons", parent_dir)

    # Fix the subfolder names
    for old_dir_name in os.listdir(parent_dir):
        # Skip files
        if not os.path.isdir(os.path.join(parent_dir, old_dir_name)):
            continue

        # Replace spaces with underscores and remove colons
        new_dir_name = old_dir_name.replace(' ', '_')

        # Create the full paths to the old and new directories
        old_dir_path = os.path.join(parent_dir, old_dir_name)
        new_dir_path = os.path.join(parent_dir, new_dir_name)

        # Rename the directory
        shutil.move(old_dir_path, new_dir_path)

    print("Mining polygons downloaded and extracted successfully.")
