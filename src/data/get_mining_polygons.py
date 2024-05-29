import requests
import os
import requests
import shutil
import zipfile
import patoolib

# Script to download mining polygons from Maus et al. (2022) and Tang et al (2023)

URL_MAUS = "https://download.pangaea.de/dataset/942325/files/global_mining_polygons_v2.gpkg"
URL_TANG = "https://zenodo.org/records/6806817/files/Supplementary%201%EF%BC%9Amine%20area%20polygons.rar?download=1"

def download_file(url, file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully.")
    else:
        print("Failed to download the file.")

def extract_rar(file_path, extract_path):
    patoolib.extract_archive(file_path, outdir=extract_path)

def download_and_extract(url, file_path, extract_path):
    download_file(url, file_path)
    if file_path.endswith(".rar"):
        extract_rar(file_path, extract_path)

# Download and extract Maus et al
geopackage_file_path = "/workspaces/mine-segmentation/data/external/global_mining_polygons.gpkg"
if not os.path.exists(geopackage_file_path):
    download_and_extract(URL_MAUS, geopackage_file_path, "/workspaces/mine-segmentation/data/external")
else:
    print("File already exists.")

# Download and extract Tang et al
rar_file_path = "/workspaces/mine-segmentation/data/external/mine_area_polygons.rar"
if not os.path.exists(rar_file_path):
    download_and_extract(URL_TANG, rar_file_path, "/workspaces/mine-segmentation/data/external")
else:
    print("File already exists.")
