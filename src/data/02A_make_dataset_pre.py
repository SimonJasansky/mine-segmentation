import os

if __name__ == '__main__':
    # Download external data
    os.system("python src/data/01_get_mining_polygons.py")

    # create the mining areas
    os.system("python src/data/02_make_tiles.py")
