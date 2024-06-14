import os

if __name__ == '__main__':
    # Download external data
    os.system("python src/data/get_mining_polygons.py")

    # create the mining areas
    os.system("python src/data/make_mining_areas.py")
