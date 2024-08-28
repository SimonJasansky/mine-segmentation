import os

if __name__ == '__main__':
    # add masks and do postprocessing
    os.system("python src/data/03_postprocess_dataset.py")

    # filter dataset to fit requirements
    os.system("python src/data/04_filter_and_split_dataset.py preferred_polygons --val_ratio 0.2 --test_ratio 0.1 --only_valid_surface_mines")

    # persist images to disk
    os.system("python src/data/05_persist_pixels_masks.py data/processed/files")

    # make the chips for model training
    # for CLAY and CNN models
    # os.system("python src/data/06_make_chips.py data/processed/files data/processed/chips 512 npy --must_contain_mining")
    # for SAMGEO
    os.system("python src/data/06_make_chips.py data/processed/files data/processed/chips 1024 tif --must_contain_mining --normalize")