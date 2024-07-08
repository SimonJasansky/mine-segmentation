import os

if __name__ == '__main__':
    # add masks and do postprocessing
    os.system("python src/data/postprocess_dataset.py")

    # persist images to disk
    os.system("python src/data/persist_pixels_masks.py data/processed/files preferred_polygons")

    # make the chips for model training
    os.system("python src/data/make_chips.py data/processed/files data/processed/chips 512")