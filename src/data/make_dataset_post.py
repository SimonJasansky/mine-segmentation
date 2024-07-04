import os

if __name__ == '__main__':
    # add masks and do postprocessing
    os.system("python src/data/postprocess_dataset.py")

    # persist small sample (6) to disk
    os.system("python src/data/persist_pixels_masks.py --limit 6 data/processed/files preferred_polygons")

    # optionally do chipping
    os.system("python src/models/clay/segment/preprocess_data.py data/processed/files data/processed/chips 512")