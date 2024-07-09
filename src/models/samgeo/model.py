from samgeo import split_raster
from samgeo.text_sam import LangSAM
import os
import torch
import numpy as np
from PIL import Image
import glob

from src.data.get_satellite_images import ReadSTAC
from src.visualization.visualize import plot_multiple_masks_on_images, plot_mask_on_image, plot_S2_geotiff_rgb
from src.utils import geotiff_to_PIL, normalize_geotiff, merge_geotiffs


class MineSamGeo:
    def __init__(
        self, 
        model_type="vit_h", 
        chips_dir=None, 
        output_dir=None,
        interim_dir=None,
    ):
        self.model = LangSAM(model_type=model_type)
        self.chips_dir = chips_dir
        self.output_dir = output_dir
        self.interim_dir = interim_dir

    def load_chip(self, idx):
        """
        Load chip (np.array) from chip dir, normalize, and return PIL image
        """
        # get all files in chip dir
        chip_files = glob.glob(os.path.join(self.chips_dir, "*.npy"))

        if idx >= len(chip_files):
            raise ValueError("Index out of range")
        
        # load chip
        chip_array = np.load(chip_files[idx])

        # normalize chip
        chip_array = (chip_array - chip_array.min()) / (chip_array.max() - chip_array.min()) * 255
        chip_array = chip_array.astype(np.uint8)

        # convert to PIL image
        chip_pil = Image.fromarray(chip_array.transpose(1, 2, 0))

        return chip_pil

    def predict_dino(
        self, 
        chip,
        text_prompt, 
        box_threshold, 
        text_threshold,
    ):
        """
        Predict on chip using DINO model

        Args:
            chip (Image): Input PIL Image.
            text_prompt (str): Text prompt for model
            box_threshold (float): Threshold for bounding box
            text_threshold (float): Threshold for text

        Returns:
            tuple: Tuple containing boxes, logits, and phrases.
        """
        # get bounds of the chip
        chip_bounds = chip.bounds

        # predict
        prediction = self.model.predict_dino(chip, text_prompt, box_threshold, text_threshold)

        # filter out large boxes that are almost the size of the image
        boxes = prediction["boxes"].tolist()
        coords = rowcol_to_xy(self.source, boxes=boxes, dst_crs=dst_crs, **kwargs)




        return prediction

    def predict_sam(self):
        pass

    def show_results(self):
        pass

    def batch_metrics(self):
        # IoU, F1, Precision, Recall
        pass