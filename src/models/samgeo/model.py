from samgeo import split_raster, array_to_image
from samgeo.text_sam import LangSAM
import os
import torch
import numpy as np
from PIL import Image
import glob
import rasterio
from pathlib import Path


from src.data.get_satellite_images import ReadSTAC
from src.visualization.visualize import plot_multiple_masks_on_images, plot_mask_on_image, plot_S2_geotiff_rgb
from src.utils import geotiff_to_PIL, normalize_geotiff, merge_geotiffs


class MineSamGeo:
    def __init__(
        self, 
        chips_dir,
        mask_dir,
        output_dir,
        model_type="vit_h", 
    ):
        self.model = LangSAM(model_type=model_type)
        self.chips_dir = chips_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        chip_files = glob.glob(os.path.join(self.chips_dir, "*.tif"))
        self.chip_files = chip_files
        self.num_chips = len(chip_files)


    def get_chip_path(self, idx):
        """
        Load normalized chip path (.tif) from chip dir
        """

        if idx >= len(self.chip_files):
            raise ValueError("Index out of range")
        
        # load chip path
        chip_path = self.chip_files[idx]
        self.model.source = chip_path

        return chip_path

    def predict_dino(
        self, 
        chip_path,
        text_prompt, 
        box_threshold, 
        text_threshold,
    ):
        """
        Predict on chip using DINO model

        Args:
            chip_path (str): Path to the chip (.tif)
            text_prompt (str): Text prompt for model
            box_threshold (float): Threshold for bounding box
            text_threshold (float): Threshold for text

        Returns:
            tuple: Tuple containing boxes, logits, and phrases.
        """

        # Load the georeferenced image
        with rasterio.open(chip_path) as src:
            chip_np = src.read().transpose(
                (1, 2, 0)
            )  # Convert rasterio image to numpy array
            self.model.transform = src.transform  # Save georeferencing information
            self.model.crs = src.crs  # Save the Coordinate Reference System
            chip_pil = Image.fromarray(
                chip_np[:, :, :3]
            )  # Convert numpy array to PIL image, excluding the alpha channel

        # predict
        boxes, logits, phrases = self.model.predict_dino(
            chip_pil, text_prompt, box_threshold, text_threshold
        )

        # filter out bounding boxes that are almost the size of the image
        w, h = chip_pil.size
        boxes = boxes.tolist()
        logits = logits.tolist()

        for i, box in enumerate(boxes):
            # check if first two coords are within 5% of image size, and last two coords are within 95% of image size
            if box[0] < 0.05 * w and box[1] < 0.05 * h and box[2] > 0.95 * w and box[3] > 0.95 * h:
                boxes.pop(i)
                logits.pop(i)
                phrases.pop(i)

            # also remove very large boxes that cover more than 75% of the image
            elif (box[2] - box[0]) * (box[3] - box[1]) > 0.75 * w * h:
                boxes.pop(i)
                logits.pop(i)
                phrases.pop(i)

        boxes = torch.tensor(boxes)
        logits = torch.tensor(logits)

        self.chip_np = chip_np
        self.chip_pil = chip_pil
        self.model.image = chip_pil
        self.boxes = boxes
        self.logits = logits
        self.phrases = phrases

        return boxes, logits, phrases

    def predict_sam(self, dtype=np.uint8, mask_multiplier=255, output=None):
        """
        Predict on chip using SAM model

        Args:
            dtype (numpy.dtype, optional): Data type for the mask overlay. Defaults to np.uint8.
            mask_multiplier (int, optional): Multiplier for the mask overlay. Defaults to 255.
            output (str, optional): Path to save the mask overlay image. Defaults to None.

        Returns:
            None
        """

        masks = torch.tensor([])
        if len(self.boxes) > 0:
            masks = self.model.predict_sam(self.chip_pil, self.boxes)
            masks = masks.squeeze(1)

        if self.boxes.nelement() == 0:  # No "object" instances found
            print("No objects found in the image.")
            return
        else:
            # Create an empty image to store the mask overlays
            mask_overlay = np.zeros_like(
                self.chip_np[..., 0], dtype=dtype
            )  # Adjusted for single channel

            for i, (box, mask) in enumerate(zip(self.boxes, masks)):
                # Convert tensor to numpy array if necessary and ensure it contains integers
                if isinstance(mask, torch.Tensor):
                    mask = (
                        mask.cpu().numpy().astype(dtype)
                    )  # If mask is on GPU, use .cpu() before .numpy()
                mask_overlay += ((mask > 0) * (i + 1)).astype(
                    dtype
                )  # Assign a unique value for each mask

            # Normalize mask_overlay to be in [0, 255]
            mask_overlay = (
                mask_overlay > 0
            ) * mask_multiplier  # Binary mask in [0, 255]

        if output is not None:
            array_to_image(mask_overlay, output, self.source, dtype=dtype)

        self.masks = masks
        self.prediction = mask_overlay

        # save predictions to model
        self.model.masks = masks
        self.model.boxes = self.boxes
        self.model.phrases = self.phrases
        self.model.logits = self.logits
        self.model.prediction = mask_overlay

    def predict(self, chip_path, text_prompt, box_threshold, text_threshold):
        """
        Predict on chip using DINO and SAM models

        Args:
            chip_path (str): Path to the chip (.tif)
            text_prompt (str): Text prompt for model
            box_threshold (float): Threshold for bounding box
            text_threshold (float): Threshold for text

        Returns:
            None
        """

        # predict with DINO
        boxes, logits, phrases = self.predict_dino(chip_path, text_prompt, box_threshold, text_threshold)

        # predict with SAM
        self.predict_sam()

    def show_preds_and_mask(self, chip_path):
        # first, plot the predictions
        self.model.show_anns(
            cmap="Blues",
            box_color="red",
            title="Automatic Segmentation of Mining Areas",
            blend=True,
            alpha=0.1
            )

        # then, plot the mask
        # get mask path
        mask_path = Path(chip_path).stem.replace("_img", "_mask")
        mask_path = os.path.join(self.mask_dir, mask_path) + ".tif"
        
        # show the mask
        # TODO 

    def batch_predict(self):
        pass

    def show_results_batch(self):
        pass

    def batch_metrics(self):
        # IoU, F1, Precision, Recall
        pass