import os
from pathlib import Path
import glob
import warnings

import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from PIL import Image

from samgeo import split_raster, array_to_image
from samgeo.text_sam import LangSAM

from src.data.get_satellite_images import ReadSTAC
from src.visualization.visualization_funcs import plot_multiple_masks_on_images, plot_mask_on_image, plot_S2_geotiff_rgb
from src.utils import geotiff_to_PIL, normalize_geotiff, merge_geotiffs


class MineSamGeo:
    """
    Wrapper class for the SamGeo model
    """
    def __init__(
        self,
        chips_dir,
        mask_dir,
        output_dir,
        model_type="vit_h",
    ):
        """
        Initialize the model

        Args:
            chips_dir (str): Path to the chips directory. Chips must be normalized already.
            mask_dir (str): Path to the mask directory
            output_dir (str): Path to the output directory
            model_type (str, optional): Type of model to use. Defaults to "vit_h".
        """
        self.model = LangSAM(model_type=model_type)
        self.chips_dir = chips_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir

        chip_files = glob.glob(os.path.join(self.chips_dir, "*.tif"))
        self.chip_files = chip_files
        self.num_chips = len(chip_files)

        # save attributes to self.model
        self.model.source = None
        self.model.image = None
        self.model.transform = None
        self.model.crs = None
        self.model.boxes = None
        self.model.logits = None
        self.model.phrases = None


    def get_chip_path(self, idx):
        """
        Load path of one chip (.tif) from chip dir.
        """

        if idx >= len(self.chip_files):
            raise ValueError("Index out of range")

        # load chip path
        chip_path = self.chip_files[idx]
        self.model.source = chip_path

        return chip_path

    def get_mask_path(self, chip_path):
        """
        Get the mask path given the chip path
        """
        if chip_path is None:
            print("Please run predict() first.")
            return

        # get mask path
        mask_path = str(Path(self.mask_dir) / f"{Path(chip_path).stem}.tif")

        # replace _img with _mask
        mask_path = mask_path.replace("_img", "_mask")

        return mask_path


    def predict_dino(
        self,
        chip_path,
        text_prompt,
        box_threshold,
        text_threshold,
        box_size_threshold=0.5,
    ):
        """
        Predict on chip using DINO model

        Args:
            chip_path (str): Path to the chip (.tif)
            text_prompt (str): Text prompt for model
            box_threshold (float): Threshold for bounding box
            text_threshold (float): Threshold for text
            box_size_threshold (float, optional): max size of bounding box as a fraction of the image. Defaults to 0.5.

        Returns:
            tuple: Tuple containing boxes, logits, and phrases.
        """

        # Load the georeferenced image
        with rasterio.open(chip_path) as src:
            chip_np = src.read().transpose(
                (1, 2, 0)
            )  # Convert rasterio image to numpy array
            transform = src.transform
            crs = src.crs
            chip_pil = Image.fromarray(
                chip_np[:, :, :3]
            )  # Convert numpy array to PIL image, excluding the alpha channel

        # predict
        boxes, logits, phrases = self.model.predict_dino(
            chip_pil, text_prompt, box_threshold, text_threshold
        )

        # filter out large bounding boxes
        w, h = chip_pil.size
        boxes = boxes.tolist()
        logits = logits.tolist()

        indices_to_remove = []
        for i in reversed(range(len(boxes))):
            box = boxes[i]
            # 1. Filter bboxes that cover the entire image
            if box[0] < 0.05 * w and box[1] < 0.05 * h and box[2] > 0.95 * w and box[3] > 0.95 * h:
                indices_to_remove.append(i)

            # 2. also remove very large boxes that cover more than x% of the image
            elif (box[2] - box[0]) * (box[3] - box[1]) > box_size_threshold * w * h:
                indices_to_remove.append(i)

        # Remove the items at the indices
        for index in indices_to_remove:
            # print(f"Removing box: {boxes[index]}")
            del boxes[index]
            del logits[index]
            del phrases[index]

        # convert back to tensor
        boxes = torch.tensor(boxes)
        logits = torch.tensor(logits)

        # save attributes to self
        self.chip_np = chip_np

        # save attributes to self.model
        self.model.source = chip_path
        self.model.image = chip_pil
        self.model.transform = transform
        self.model.crs = crs
        self.model.boxes = boxes
        self.model.logits = logits
        self.model.phrases = phrases

        return boxes, logits, phrases


    def predict_sam(
            self,
            dtype=np.uint8,
            mask_multiplier=1,
            output=None
        ):
        """
        Predict on chip using SAM model

        Args:
            dtype (numpy.dtype, optional): Data type for the mask overlay. Defaults to np.uint8.
            mask_multiplier (int, optional): Multiplier for the mask overlay. Defaults to 1.
            output (str, optional): Path to save the mask overlay image. Defaults to None.

        Returns:
            None
        """

        masks = torch.tensor([])
        if len(self.model.boxes) > 0:
            masks = self.model.predict_sam(self.model.image, self.model.boxes)
            masks = masks.squeeze(1)

        if self.model.boxes.nelement() == 0:  # No "object" instances found
            # print("No objects found in the image, returning empty mask.")
            mask_overlay = np.zeros_like(self.chip_np[..., 0], dtype=dtype)

        else:
            # Create an empty image to store the mask overlays
            mask_overlay = np.zeros_like(
                self.chip_np[..., 0], dtype=dtype
            )  # Adjusted for single channel

            for i, (box, mask) in enumerate(zip(self.model.boxes, masks)):
                # Convert tensor to numpy array if necessary and ensure it contains integers
                if isinstance(mask, torch.Tensor):
                    mask = (
                        mask.cpu().numpy().astype(dtype)
                    )  # If mask is on GPU, use .cpu() before .numpy()
                mask_overlay += ((mask > 0) * (i + 1)).astype(
                    dtype
                )  # Assign a unique value for each mask

            # Normalize mask_overlay to be in [0, 255] if mask_multiplier is 255
            mask_overlay = (
                mask_overlay > 0
            ) * mask_multiplier  # Binary mask in [0, 255] if mask_multiplier is 255

        if output is not None:
            # Save mask as NumPy array as a GeoTIFF using projection from an existing GeoTIFF
            array_to_image(mask_overlay, output, self.model.source, dtype=dtype)

        # save attributes to self.model
        self.model.masks = masks
        self.model.prediction = mask_overlay


    def predict(
            self,
            chip_path,
            text_prompt,
            box_threshold,
            text_threshold,
            box_size_threshold=0.5,
            output=None,
        ):
        """
        Predict on chip using DINO and SAM models

        Args:
            chip_path (str): Path to the chip (.tif)
            text_prompt (str): Text prompt for model
            box_threshold (float): Threshold for bounding box
            text_threshold (float): Threshold for text,
            output (str, optional): Path to save the mask overlay image. Defaults to None.

        Returns:
            None
        """

        # reset self.model attributes
        self.reset_model()

        # predict with DINO
        boxes, logits, phrases = self.predict_dino(
            chip_path=chip_path,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            box_size_threshold=box_size_threshold,
        )

        # predict with SAM
        self.predict_sam(
            dtype=np.uint8,
            mask_multiplier=1,
            output=output
        )


    def reset_model(self):
        """
        Reset the model attributes
        """
        self.chip_np = None

        self.model.source = None
        self.model.image = None
        self.model.masks = None
        self.model.boxes = None
        self.model.phrases = None
        self.model.logits = None
        self.model.prediction = None

        self.model.transform = None
        self.model.crs = None


    def calculate_metrics(self):
        """
        Calculate metrics for the predicted mask
        """
        # Calculate IoU
        chip_path = self.model.source
        mask_path = self.get_mask_path(chip_path)
        with rasterio.open(mask_path) as src:
            mask_true = src.read(1)

        mask_pred = self.model.prediction

        # precision / specificity
        if mask_pred.sum() == 0 and mask_true.sum() != 0:
            precision = 0
        else:
            precision = (mask_pred & mask_true).sum() / mask_pred.sum()

        # recall / sensitivity
        recall = (mask_pred & mask_true).sum() / mask_true.sum()

        # accuracy / Rand index
        accuracy = (mask_pred == mask_true).sum() / mask_pred.size

        # IoU / JaccardIndex
        iou = (mask_pred & mask_true).sum() / (mask_pred | mask_true).sum()

        # F1 score / Dice coefficient
        f1_score = 2 * (mask_pred & mask_true).sum() / (mask_pred.sum() + mask_true.sum())

        # Return metrics as dictionary
        metrics = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'accuracy': round(accuracy, 4),
            'iou': round(iou, 4),
            'f1_score': round(f1_score, 4),
        }
        return metrics


    ######################
    ### Visualization ###
    ######################

    def show_anns(
        self,
        figsize=(12, 10),
        axis="off",
        cmap="viridis",
        alpha=0.4,
        add_boxes=True,
        add_masks=True,
        box_color="r",
        box_linewidth=1,
        title=None,
        output=None,
        blend=True,
        **kwargs,
    ):
        """Show the annotations (objects with random color) on the input image.
        Adapted from https://samgeo.gishub.org/text_sam/#samgeo.text_sam.LangSAM.show_anns

        Args:
            figsize (tuple, optional): The figure size. Defaults to (12, 10).
            axis (str, optional): Whether to show the axis. Defaults to "off".
            cmap (str, optional): The colormap for the annotations. Defaults to "viridis".
            alpha (float, optional): The alpha value for the annotations. Defaults to 0.4.
            add_boxes (bool, optional): Whether to show the bounding boxes. Defaults to True.
            add_masks (bool, optional): Whether to show the masks. Defaults to True.
            box_color (str, optional): The color for the bounding boxes. Defaults to "r".
            box_linewidth (int, optional): The line width for the bounding boxes. Defaults to 1.
            title (str, optional): The title for the image. Defaults to None.
            output (str, optional): The path to the output image. Defaults to None.
            blend (bool, optional): Whether to show the input image. Defaults to True.
            kwargs (dict, optional): Additional arguments for matplotlib.pyplot.savefig().

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """

        warnings.filterwarnings("ignore")

        anns = self.model.prediction
        logits = self.model.logits
        phrases = self.model.phrases

        if anns is None:
            print("Please run predict() first.")
            return
        elif len(anns) == 0:
            print("No objects found in the image.")
            return

        fig = plt.figure(figsize=figsize)
        plt.imshow(self.model.image)

        if add_boxes:
            for i, box in enumerate(self.model.boxes):
                # Draw bounding box
                box = box.cpu().numpy()  # Convert the tensor to a numpy array
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=box_linewidth,
                    edgecolor=box_color,
                    facecolor="none",
                )
                plt.gca().add_patch(rect)

                # Add phrase and logits in top left corner
                plt.text(box[0] + 5, box[1] - 10, f"{phrases[i]} {logits[i]:.2f}", 
                         color='white', backgroundcolor=box_color, fontsize=8)

        if "dpi" not in kwargs:
            kwargs["dpi"] = 100

        if "bbox_inches" not in kwargs:
            kwargs["bbox_inches"] = "tight"

        if add_masks:
            plt.imshow(anns, cmap=cmap, alpha=alpha)

        if title is not None:
            plt.title(title)
        plt.axis(axis)

        if output is not None:
            if blend:
                plt.savefig(output, **kwargs)
            else:
                array_to_image(self.model.prediction, output, self.model.source)

        return fig

    def show_true_mask(
        self,
        figsize=(12, 10),
        axis="off",
        title=None,
        output=None,
        cmap="Blues",
        alpha=1.0,
        blend=True,
        **kwargs
    ):
        """Show the true mask on the input image.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (12, 10).
            axis (str, optional): Whether to show the axis. Defaults to "off".
            title (str, optional): The title for the image. Defaults to None.
            output (str, optional): The path to the output image. Defaults to None.
            cmap (str, optional): The colormap for the mask. Defaults to "Blues".
            alpha (float, optional): The alpha value for the mask. Defaults to 1.0.
            blend (bool, optional): Whether to blend the mask with the input image. Defaults to True.
            kwargs (dict, optional): Additional arguments for matplotlib.pyplot.savefig().

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """

        chip_path = self.model.source

        if chip_path is None:
            print("Please run predict() first.")
            return

        # get mask path
        mask_path = str(Path(self.mask_dir) / f"{Path(chip_path).stem}.tif")

        # replace _img with _mask
        mask_path = mask_path.replace("_img", "_mask")

        # load mask with rasterio
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # plot mask
        fig = plt.figure(figsize=figsize)
        if blend:
            plt.imshow(self.model.image)
            plt.imshow(mask, cmap=cmap, alpha=alpha)
        else:
            plt.imshow(mask, cmap=cmap, alpha=alpha)
        plt.axis(axis)

        if title is not None:
            plt.title(title)

        if output is not None:
            plt.savefig(output, **kwargs)

        return fig

    def show_pred_vs_true_mask(
        self,
        figsize=(12, 10),
        axis="off",
        title="",
        output=None,
        alpha=0.4,
        blend=True,
        **kwargs
    ):
        """Show the predicted mask and the true mask on the input image.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (12, 10).
            axis (str, optional): Whether to show the axis. Defaults to "off".
            title (str, optional): The title for the image. Defaults to None.
            output (str, optional): The path to save the image. Defaults to None.
            alpha (float, optional): The alpha value for the mask. Defaults to 0.4.
            blend (bool, optional): Whether to blend the mask with the input image. Defaults to True.
            kwargs (dict, optional): Additional arguments for matplotlib.pyplot.savefig().

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """

        chip_path = self.model.source

        # get mask path
        mask_path = self.get_mask_path(chip_path)

        # load mask with rasterio
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # From here, use combined function 

        # Custom single-color colormap
        blue_cmap = ListedColormap(['#FF000000', 'blue'])
        yellow_cmap = ListedColormap(['#FF000000', 'yellow'])
        green_cmap = ListedColormap(['#FF000000', 'green'])
        red_cmap = ListedColormap(['#FF000000', 'red'])

        # Calculate different areas
        true_positive = np.logical_and(mask == 1, self.model.prediction == 1)
        true_negative = np.logical_and(mask == 0, self.model.prediction == 0)
        false_positive = np.logical_and(mask == 0, self.model.prediction == 1)
        false_negative = np.logical_and(mask == 1, self.model.prediction == 0)

        # Calculate metrics
        metrics = self.calculate_metrics()
        iou = metrics['iou']
        f1_score = metrics['f1_score']

        # plot mask
        fig, ax = plt.subplots(figsize=figsize)
        if blend:
            ax.imshow(self.model.image, cmap='gray')
            ax.imshow(true_positive, cmap=yellow_cmap, alpha=alpha)
            ax.imshow(true_negative, cmap=blue_cmap, alpha=alpha)
            ax.imshow(false_positive, cmap=red_cmap, alpha=alpha)
            ax.imshow(false_negative, cmap=green_cmap, alpha=alpha)
        else:
            ax.imshow(true_positive, cmap=yellow_cmap, alpha=alpha)
            ax.imshow(true_negative, cmap=blue_cmap, alpha=alpha)
            ax.imshow(false_positive, cmap=red_cmap, alpha=alpha)
            ax.imshow(false_negative, cmap=green_cmap, alpha=alpha)
        ax.axis(axis)

        # Add legend
        legend_elements = [
            patches.Patch(facecolor='yellow', alpha=alpha, label='True Positive'),
            patches.Patch(facecolor='blue', alpha=alpha, label='True Negative'),
            patches.Patch(facecolor='red', alpha=alpha, label='False Positive'),
            patches.Patch(facecolor='green', alpha=alpha, label='False Negative')
        ]
        ax.legend(handles=legend_elements)

        # Add metrics to title
        if title is not None:
            title += f" -- IoU: {iou:.2f}, F1 Score: {f1_score:.2f}"
            ax.set_title(title)

        if output is not None:
            fig.savefig(output, **kwargs)

        return fig
    
    def get_image_labels_preds(self):
        """
        Get the image, labels, and predictions
        """
        chip_path = self.model.source

        # get mask path
        mask_path = self.get_mask_path(chip_path)

        # load mask with rasterio
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        return self.model.image, mask, self.model.prediction
