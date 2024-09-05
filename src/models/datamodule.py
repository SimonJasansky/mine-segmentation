"""
DataModule for the Mining dataset for segmentation tasks.

This implementation provides a structured way to handle the data loading and
preprocessing required for training and validating a segmentation model.

"""

import re
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class MineDataset(Dataset):
    """
    Dataset class for the Mine segmentation dataset.

    Args:
        chip_dir (str): Directory containing the image chips.
        label_dir (str): Directory containing the labels.
        metadata (Box): Metadata for normalization and other dataset-specific details.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(self, chip_dir, label_dir, metadata, platform):
        self.chip_dir = Path(chip_dir)
        self.label_dir = Path(label_dir)
        self.metadata = metadata
        self.image_transform, self.flip_transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )

        # Load chip and label file names
        self.chips = [chip_path.name for chip_path in self.chip_dir.glob("*.npy")]
        self.labels = [re.sub("_img", "_mask", chip) for chip in self.chips]

    def create_transforms(self, mean, std):
        """
        Create normalization and flipping transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            tuple: A tuple containing the image transform and the flip transform.
        """
        image_transform = v2.Normalize(mean=mean, std=std)
        flip_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
        return image_transform, flip_transform

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the image, label, and additional information.
        """
        chip_name = self.chip_dir / self.chips[idx]
        label_name = self.label_dir / self.labels[idx]

        chip = np.load(chip_name).astype(np.float32)
        label = np.load(label_name)

        # Remap labels to match desired classes
        label_mapping = {0: 0, 1: 1}
        remapped_label = np.vectorize(label_mapping.get)(label)

        chip_tensor = torch.from_numpy(chip)
        label_tensor = torch.from_numpy(remapped_label[0])

        # Apply normalization only to the image
        chip_tensor = self.image_transform(chip_tensor)

        # Apply the same transformations to both image and mask
        chip_tensor, label_tensor = self.flip_transform(chip_tensor, label_tensor)

        sample = {
            "pixels": chip_tensor,
            "label": label_tensor,
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for latlon information
        }
        return sample


class MineDataModule(L.LightningDataModule):
    """
    DataModule class for the Mine dataset.

    Args:
        train_chip_dir (str): Directory containing training image chips.
        train_label_dir (str): Directory containing training labels.
        val_chip_dir (str): Directory containing validation image chips.
        val_label_dir (str): Directory containing validation labels.
        test_chip_dir (str): Directory containing test image chips.
        test_label_dir (str): Directory containing test labels.
        metadata_path (str): Path to the metadata file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        platform (str): Platform identifier used in metadata.
    """

    def __init__( 
        self,
        train_chip_dir,
        train_label_dir,
        val_chip_dir,
        val_label_dir,
        test_chip_dir,
        test_label_dir,
        metadata_path,
        batch_size,
        num_workers,
        platform,
    ):
        super().__init__()
        self.train_chip_dir = train_chip_dir
        self.train_label_dir = train_label_dir
        self.val_chip_dir = val_chip_dir
        self.val_label_dir = val_label_dir
        self.test_chip_dir = test_chip_dir
        self.test_label_dir = test_label_dir
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.platform = platform

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('fit' or 'test').
        """
        if stage in {"fit", None}:
            self.trn_ds = MineDataset(
                self.train_chip_dir,
                self.train_label_dir,
                self.metadata,
                self.platform,
            )
            self.val_ds = MineDataset(
                self.val_chip_dir,
                self.val_label_dir,
                self.metadata,
                self.platform,
            )
        elif stage == "test":
            self.test_ds = MineDataset(
                self.test_chip_dir,
                self.test_label_dir,
                self.metadata,
                self.platform,
            )

    def train_dataloader(self):
        """
        Create DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training dataset.
        """
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Create DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """
        Create DataLoader for test data.

        Returns:
            DataLoader: DataLoader for test dataset.
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )