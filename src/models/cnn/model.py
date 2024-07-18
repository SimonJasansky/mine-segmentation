"""
LightningModule for training and validating a CNN segmentation model
"""

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

class MineSegmentorCNN(L.LightningModule):
    """
    LightningModule for segmentation tasks

    Attributes:
        model (nn.Module): The model for segmentation.
        loss_fn (nn.Module): The loss function.
        iou (Metric): Intersection over Union metric.
        f1 (Metric): F1 Score metric.
        lr (float): Learning rate.
    """

    def __init__( 
        self,
        arch,
        encoder_name,
        encoder_weights,
        in_channels,
        num_classes,
        lr,
        wd,
        b1,
        b2,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels, 
            classes=num_classes,
            **kwargs
        )

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.iou = BinaryJaccardIndex()
        self.f1 = BinaryF1Score()

    def forward(self, image):
        """
        Forward pass through the segmentation model.

        Args:
            image (tensor): A tensor containing the image data.

        Returns:
            torch.Tensor: The segmentation logits.
        """

        # Forward pass through the network
        return self.model(image)


    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler
            configuration.
        """
        # optimizer = optim.AdamW(
        #     [
        #         param
        #         for name, param in self.model.named_parameters()
        #         if param.requires_grad
        #     ],
        #     lr=self.hparams.lr,
        #     weight_decay=self.hparams.wd,
        #     betas=(self.hparams.b1, self.hparams.b2),
        # )
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=1000,
        #     T_mult=1,
        #     eta_min=self.hparams.lr * 100,
        #     last_epoch=-1,
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #     },
        # }
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def shared_step(self, batch, phase):
        """
        Shared step for training and validation.

        Args:
            batch (dict): A dictionary containing the batch data.
            phase (str): The phase (train or val).

        Returns:
            torch.Tensor: The loss value.
        """
        labels = batch["label"]
        image = batch["pixels"]

        # change image range from [-1, 1] to [0, 1]
        # image = (image + 1) / 2

        # switch the first and second dimensions
        # image = image.permute(1, 0, 2, 3)

        logits_mask = self.forward(image)

        # expand the first dimension of the labels
        labels = labels.unsqueeze(1)

        print(image.shape, logits_mask.shape, labels.shape)
        print("Image")
        print(image)
        # check max and min values
        print(image.max(), image.min())
        # check if there is any nan in the image
        print(torch.isnan(image).any())

        print("Logits")
        print(logits_mask)

        # print(labels)

        loss = self.loss_fn(logits_mask, labels)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        
        iou = self.iou(pred_mask, labels)
        f1 = self.f1(pred_mask, labels)

        # Log metrics
        self.log(
            f"{phase}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/iou",
            iou,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/f1",
            f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch):
        """
        Training step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, "train")

    def validation_step(self, batch):
        """
        Validation step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, "val")
