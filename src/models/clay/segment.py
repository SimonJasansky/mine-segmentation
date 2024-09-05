"""
Command line interface to run the neural network model!

From the project root directory, do:

    python src/models/clay/segment.py fit --config configs/clay/clay_segment_config.yaml

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""
import argparse 

from lightning.pytorch.cli import LightningCLI

from src.models.datamodule import MineDataModule 
from src.models.clay.segment.model import MineSegmentor


# %%
def cli_main():
    """
    Command-line inteface to run ClayMAE with ClayDataModule.
    """
    parser = argparse.ArgumentParser(description="Train the neural network model.")
    parser.add_argument('--data_augmentation', action='store_true', help="Apply data augmentation.")
    args = parser.parse_args()

    cli = LightningCLI(
        MineSegmentor,
        MineDataModule,
        save_config_kwargs={"overwrite": True},
        run=False
    )

    # Update datamodule with flip argument
    cli.datamodule.data_augmentation = args.data_augmentation
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    return cli


# %%
if __name__ == "__main__":
    cli_main()

    print("Done!")
