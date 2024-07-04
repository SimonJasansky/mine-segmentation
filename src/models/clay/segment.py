"""
Command line interface to run the neural network model!

From the project root directory, do:

    python src/models/clay/segment.py fit --config configs/clay_segment_config.yaml

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""

from lightning.pytorch.cli import LightningCLI

from segment.datamodule import MineDataModule 
from segment.model import MineSegmentor


# %%
def cli_main():
    """
    Command-line inteface to run ClayMAE with ClayDataModule.
    """
    cli = LightningCLI(MineSegmentor, MineDataModule)
    return cli


# %%
if __name__ == "__main__":
    cli_main()

    print("Done!")
