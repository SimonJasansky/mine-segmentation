"""
Command line interface to train the neural network model!

From the project root directory, do:

    python src/models/cnn/segment.py fit --config configs/cnn/cnn_segment_config_cpu.yaml

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""

import torch
# import yaml

# Set matmul precision
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')  # or 'high'

from lightning.pytorch.cli import LightningCLI

from src.models.datamodule import MineDataModule
from src.models.cnn.model import MineSegmentorCNN

# %%
def cli_main():
    """
    Command-line inteface to run the CNN with MineDataModule.
    """
    cli = LightningCLI(MineSegmentorCNN, MineDataModule, save_config_kwargs={"overwrite": True})

    # Print the configuration
    # print(yaml.dump(cli.config))

    return cli


# %%
if __name__ == "__main__":
    cli_main()

    print("Done!")
