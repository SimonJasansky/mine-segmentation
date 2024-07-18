# MineSegmentorCNN

The `MineSegmentorCNN` class is designed for semantic segmentation tasks, using different architectures and backbones available from [Segmentation Models Pytorch](https://smp.readthedocs.io)

## Parameters

- `arch (str)`
- `encoder_name (str)`
- `encoder_weights (str)`
- `in_channels (int)`
- `classes (int)`

See the docs here: https://smp.readthedocs.io/en/latest/quickstart.html 

## Example

Here, the `MineSegmentorCNN` class is used to segment the custom created dataset of mining areas. The implementation includes data preprocessing, data loading, and model training workflow using PyTorch Lightning.

## Dataset

## Setup

Follow the instructions in the [README](../../README.md) to install the required dependencies. Use the `environment-clay.yml` file to set up the environment.

## Usage

### Preparing the Dataset
Check the configurations in the file `make_dataset_post.py`, and create the dataset with: 

```bash
python src/data/make_dataset_post.py
```

Or use custom configurations as outlined in the top-level README

Directory structure:

```
data/processed/
└── images/
    ├── files/
    │   ├── train/
    │   └── val/
    └── chips/
        ├── train/
        │   ├── chips/
        │   └── labels/
        └── val/
            ├── chips/
            └── labels/
```

### Training the Model

The model can be run via LightningCLI using configurations in `configs/cnn_segment_config_gpu.yaml` or `configs/cnn/cnn_segment_config_gpu.yaml`.

1. Modify the batch size, learning rate, and other hyperparameters in the configuration file as needed:
    ```yaml
    data:
        batch_size: 1
        num_workers: 4 # Set to number of CPU cores
        platform: sentinel-2-l2a
    model:
        arch: Unet
        encoder_name: resnet34
        encoder_weights: imagenet
        num_classes: 2
        in_channels: 3
    ```

2. Update the [WandB logger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html#lightning.pytorch.loggers.WandbLogger) configuration in the configuration file with your WandB details or use [CSV Logger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html#lightning.pytorch.loggers.CSVLogger) if you don't want to log to WandB:
    ```yaml
    logger:
      - class_path: lightning.pytorch.loggers.WandbLogger
        init_args:
          entity: <wandb-entity>
          project: <wandb-project>
          log_model: false
    ```

3. Train the model:
    ```bash
    # on CPU
    python src/models/cnn/train.py fit --config configs/cnn/cnn_segment_config_cpu.yaml

    # on GPU
    python src/models/cnn/train.py fit --config configs/cnn/cnn_segment_config_gpu_pc.yaml
    python src/models/cnn/train.py fit --config configs/cnn/cnn_segment_config_gpu_T4.yaml
    python src/models/cnn/train.py fit --config configs/cnn/cnn_segment_config_gpu_L4.yaml
    ```

## Acknowledgments

Models are adapted from [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)

Code is inspired by the implementation of finetuning the Clay model: https://github.com/Clay-foundation/model/tree/main/finetune/segment. Credits to the authors of the Clay Foundation Model.
