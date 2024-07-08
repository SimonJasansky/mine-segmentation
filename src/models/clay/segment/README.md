# Segmentor

The `Segmentor` class is designed for semantic segmentation tasks, extracting feature maps from intermediate layers of the Clay Encoder and adding a Feature Pyramid Network (FPN) on top of it.

Decoder is inspired by the Segformer paper.

## Parameters

- `feature_maps (list)`: Indices of intermediate layers of the Clay Encoder used by FPN layers.
- `ckpt_path (str)`: Path to the Clay model checkpoint.

## Example

Here, the `Segmentor` class is used to segment the custom created dataset of mining areas. The implementation includes data preprocessing, data loading, and model training workflow using PyTorch Lightning.

## Dataset

## Setup

Follow the instructions in the [README](../../README.md) to install the required dependencies. Use the `environment-clay.yml` file to set up the environment.

## Usage

### Preparing the Dataset
Check the configurations in the file `make_dataset_post.py`, and create the dataset with: 

```bash
python src/data/make_dataset_post.py
```

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

The model can be run via LightningCLI using configurations in `configs/clay_segment_config_gpu.yaml` or `configs/clay_segment_config_gpu.yaml`.

1. Download the Clay model checkpoint from [Huggingface model hub](https://huggingface.co/made-with-clay/Clay/blob/main/clay-v1-base.ckpt) and save it in the `models/` directory.

```bash
python src/models/clay/download_checkpoint.py
```

2. Modify the batch size, learning rate, and other hyperparameters in the configuration file as needed:
    ```yaml
    data:
      batch_size: 40
      num_workers: 8
    model:
      num_classes: 7
      feature_maps:
        - 3
        - 5
        - 7
        - 11
      ckpt_path: checkpoints/clay-v1-base.ckpt
      lr: 1e-5
      wd: 0.05
      b1: 0.9
      b2: 0.95
    ```

3. Update the [WandB logger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html#lightning.pytorch.loggers.WandbLogger) configuration in the configuration file with your WandB details or use [CSV Logger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html#lightning.pytorch.loggers.CSVLogger) if you don't want to log to WandB:
    ```yaml
    logger:
      - class_path: lightning.pytorch.loggers.WandbLogger
        init_args:
          entity: <wandb-entity>
          project: <wandb-project>
          log_model: false
    ```

4. Train the model:
    ```bash
    # on CPU
    python src/models/clay/segment.py fit --config configs/clay_segment_config_cpu.yaml

    # on GPU
    python src/models/clay/segment.py fit --config configs/clay_segment_config_gpu.yaml
    ```

## Acknowledgments

Code and scripts for the finetuning of the clay model are modified from https://github.com/Clay-foundation/model/tree/main/finetune/segment. Credits to the authors of the Clay Foundation Model.

Decoder implementation is inspired by the Segformer paper:
```
Segformer: Simple and Efficient Design for Semantic Segmentation with Transformers
Paper URL: https://arxiv.org/abs/2105.15203
```
