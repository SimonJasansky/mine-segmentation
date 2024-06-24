mine-segmentation
==============================

This project aims to automate the semantic segmentation of mining areas in satellite images. It explores the use of Visual Foundation Models (VFM) for object detection and segmentation, and compares their performance with custom trained models.

# Installation

## Environments
The project uses external models, which means you need to set up different environments. You can use Conda or Mamba to manage the environments. There are three YAML files available for different environments:
- ``environment.yml`` (base)
- ``environment-clay.yml`` (for using the [Clay model](https://github.com/Clay-foundation/model))
- ``environment-samgeo.yml`` (for using the [segment-geospatial model](https://github.com/opengeos/segment-geospatial))

 To install the right environment, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the project's root directory.
3. Run the following command to install the environment:
```bash
mamba create --name myenv --file environment.yml
```
or to update an existing conda environment: 

```bash
conda env update --name myenv --file environments/environment.yml --prune
```
Make sure to replace `myenv` with the desired name for your environment.

4. Additionally, you have to install `unrar` using `apt-get`: 
```bash
sudo apt-get install unrar
```

## Initializing Submodules

After cloning the repository, you need to initialize and update the submodules, including the external model repository (Clay). To do this, run the following commands in the root directory of the project:

```bash
git submodule init
git submodule update
```

These commands will fetch and update the contents of the `clay` submodule directory based on the commit specified in the main project.

# Running the streamlit app for producing the source dataset

Run the following from the `mine-segmentation` directory:
```bash
streamlit run streamlit_app/app.py
```

# Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump, including the manually produced dataset.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    |── environments       <- environment.yml files
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
