# Automatic Mine Segmentation

This project aims to automate the semantic segmentation of mining areas in satellite images. It explores the use of Visual Foundation Models (VFM) for object detection and segmentation, and compares their performance with custom trained models.

## Installation

### Environment Management

#### Docker

TODO

#### Conda

The project uses external models, which means you need to set up different environments. You can use Conda or Mamba to manage the environments. There are three YAML files available for different environments:

- ``environment.yml`` (base)
- ``environment-clay.yml`` (for using the [Clay model](https://github.com/Clay-foundation/model))
- ``environment-samgeo.yml`` (for using the [segment-geospatial model](https://github.com/opengeos/segment-geospatial))

 To install the right environment, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the project's root directory.
3. Run the following command to install the environment:

```bash
conda env create --name mineseg-base --file environment.yml
```

or to update an existing conda environment:

```bash
conda env update --name mineseg-base --file environments/environment.yml --prune
```

Make sure to replace `mineseg-base` with the desired name for your environment.

1. Additionally, you have to install `unrar` using `apt-get`: 

```bash
sudo apt-get install unrar
```

### Initializing Submodules

Currently, the project includes the [Clay model repository](https://github.com/Clay-foundation/model) as a submodule. As the mine-segmentation project relies on the Clay foundation model as one of the models used, having access to the original source code is beneficial during development.

After cloning the repository, the `clay` folder will be empty, and if you want to have access to the submodule code, you need to initialize and update the submodule. To do this, run the following commands in the root directory of the project:

```bash
git submodule init
git submodule update
```

These commands will fetch and update the contents of the `clay` submodule directory based on the commit specified in the main project.

### Lightning Studios

To set up the repo in a Lightning Studio, do this before proceeding with the installation as detailed above:

1. Change the Python version to ``3.12.2``. The default version of ``3.10.10`` (by the time of testing this) caused problems with an import of sqlite3. Additionally, ``3.12.2`` is also the version used in the dev Docker image. Changing Python version can be done on the top right by clicking on "4 CPU".
2. Clone the repo into `this_studio/workspaces`. This ensures that most hardcoded paths are compatible with how the paths are inside the Docker devcontainer. 

```bash
git clone https://github.com/SimonJasansky/mine-segmentation.git workspaces/mine-segmentation
```

3. Go to project root directory:

```bash
cd workspaces/mine-segmentation
```

4. Install one of the environments. Here, it is important that in the **command the `--name cloudspace` tag is added**, as Lightning studios only allows one environment (named cloudspace by default). If the `--name cloudspace` flag is not correctly added, conda will try to create a new environment, and Lightning Studios will break. 
Also, it is **important that the environment.yml file has `name: cloudspace` as the first property**. If not, conda again will try to create a new environment, and Lightning Studios will break. 

```bash
conda env update --name cloudspace --file environments/environment.yml --prune
```

5. Add the following to the `on_start.sh` file:

```bash
cd workspaces/mine-segmentation
code -r .
```

## Using the code

### Make the dataset

To download the extenal datasets, generate global square tiles containing mining areas:

```bash
python src/data/make_dataset.py 
```

### Running the streamlit app for producing the source dataset

Run the following from the `mine-segmentation` directory:

```bash
streamlit run streamlit_app/app.py
```

### Postprocess the manually validated dataset

Add bounding boxes to the polygons:

```bash
python src/data/postprocess_dataset.py
```

## Other Info

### Setting the PYTHONPATH

To ensure relative imports work as expected, we can set the pythonpath manually. That's a bit of a hacky solution, but works for now.

```bash
export PYTHONPATH="${PYTHONPATH}:/workspaces/mine-segmentation"
```

## Acknowledgements

- Clay
- SegFormer Paper
- Samgeo

# Project Organization

> 🚧 Project Organization might not be up to date.
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
