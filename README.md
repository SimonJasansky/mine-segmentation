# Automatic Mine Segmentation

This project aims to automate the semantic segmentation of mining areas in satellite images. It explores the use of Visual Foundation Models (VFM) for object detection and segmentation, and compares their performance with custom trained models.

## Installation

### Environment Management

#### Conda

The project uses external models, which means you need to set up different environments. You can use Conda or Mamba to manage the environments. There are three YAML files available for different environments:

- ``environment.yml`` (base)
- ``environment-clay.yml`` (for using the [Clay model](https://github.com/Clay-foundation/model))
- ``environment-samgeo.yml`` (for using the [segment-geospatial model](https://github.com/opengeos/segment-geospatial))

1. Install environment:

```bash
conda env create --name mineseg-base --file environment.yml
```

or to update an existing conda environment:

```bash
conda env update --name mineseg-base --file environments/environment.yml --prune
```

Make sure to replace `mineseg-base` with the desired name for your environment.

1. Additionally, if you want to make the tiles or use the source datasets ([Maus et al.](https://doi.pangaea.de/10.1594/PANGAEA.942325) and [Tang et al.](https://zenodo.org/records/6806817)) you have to install `unrar` using `apt-get`:

```bash
sudo apt-get install unrar
```

#### Docker

TODO

### Lightning Studios

To set up the repo in a Lightning Studio, do this before proceeding with the installation as detailed above:

1. Change the Python version to ``3.11``. Changing Python version can be done on the top right by clicking on "4 CPU".
2. Clone the repoository:

```bash
git clone https://github.com/SimonJasansky/mine-segmentation.git
```

3. Go to project root directory:

```bash
cd mine-segmentation
```

4. Install one of the environments. Here, it is important that in the **command the `--name cloudspace` tag is added**, as Lightning studios only allows one environment (named cloudspace by default). If the `--name cloudspace` flag is not correctly added, conda will try to create a new environment, and Lightning Studios will break.
Also, it is **important that the environment.yml file has `name: cloudspace` as the first property**. If not, conda again will try to create a new environment, and Lightning Studios will break.

```bash
conda env update --name cloudspace --file environments/environment.yml --prune
```

5. Add the following to the `on_start.sh` file, to always open the repository directly:

```bash
cd mine-segmentation
code -r .
```

## Using the code

### Make the dataset

To download the extenal datasets, generate global square tiles containing mining areas:

```bash
python src/data/02A_make_dataset_pre.py
```

### Running the streamlit app for producing the source dataset

Run the following from the `mine-segmentation` directory:

```bash
streamlit run streamlit_app/app.py
```

### Postprocess the manually validated dataset, download images, and create chips for model training

To run all post-processing steps with preconfigured settings: 

```bash
python src/data/06A_make_dataset_post.py
```

Individual steps can be run with:

```bash
# postprocess & generate bounding boxes
python src/data/03_postprocess_dataset.py

# filter dataset to fit requirements
python src/data/04_filter_and_split_dataset.py preferred_polygons --val_ratio 0.15 --test_ratio 0.10 --only_valid_surface_mines

# download S2 images & create masks
python src/data/05_persist_pixels_masks.py data/processed/files preferred_polygons --split all

# chip images
python src/data/06_make_chips.py data/processed/files data/processed/chips/npy/512 512 npy --must_contain_mining --split all
```

## Other Info

### Setting the PYTHONPATH

To ensure relative imports work as expected, we can set the pythonpath manually. That's a bit of a hacky solution, but works for now.

```bash
export PYTHONPATH="${PYTHONPATH}:/mine-segmentation"
```

## Acknowledgements

This project relies on code and models provided by third party sources.
Credit for their amazing work goes to:

- Clay
  - Website: https://madewithclay.org/
  - Docs: https://clay-foundation.github.io/model/index.html
  - Repo: https://github.com/Clay-foundation/model
- Samgeo
  - Website & Docs: https://samgeo.gishub.org/
  - Repo: https://github.com/opengeos/segment-geospatial

## Project Organization

🚧 Project Organization might not be up to date.

------------
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump, including the manually produced dataset.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    |── configs            <- config files for training and using models
    |
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    |── environments       <- environment.yml files
    │
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        ├── features       <- Scripts to turn raw data into features for modeling
        ├── models         <- Scripts to train models and then use trained models to make predictions
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
--------
