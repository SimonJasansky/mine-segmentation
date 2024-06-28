# Use an official GDAL image as the base image
FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

# install the necessary packages
RUN apt-get update && apt-get -y install \
    git \
    libjpeg-dev zlib1g-dev \
    curl tar \
    unrar \
    python3-pip --fix-missing

# # Install Miniconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
#     bash /miniconda.sh -b -p /miniconda && \
#     rm /miniconda.sh
# ENV PATH="/miniconda/bin:${PATH}"
# RUN conda init bash
# RUN conda install -c conda-forge mamba

# # Create a conda environment from the environment.yml file
# COPY environments/environment.yml .
# RUN mamba env create -n mineseg-base -f environment.yml

# # Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "mineseg-base", "/bin/bash", "-c"]

# # Add the conda environment to Jupyter
# RUN python -m ipykernel install --user --name=mineseg-base

##############
# Using VENV #
##############

RUN apt-get install -y python3-venv

# Set an environment variable with the directory
# where we'll be running our venv
ENV VIRTUAL_ENV=/opt/venv

# Create a virtual environment and activate it
RUN python3 -m venv $VIRTUAL_ENV --system-site-packages
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install the necessary dependencies
COPY environments/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Add the virtual environment to Jupyter
RUN python -m ipykernel install --user --name=venv

# Add the root directory to the PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/workspaces/mine-segmentation"