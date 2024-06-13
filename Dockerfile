# Use an official GDAL image as the base image
FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

# install the necessary packages
RUN apt-get update && apt-get -y install \
    git \
    libjpeg-dev zlib1g-dev \
    unrar \
    python3-pip --fix-missing \
    python3-venv

# Set an environment variable with the directory
# where we'll be running our venv
ENV VIRTUAL_ENV=/opt/venv

# Create a virtual environment and activate it
RUN python3 -m venv $VIRTUAL_ENV --system-site-packages
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install the necessary dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Add the virtual environment to Jupyter
RUN python -m ipykernel install --user --name=venv

# Add the root directory to the PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/workspaces/mine-segmentation"