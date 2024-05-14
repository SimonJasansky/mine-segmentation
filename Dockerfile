# Use an official GDAL image as the base image
FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

# install pip
RUN apt-get update && apt-get -y install \
    # git \
    libjpeg-dev zlib1g-dev \
    python3-pip --fix-missing 

# Set the working directory in the container
WORKDIR /src

# Install the necessary dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --break-system-packages