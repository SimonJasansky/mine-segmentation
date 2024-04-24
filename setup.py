from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project aims to automate the semantic segmentation of mining areas in satellite images. It explores the use of Visual Foundation Models (VFM) for object detection and segmentation, and compares their performance with custom trained models.',
    author='Simon Jasansky',
    license='MIT',
)
