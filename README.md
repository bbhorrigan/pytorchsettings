# My Personal PyTorch Settings
This is my GitHub repository where I store my personal PyTorch configurations. Here, you'll find settings that I've tailored for my own use in various machine learning projects.

## Overview

This repository includes various configuration files and scripts that I use to set up PyTorch environments for my projects. These settings help streamline my workflow and ensure consistency across different environments.

## Contents

- `environment.yml`: Contains a Conda environment file for setting up PyTorch with all necessary dependencies.
- `settings.py`: Custom Python script to configure PyTorch settings like device handling, seed settings for reproducibility, and performance optimizations.
- `utils/`: This directory contains utility scripts that enhance PyTorch functionality, making tasks like data loading and model training more efficient.

## Usage

To use these settings, clone this repository and install the required dependencies using the provided environment file:

```bash
git clone https://github.com/bbhorrigan/pytorchsettings.git
cd your-repo-name
conda env create -f environment.yml
