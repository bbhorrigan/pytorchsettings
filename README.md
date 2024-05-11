
# PyTorch Settings

Welcome to my PyTorch Settings repository! Here, you'll find configurations tailored for my personal use in various machine learning projects. These settings are designed to streamline my workflow and ensure consistency across different environments.  When I started using Pytorch, I realized that there are settings that I might have to move back and forth from different devices, beacuse of testing on different CPU sets. This is the utilities I wrote that allow me to move back and forth from machine to machine and copying my settings. I am going to build a utility to drop this into something like OneDrive or FTP. You will see some of that logic already built.

## Overview

This repository houses various configuration files and scripts that I utilize to set up PyTorch environments for my projects. These configurations encompass aspects such as device handling, seed settings for reproducibility, and performance optimizations.

## Contents

- **environment.yml**: Contains a Conda environment file for easily setting up PyTorch with all necessary dependencies.
- **settings.py**: Custom Python script to configure PyTorch settings. It covers device management, reproducibility through seed settings, and performance optimization.
- **utils/**: This directory holds utility scripts that enhance PyTorch functionality, making tasks like data loading and model training more efficient.

## Usage

To incorporate these settings into your projects, follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/bbhorrigan/pytorchsettings.git
    cd pytorchsettings
    ```

2. Set up the Conda environment using the provided environment file:

    ```bash
    conda env create -f environment.yml
    ```

3. Activate the created environment. Replace `<environment-name>` with the name you choose for the Conda environment:

    ```bash
    conda activate <environment-name>
    ```

   For example, if you named your environment "pytorch_env", the command would be:

    ```bash
    conda activate pytorch_env
    ```

