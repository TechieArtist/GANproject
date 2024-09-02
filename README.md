# MNIST GAN Project 
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## About <a name = "about"></a>
**[Current Issues](https://github.com/yourusername/mnist-gan/issues)**

# MNIST GAN Project
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## About <a name="about"></a>
This project implements a Generative Adversarial Network (GAN) to generate images resembling handwritten digits from the MNIST dataset.

### Libraries Overview <a name="lib_overview"></a>

All the libraries and code for this project are located under the `./libs` directory:
- `./libs/gan`: Contains the GAN model architecture and training scripts.
- `./libs/preprocessing`: Functions and classes for preprocessing the MNIST dataset.
- `./libs/logger`: Custom logger for text formatting and debugging output.

### Where to Put the Code <a name="putcode"></a>
- Place the preprocessing functions/classes in `./libs/preprocessing.py`.
- The GAN model architecture and training functions/classes should go in `./libs/gan.py`.
- The custom logger and any utility functions should be placed in `./libs/logger.py`.

**The code is reloaded automatically. Any class object needs to be reinitialized though.** 

## Table of Contents

+ [About](#about)
  + [Libraries Overview](#lib_overview)
  + [Where to Put the Code](#putcode)
+ [Prerequisites](#prerequisites)
+ [Bootstrap Project](#bootstrap)
+ [Running the Code](#running_code)
  + [Configuration](#configuration)
  + [Local Jupyter](#local_jupyter)
  + [Google Colab](#google_colab)
+ [Adding New Libraries](#adding_libs)
+ [TODO](#todo)
+ [License](#license)

## Prerequisites <a name="prerequisites"></a>

You need to have a machine with Python >= 3.9 and any Bash-based shell (e.g., zsh) installed. Installing `conda` is also recommended.

```shell
$ python3.9 -V
Python 3.9.7

$ echo $SHELL
/usr/bin/zsh
