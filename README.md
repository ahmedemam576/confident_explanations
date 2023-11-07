# confident_explanations
# Image Segmentation with UNet and Monte Carlo Dropout

## Introduction

This repository contains code for performing image segmentation using a UNet architecture with Monte Carlo Dropout for uncertainty quantification. The code is designed for a specific task and includes data preprocessing, model creation, training settings, and uncertainty assessment.

## Features

- UNet architecture for image segmentation.
- Monte Carlo Dropout for uncertainty quantification.
- Data preprocessing and transformation pipelines.
- Training settings and hyperparameters.
- Evaluation of uncertainty at the class level.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- pandas
- numpy
- OpenCV (cv2)
- tifffile
- tqdm

## Setup

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
## Usage

    Prepare your dataset and update the paths in the code to point to your data and labels.

    Set your desired hyperparameters and training settings in the code.

    Train the UNet model using the provided functions.

    Use Monte Carlo Dropout to assess uncertainty by running the mc_dropout_all_batches function.

    View uncertainty at the class level using the class_std function.

    Adjust and fine-tune the code for your specific project needs.
