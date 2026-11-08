# Animal Image Classification Using Convolutional Neural Networks

See the [project report](report.pdf) for details about the methodology, experiments, and results.

## Overview

An animal image classification project built with Python, PyTorch, and convolutional neural networks.

Completed in November 2023.

## Dataset

This project uses the [Animals-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10), which contains images from 10 animal categories:

Dogs, cats, horses, spiders, butterflies, chickens, sheep, cows, squirrels, and elephants.

## Approach

The project explores transfer learning using pretrained EfficientNet and ResNet models, with image preprocessing, training, and validation implemented in PyTorch.

## Results

The final model achieved an accuracy of 71%.

## Repository

- `report.pdf`: project report.
- `model.py`: model definitions.
- `train.py` and `trainefficient.py`: training scripts.
- `run.py`: inference script.
- `utils/`: preprocessing and data-loading utilities.
- `checkpoints/`: saved model weights.

## Technologies

Python · PyTorch · Torchvision · EfficientNet-PyTorch · NumPy · Scikit-learn · Matplotlib