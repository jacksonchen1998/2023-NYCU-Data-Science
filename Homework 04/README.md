# Crowd Estimation

[Kaggle Competition](https://www.kaggle.com/competitions/data-science-2023-hw4-crowd-counting/overview)

In this homework, you need to implement crowd counting algorithms to solve the given tasks.

## Dataset

- Training data: `4135` images in `train` folder
- Training labels: `4135` text files containing the locations of heads
- Testing data: `1772` images in `test` folder

Since the provided dataset's txt file has wrong sot x/y coordinates, the `data_preprocess.py` script is provided to fix the coordinates. We can get `npy` files for training data to get the right coordinates.

## Problem

It needs to use MAE as the evaluation metric.
