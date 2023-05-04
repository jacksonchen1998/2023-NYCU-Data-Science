# Crowd Estimation

[Kaggle Competition](https://www.kaggle.com/competitions/data-science-2023-hw4-crowd-counting/overview)

In this homework, you need to implement crowd counting algorithms to solve the given tasks.

## Dataset

- Training data: `4135` images in `train` folder
- Training labels: `4135` text files containing the locations of heads
- Testing data: `1772` images in `test` folder

Since the provided dataset's txt file has wrong sot x/y coordinates, the `data_process.py` script is provided to fix the coordinates.

We can get `npy` files for training data to get the right coordinates.

## Method

In this homework, we can implement **Bayesian Loss for Crowd Count Estimation with Point Supervision (ICCV2019)** to solve the given tasks.

The backbone of the model is `VGG19`. And the loss function is `Bayesian Loss` with `Posterior Gaussian Distribution`.

Pretrained weights are recommended to use, but not necessary. Also, the homework is limited not to use any other dataset or pretrained model.

Most of the functions are modified from the original paper's github repository.

Here're the steps to modify the code:

1. `crowd.py`: Modify the `train_transform` function to fit the dataset. Since the `keypoints` variable, we need to modify it to get the right coordinates for cropped images. Also, we need to add some lines to get the right coordinates for the original images from `preprocess_dataset.py`'s `generate_data` function.
2. Copy loss function from `bay_loss.py` and `post_prob.py` to the repository without any modification.
3. Using the code from `regression_trainer.py` to train the model. And the dataset can be built by `crowd.py`'s `Crowd` class.
4. Add the function `train_collate` from `regression_trainer.py` be added as `collate_fn` to the `DataLoader` class.

The parameters of the model are as follows:

```
lr: 2e-5
batch_size: 8
total_epoch: 1000
optimizer: Adam
weight_decay: 1e-4
scheduler: CosineAnnealingLR (T_max=total_epoch, eta_min=8e-6)
```

The arguments of `Post_Prob` class are as follows:

```
sigma: 8
crop_size: 512
downsample_ratio: 8
use_background: True
device: device
```

The arguments of `Bay_Loss` class are as follows:

```
use_background: True
device: device
```


## Reference

- [Bayesian Loss for Crowd Count Estimation with Point Supervision (ICCV2019)](https://arxiv.org/abs/1908.03684)
- [Bayesian Loss for Crowd Count Estimation with Point Supervision (Github)](https://github.com/ZhihengCV/Bayesian-Crowd-Counting)
