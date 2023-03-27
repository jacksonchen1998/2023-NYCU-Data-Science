# Few-Shot Learning

[Kaggle Competition](https://www.kaggle.com/competitions/data-science-2023-hw3-few-shot-learning/data)

In this homework, you need to implement few-shot learning algorithms to solve the given tasks.

In order to successfully complete this homework, you need to pay close attention to two key points:
- Data formats of few-shot learning
- Basic algorithms for few-shot learning

## Dataset

* `train.pkl` - a dictionary with keys images and labels.
    - `images` - a numpy array in shape (38400, 3, 84, 84)
    - `labels` - a numpy array in shape (38400,)
    There are 64 unique labels (0~63).
    
* `validation.pkl` - a dictionary with keys images and labels.
    - `images` - a numpy array in shape (9600, 3, 84, 84)
    - `labels` - a numpy array in shape (9600,)
    There are 16 unique labels (0~15).

* `test.pkl` - a dictionary with keys sup_images, sup_labels, and qry_images.
    - `sup_images` - a numpy array in shape (600, 25, 3, 84, 84)
    - `sup_labels`  - a numpy array in shape (600, 25)
    - `qry_images` - a numpy array in shape (600, 25, 3, 84, 84)
    There are 600 5-way-5-shot tasks.

For each task, there are 5 unique labels (0~4). Note that these labels are defined locally for each task, which are different from the `train.pkl` and `validation.pkl`.