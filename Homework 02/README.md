# Data Science HW2

## Model Compression

Several ways to compress the model:
- Knowledge Distillation
- Pruning
- Model Architecture Design

### Problem Description
- Dataset: Fashion-MNIST
- Input: Well-trained ResNet-50 model
- Output: compressed model
- Constraints:
    - number of parameters <= `100000`
    - accuracy >= baseline model
    - Do not use any test data, external data

### Baseline Model
- Dataset: Fashion-MNIST
- Accuracy: `92.4%`
- Total Parameters: `25,528,522`

### Grade Policy
- Kaggle Competition (75％)
    - Constraints: number of parameters <= `100000`
    - Accuracy >= baseline model (45％)
    - Private Leaderboard ranking (30％)
- Report (20％)
    - torchsummary output (5％)
    - Brief Explanation of Compression Methods (15%)
- Demo (5％)

### Model

- Optimizer: Adam
- alpha: `0.5`

Must add `with torch.no_grad():` to avoid gradient calculation.
It can save a lot of memory and speed up the training process.

```
with torch.no_grad():
    teacher_outputs = teacher_model(inputs)
```

KL Divergence Loss for Knowledge Distillation

```
student_loss = criterion(student_outputs, labels)
distillation_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1), \ 
                    F.softmax(teacher_outputs/T, dim=1)) * (T*T * 2.0 * alpha)
loss_1 = student_loss + distillation_loss
```

Also consider with ground truth with student model's loss

```
loss_1 = student_loss + distillation_loss
loss_2 = criterion(student_outputs, labels)
loss = (loss_1 + loss_2) / 2
```

#### Simple model 1

```
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.conv = nn.Conv2d(3, 48, 3, padding=1)
        self.bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(48 * 14 * 14, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 48 * 14 * 14)
        x = self.dropout(x)
        x = self.fc(x)
        return x

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 48, 28, 28]          1,344
├─BatchNorm2d: 1-2                       [-1, 48, 28, 28]          96
├─ReLU: 1-3                              [-1, 48, 28, 28]          --
├─MaxPool2d: 1-4                         [-1, 48, 14, 14]          --
├─Dropout: 1-5                           [-1, 9408]                --
├─Linear: 1-6                            [-1, 10]                  94,090
==========================================================================================
Total params: 95,530
Trainable params: 95,530
Non-trainable params: 0
Total mult-adds (M): 1.11
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.57
Params size (MB): 0.36
Estimated Total Size (MB): 0.95
==========================================================================================
```

#### Simple model 1 result

**teacher_student model accuracy: TSMA**

- Loss Function: Cross Entropy Loss

```
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
```

| Index | TSMA | lr | epoch | batch_size | T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `0.9096` | `0.001` | `30` | `64` | `20` |
| 2 | `0.9102` | `0.0005` | `30` | `64` | `20` |
| 3 | `0.9106` | `0.00075` | `50` | `64` | `20` |
| 4 | `0.9130` | `0.00075` | `100` | `64` | `40` |

- Loss Function: MSELoss

```
criterion = nn.MSELoss()
loss = criterion(outputs, F.one_hot(labels, 10).float())
```

| Index  | SMA | TSMA | lr | epoch | batch_size | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `0.8890` | `0.8871` | `0.001` | `100` | `64` | `40` |


#### Simple model 2

```
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
       # 3 layers of convolutions, less than 100k parameters
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 48)
        self.fc2 = nn.Linear(48, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 32, 28, 28]          896
├─BatchNorm2d: 1-2                       [-1, 32, 28, 28]          64
├─Conv2d: 1-3                            [-1, 32, 28, 28]          9,248
├─BatchNorm2d: 1-4                       [-1, 32, 28, 28]          64
├─MaxPool2d: 1-5                         [-1, 32, 14, 14]          --
├─Conv2d: 1-6                            [-1, 32, 14, 14]          9,248
├─BatchNorm2d: 1-7                       [-1, 32, 14, 14]          64
├─MaxPool2d: 1-8                         [-1, 32, 7, 7]            --
├─Dropout: 1-9                           [-1, 1568]                --
├─Linear: 1-10                           [-1, 48]                  75,312
├─Dropout: 1-11                          [-1, 48]                  --
├─Linear: 1-12                           [-1, 10]                  490
==========================================================================================
Total params: 95,386
Trainable params: 95,386
Non-trainable params: 0
Total mult-adds (M): 9.78
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.86
Params size (MB): 0.36
Estimated Total Size (MB): 1.23
==========================================================================================
```

#### Simple model 2 result

**teacher_student model accuracy: TSMA**

- Loss Function: Cross Entropy Loss

```
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
```

| Index | TSMA | lr | epoch | batch_size | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `0.9311` | `0.00075` | `100` | `64` | `40` |

### Simple model 3

Same as simple model 2, but no dropout for the last layer

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 32, 28, 28]          896
├─BatchNorm2d: 1-2                       [-1, 32, 28, 28]          64
├─Conv2d: 1-3                            [-1, 32, 28, 28]          9,248
├─BatchNorm2d: 1-4                       [-1, 32, 28, 28]          64
├─MaxPool2d: 1-5                         [-1, 32, 14, 14]          --
├─Conv2d: 1-6                            [-1, 32, 14, 14]          9,248
├─BatchNorm2d: 1-7                       [-1, 32, 14, 14]          64
├─MaxPool2d: 1-8                         [-1, 32, 7, 7]            --
├─Dropout: 1-9                           [-1, 1568]                --
├─Linear: 1-10                           [-1, 48]                  75,312
├─Linear: 1-11                           [-1, 10]                  490
==========================================================================================
Total params: 95,386
Trainable params: 95,386
Non-trainable params: 0
Total mult-adds (M): 9.78
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.86
Params size (MB): 0.36
Estimated Total Size (MB): 1.23
==========================================================================================
```

#### Simple model 3 result

**teacher_student model accuracy: TSMA**

- Loss Function: Cross Entropy Loss

```
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
```

| Index   | TSMA | lr | epoch | batch_size | T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `0.9311` | `0.0005` | `100` | `64` | `40` |
| 2 | `0.9286` | `0.0005` | `100` | `128` | `40` |
| 3 | `0.9313` | `0.001` | `100` | `128` | `40` |
| 4 | `0.9302` | `0.005` | `100` | `128` | `40` |
| 5 | `0.9322` | `0.001` | `100` | `64` | `40` |
| 6 | `0.9307` | `0.001` | `100` | `64` | `4` |
| 7 | `0.9330` | `0.001` | `100` | `64` | `40` |
| 8 | `0.9319` | `0.0015` | `100` | `64` | `40` |

### Demo Platform
- OS: Ubuntu 20.04
- CPU: AMD Ryzen Threadripper (will set num_worker=8)
- GPU: RTX 3080 (8GB) *1
- Python 3.8.10
- CUDA: 11.07
- Framework: PyTorch 1.13.1