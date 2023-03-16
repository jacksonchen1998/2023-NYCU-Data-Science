# Data Science HW2

## Model Compression

[Kaggle Competition](https://www.kaggle.com/competitions/model-compression-on-fashion-mnist/overview)

Several ways to compress the model:
- Knowledge Distillation
- Pruning
- Model Architecture Design

In this homework, we use **Knowledge Distillation** to compress the model.

To run the model training code, please use the following command:

```
python3 hw2_311511052.py --batch_size 128 --num_epochs 300 \ 
--learning_rate 0.0035 --T 40 --alpha 0.5 --factor 0.2 --patience 10 --min_lr 0.0001
```

The arguments are as follows:
- `batch_size`: batch size
- `num_epochs`: number of epochs
- `learning_rate`: learning rate
- `T`: temperature for knowledge distillation
- `alpha`: alpha for knowledge distillation
- `factor`: factor for learning rate scheduler
- `patience`: patience for learning rate scheduler
- `min_lr`: minimum learning rate

And then the program will automatically store the pth file in the `pth_folder` folder. And the csv file will be stored in the `submission_folder` folder.

To test the pth file, please use the following command:

```
python3 test_model.py --model ./pth_folder/[your pth name].pth
```

Then the program will generate the summary of the model and the accuracy of the model.

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

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>TSMA</th>
      <th>lr</th>
      <th>epoch</th>
      <th>batch_size</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>1</code></td>
      <td><code>0.9096</code></td>
      <td><code>0.001</code></td>
      <td><code>30</code></td>
      <td><code>64</code></td>
      <td><code>20</code></td>
    </tr>
    <tr>
      <td><code>2</code></td>
      <td><code>0.9102</code></td>
      <td><code>0.0005</code></td>
      <td><code>30</code></td>
      <td><code>64</code></td>
      <td><code>20</code></td>
    </tr>
    <tr>
      <td><code>3</code></td>
      <td><code>0.9106</code></td>
      <td><code>0.00075</code></td>
      <td><code>50</code></td>
      <td><code>64</code></td>
      <td><code>20</code></td>
    </tr>
    <tr>
      <td><code>4</code></td>
      <td><code>0.9130</code></td>
      <td><code>0.00075</code></td>
      <td><code>100</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
    </tr>
  </tbody>
</table>

- Loss Function: MSELoss

```
criterion = nn.MSELoss()
loss = criterion(outputs, F.one_hot(labels, 10).float())
```

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>TSMA</th>
      <th>lr</th>
      <th>epoch</th>
      <th>batch_size</th>
      <th>T</th>
    </tr>
  </thead>
 <tbody>
    <tr>
      <td><code>1</code></td>
      <td><code>0.8890</code></td>
      <td><code>0.001</code></td>
      <td><code>100</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
    </tr>
  </tbody>
</table>

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

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>TSMA</th>
      <th>lr</th>
      <th>epoch</th>
      <th>batch_size</th>
      <th>T</th>
    </tr>
  </thead>
 <tbody>
    <tr>
      <td><code>1</code></td>
      <td><code>0.9311</code></td>
      <td><code>0.00075</code></td>
      <td><code>100</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
    </tr>
  </tbody>
</table>

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

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>TSMA</th>
      <th>lr</th>
      <th>epoch</th>
      <th>batch_size</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>1</code></td>
      <td><code>0.9311</code></td>
      <td><code>0.0005</code></td>
      <td><code>100</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
    </tr>
    <tr>
      <td><code>2</code></td>
      <td><code>0.9286</code></td>
      <td><code>0.0005</code></td>
      <td><code>100</code></td>
      <td><code>128</code></td>
      <td><code>40</code></td>
    </tr>
    <tr>
      <td><code>3</code></td>
      <td><code>0.9313</code></td>
      <td><code>0.001</code></td>
      <td><code>100</code></td>
      <td><code>128</code></td>
      <td><code>40</code></td>
    </tr>
    <tr>
      <td><code>4</code></td>
      <td><code>0.9302</code></td>
      <td><code>0.005</code></td>
      <td><code>100</code></td>
      <td><code>128</code></td>
      <td><code>40</code></td>
    </tr>
    <tr>
      <td><code>5</code></td>
      <td><code>0.9322</code></td>
      <td><code>0.001</code></td>
      <td><code>100</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
    </tr>
    <tr>
      <td><code>6</code></td>
      <td><code>0.9307</code></td>
      <td><code>0.001</code></td>
      <td><code>100</code></td>
      <td><code>64</code></td>
      <td><code>4</code></td>
    </tr>
    <tr>
      <td><code>7</code></td>
      <td><code>0.9330</code></td>
      <td><code>0.001</code></td>
      <td><code>100</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
    </tr>
    <tr>
      <td><code>8</code></td>
      <td><code>0.9319</code></td>
      <td><code>0.0015</code></td>
      <td><code>100</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
    </tr>
  </tbody>
</table>

### Simple model 4

Change MaxPool2d to AvgPool2d


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
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 48)
        self.fc2 = nn.Linear(48, 10)
        self.dropout = nn.Dropout(0.35)

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
        #x = self.dropout(x)
        x = self.fc2(x)
        return x

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 32, 28, 28]          896
├─BatchNorm2d: 1-2                       [-1, 32, 28, 28]          64
├─Conv2d: 1-3                            [-1, 32, 28, 28]          9,248
├─BatchNorm2d: 1-4                       [-1, 32, 28, 28]          64
├─AvgPool2d: 1-5                         [-1, 32, 14, 14]          --
├─Conv2d: 1-6                            [-1, 32, 14, 14]          9,248
├─BatchNorm2d: 1-7                       [-1, 32, 14, 14]          64
├─AvgPool2d: 1-8                         [-1, 32, 7, 7]            --
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

#### Simple model 4 result

* Without data augmentation

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>TSMA</th>
      <th>lr</th>
      <th>epoch</th>
      <th>batch_size</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>1</code></td>
      <td><code>0.9352</code></td>
      <td><code>0.002</code></td>
      <td><code>200</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
    </tr>
  </tbody>
</table>

* Add scheduler

After first tried, suggesting to set the total training epochs to `100`, since the model does not improve accuracy after first `50` epochs.

And I also tried to set the gamma to `0.99` and update the scheduler every `150` epochs, for total `300` epochs with learning rate `0.002`.

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>TSMA</th>
      <th>lr</th>
      <th>epoch</th>
      <th>batch_size</th>
      <th>T</th>
      <th>Scheduler<br>(gamma per eopch)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>1</code></td>
      <td><code>0.9377</code></td>
      <td><code>0.002</code></td>
      <td><code>200</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
      <td><code>0.99</code> / <code>100</code></td>
    </tr>
    <tr>
      <td><code>2</code></td>
      <td><code>0.9345</code></td>
      <td><code>0.003</code></td>
      <td><code>100</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
      <td><code>0.99</code> / <code>100</code></td>
    </tr>
    <tr>
      <td><code>3</code></td>
      <td><code>0.9365</code></td>
      <td><code>0.0025</code></td>
      <td><code>100</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
      <td><code>0.99</code> / <code>100</code></td>
    </tr>
    <tr>
      <td><code>4</code></td>
      <td><code>0.9352</code></td>
      <td><code>0.002</code></td>
      <td><code>300</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
      <td><code>0.99</code> / <code>150</code></td>
    </tr>
  </tbody>
</table>

==**Shell Script Auto**==

Set the gamma to `0.99` and update the scheduler every `50` epochs, for total `500` epochs.
<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>TSMA</th>
      <th>lr</th>
      <th>epoch</th>
      <th>batch_size</th>
      <th>T</th>
      <th>Scheduler<br>(gamma per eopch)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>1</code></td>
      <td><code>0.9358</code></td>
      <td><code>0.0025</code></td>
      <td><code>500</code></td>
      <td><code>128</code></td>
      <td><code>40</code></td>
      <td><code>0.99</code> / <code>50</code></td>
    </tr>
    <tr>
      <td><code>2</code></td>
      <td><code>0.9361</code></td>
      <td><code>0.003</code></td>
      <td><code>500</code></td>
      <td><code>128</code></td>
      <td><code>40</code></td>
      <td><code>0.99</code> / <code>50</code></td>
    </tr>
    <tr>
      <td><code>3</code></td>
      <td><code>0.9377</code></td>
      <td><code>0.0035</code></td>
      <td><code>500</code></td>
      <td><code>128</code></td>
      <td><code>40</code></td>
      <td><code>0.99</code> / <code>50</code></td>
    </tr>
    <tr>
      <td><code>4</code></td>
      <td><code>0.9339</code></td>
      <td><code>0.0025</code></td>
      <td><code>500</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
      <td><code>0.99</code> / <code>50</code></td>
    </tr>
    <tr>
      <td><code>5</code></td>
      <td><code>0.9340</code></td>
      <td><code>0.0025</code></td>
      <td><code>500</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
      <td><code>0.99</code> / <code>50</code></td>
    </tr>
    <tr>
      <td><code>6</code></td>
      <td><code>0.9353</code></td>
      <td><code>0.003</code></td>
      <td><code>500</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
      <td><code>0.99</code> / <code>50</code></td>
    </tr>
  </tbody>
</table>

* Data augmentation: RandomCrop, RandomHorizontalFlip, RandomRotation, ToTensor, RandomGrayscale, Normalize
```
train_transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.RandomCrop(28, padding=4),
        transforms.Normalize((0.5,), (0.5,))])
```

### Simple model 5

```
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
       # 3 layers of convolutions, less than 100k parameters
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 33, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(33)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(33 * 7 * 7, 49)
        self.fc2 = nn.Linear(49, 10)
        self.dropout = nn.Dropout(0.35)

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
        x = x.view(-1, 33 * 7 * 7)
        x = self.dropout(x)
        x = self.fc1(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        return x

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 32, 28, 28]          896
├─BatchNorm2d: 1-2                       [-1, 32, 28, 28]          64
├─Conv2d: 1-3                            [-1, 32, 28, 28]          9,248
├─BatchNorm2d: 1-4                       [-1, 32, 28, 28]          64
├─AvgPool2d: 1-5                         [-1, 32, 14, 14]          --
├─Conv2d: 1-6                            [-1, 33, 14, 14]          9,537
├─BatchNorm2d: 1-7                       [-1, 33, 14, 14]          66
├─AvgPool2d: 1-8                         [-1, 33, 7, 7]            --
├─Dropout: 1-9                           [-1, 1617]                --
├─Linear: 1-10                           [-1, 49]                  79,282
├─Linear: 1-11                           [-1, 10]                  500
==========================================================================================
Total params: 99,657
Trainable params: 99,657
Non-trainable params: 0
Total mult-adds (M): 9.85
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.86
Params size (MB): 0.38
Estimated Total Size (MB): 1.25
```
#### Simple model 5 result

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>TSMA</th>
      <th>lr</th>
      <th>epoch</th>
      <th>batch_size</th>
      <th>T</th>
      <th>Scheduler<br>(gamma per eopch)</th>
      <th>Dropout</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>1</code></td>
      <td><code>0.9357</code></td>
      <td><code>0.0025</code></td>
      <td><code>100</code></td>
      <td><code>64</code></td>
      <td><code>40</code></td>
      <td><code>0.99</code> / <code>50</code></td>
      <td><code>0.35</code></td>
    </tr>
  </tbody>
</table>

### Demo Platform
- OS: Ubuntu 20.04
- CPU: AMD Ryzen Threadripper (will set num_worker=8)
- GPU: RTX 3080 (8GB) *1
- Python 3.8.10
- CUDA: 11.07
- Framework: PyTorch 1.13.1