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

### Demo Platform
- OS: Ubuntu 20.04
- CPU: AMD Ryzen Threadripper (will set num_worker=8)
- GPU: RTX 3080 (8GB) *1
- Python 3.8.10
- CUDA: 11.07
- Framework: PyTorch 1.13.1