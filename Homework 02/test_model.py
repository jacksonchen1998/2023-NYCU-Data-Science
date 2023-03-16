import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
import argparse
import torchsummary as summary

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
        #x = self.dropout(x)
        x = self.fc1(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        return x


def test(model, testloader):
    model = model.to(device)
    model.eval()
    running_corrects = 0
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects.double() / len(testloader.dataset)
    print(f"Test Acc: {acc:.4f}")


if __name__ == "__main__":

    # python test_model.py --model distill_model.pth
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distill_model.pth', help='path to model')
    args = parser.parse_args()

    test_transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=test_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

    model = StudentNet()

    checkpoint = torch.load(args.model, map_location='cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(checkpoint, strict=False)
    summary.summary(model, (3, 28, 28))
    test(model, testloader)