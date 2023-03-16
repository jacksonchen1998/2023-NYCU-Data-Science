import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torchvision.models as models
from torchsummary import summary
import numpy as np
import pandas as pd
import torch.nn.functional as F
import copy
import datetime
import warnings

warnings.filterwarnings("ignore")

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
        self.dropout = nn.Dropout(0.1)

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

# tarin student model
def train_student_model(student_model, trainloader, testloader, num_epochs=10, learning_rate=0.001, factor=0.2, patience=5, min_lr=0.00001):
    student_model = student_model.to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=min_lr, eps=1e-08)
    criterion = nn.CrossEntropyLoss()
    best_model_wts = copy.deepcopy(student_model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                student_model.train()
            else:
                student_model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in trainloader if phase == 'train' else testloader:
                inputs = inputs.to(device)
                labels = labels.to(device) # 32, type: torch.int64
                print(optimizer.state_dict()['param_groups'][0]['lr'])
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # print learning rate
                    
                    outputs = student_model(inputs) # 32, 10
                    _, preds = torch.max(outputs, 1) # 32, type: torch.int64
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(trainloader.dataset) if phase == 'train' else running_loss / len(testloader.dataset)
            epoch_acc = running_corrects.double() / len(trainloader.dataset) if phase == 'train' else running_corrects.double() / len(testloader.dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(student_model.state_dict())
                print(f"Best val Acc: {best_acc:4f}")
            if phase == 'test':
                print(f"Best val Acc: {best_acc:.4f}")
            if phase == 'train':
                scheduler.step(epoch_loss)
        print()
    print(f"Best val Acc: {best_acc:4f}")
    student_model.load_state_dict(best_model_wts)
    return student_model, best_acc

# teacher_student_distillation
def distill(teacher_model, student_model, trainloader, testloader, num_epochs=10, learning_rate=0.001, T=20, alpha=0.5, factor=0.5, patience=5, min_lr=0.00001):
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    # scheduler set min learning rate to 0.0001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=min_lr, eps=1e-08)
    # patience: number of epochs with no improvement after which learning rate will be reduced, the checking parameter is loss
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    cur_best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                teacher_model.eval()
                student_model.train()
                # print student model learning rate
                print("learning_rate: ", optimizer.state_dict()['param_groups'][0]['lr'])
            else:
                teacher_model.eval()
                student_model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in trainloader if phase == 'train' else testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.no_grad():
                        teacher_outputs = teacher_model(inputs)
                    student_outputs = student_model(inputs)
                    # distillation loss 0.5 * student_loss + 0.5 * distillation_loss
                    student_loss = criterion(student_outputs, labels)
                    distillation_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T*T * 2.0 * alpha)
                    loss_1 = student_loss + distillation_loss
                    # student loss with ground truth
                    loss_2 = criterion(student_outputs, labels)
                    # choose better loss
                    loss = loss_1 * 0.5 + loss_2 * 0.5
                    _, preds = torch.max(student_outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(trainloader.dataset) if phase == 'train' else running_loss / len(testloader.dataset)
            epoch_acc = running_corrects.double() / len(trainloader.dataset) if phase == 'train' else running_corrects.double() / len(testloader.dataset)
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(student_model.state_dict())
            if phase == 'test':
                print(f"Best val Acc: {best_acc:.4f}")
            if phase == 'train':
                scheduler.step(epoch_loss)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print()
        
    print(f"Best val Acc: {best_acc:4f}")
    student_model.load_state_dict(best_model_wts)
    return student_model, best_acc

# test
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

# save student model's prediction into csv file [id, label] with dataframe
# id: 0~9999
def save_csv(model, testloader, filename):
    model = model.to(device)
    model.eval()
    ids = []
    labels = []
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        ids.extend(preds.cpu().numpy())
    df = pd.DataFrame({'id': range(10000), 'pred': ids})
    df.to_csv(filename, index=False)

# test
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


if __name__ == '__main__':
    # python command: python hw2.py --batch_size 128 --num_epochs 100 --learning_rate 0.001 --T 20 --alpha 0.5
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--factor', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    args = parser.parse_args()

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.15),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Normalize((0.5,), (0.5,))])

    test_transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # build teacher model with ./resnet50.pth
    teacher_model = models.resnet50(pretrained=False)
    teacher_model.fc = nn.Linear(2048, 10)
    teacher_model = teacher_model.to(device)

    # distill
    student_model = StudentNet()
    summary(student_model, (3, 28, 28))

    #only, best_acc = train_student_model(student_model, trainloader, testloader, args.num_epochs, args.learning_rate)

    # more student_model alpha, more student_model loss, alpha = 0.5
    distill_model, best_acc = distill(teacher_model, student_model, trainloader, testloader, args.num_epochs, args.learning_rate, args.T, args.alpha, args.factor, args.patience, args.min_lr)
    # sabe pth file and it can be loaded by map_location = torch.device('cpu')
    # get datatime, store month, day, hour, minute, but not second
    now = datetime.datetime.now()
    torch.save(distill_model.state_dict(), './pth_folder/' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute) + '_' + str(int(best_acc*10000)) + '.pth')
    test(distill_model, testloader)
    save_csv(distill_model, testloader, './submission_folder/' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute) + '_' + str(int(best_acc*10000)) + '.csv')