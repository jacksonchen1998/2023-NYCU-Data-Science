import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
import numpy as np
from torchsummary import summary
import torch.nn as nn
import matplotlib.pyplot as plt
import csv

with open("./train.pkl", "rb") as f:
    train = pickle.load(f) # a dictionary

with open("./validation.pkl", "rb") as f:
    val = pickle.load(f) # a dictionary
    
with open("./test.pkl", "rb") as f:
    test = pickle.load(f) # a dictionary

new_train = {}
new_train['images'] = np.concatenate((train['images'], val['images']), axis=0)
new_train['labels'] = np.concatenate((train['labels'], val['labels']), axis=0)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize((0.5,), (0.5,))])

val_transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3),
        transforms.Normalize((0.5,), (0.5,))])

class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels, transform=None):
                # load data from numpy array
                self.images = torch.from_numpy(images) # transform from numpy to tensor
                self.labels = labels # transform from numpy to tensor, but it is not necessary, since the label is a number
                self.transform = transform
                
        def __len__(self):
                return len(self.images)
        
        def __getitem__(self, idx):
                image = self.images[idx]
                label = self.labels[idx]
                if self.transform:
                        image = self.transform(image)
                return image, label

# doing data augmentation for train["images"]
train_dataset = CustomDataset(new_train["images"], new_train["labels"], transform=train_transform)
val_dataset = CustomDataset(val["images"], val["labels"], transform=val_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=0)


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None: # identity mapping used
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    
class ResNet18(torch.nn.Module):
    def __init__(self, num_classes=80):
        super(ResNet18, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2) 
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, block_num, stride=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride=stride))
        for i in range(1, block_num):
            layers.append(ResNetBlock(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def predict(image_tensor, model):
        model.eval()  # switch to evaluation mode
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            return predicted.item()


#for param in model.parameters():
#    param.requires_grad = False

qry_images = test["qry_images"]
# test_labels size is 15000
test_labels = np.zeros((15000))

test_acc = []
test_loss = []

for task_idx in range(600):

    sup_transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3),
        transforms.Normalize((0.5,), (0.5,))])

    sup_dataset = CustomDataset(test["sup_images"][task_idx], test["sup_labels"][task_idx], transform=sup_transform)
    sup_loader = torch.utils.data.DataLoader(sup_dataset, batch_size = 25, shuffle=False, num_workers=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNet18().to(device)
    checkpoint = torch.load('model.pth', map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 80)

    # Add a new fully connected layer after the original classifier to output 5 classes
    #fnn = nn.Sequential(
    #    nn.Linear(80, 5),
    #)

    fnn = nn.Sequential(
        #nn.Linear(80, 5),
        nn.Linear(80, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 5)
    )

    # Combine the ResNet18 model and the FNN
    fine_model = nn.Sequential(model, fnn).to(device)
    optimizer = torch.optim.Adam(fine_model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6000, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    fine_model.train()
    sup_loss = 0
    sup_acc = 0
    epoch = 0
    # fine tune the model with sup images util the accuracy larger than 0.95
    while sup_acc/(25) < 0.95 and epoch < 50:
        sup_acc = 0
        sup_loss = 0
        for batch_idx, (data, target) in enumerate(sup_loader):
            optimizer.zero_grad()
            output = fine_model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
            sup_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            sup_acc += (predicted == target.to(device)).sum().item()
        epoch += 1
        print("Task: {}, Epoch: {}, Accuarcy: {}, Loss: {}".format(task_idx, epoch, sup_acc/(25), sup_loss/(25)))
        # predict the qry images
        fine_model.eval()
        with torch.no_grad():
            # use qry_images[i][j] to predict the label, but no label
            # np array to tensor
            for j in range(25):
                image = torch.from_numpy(qry_images[task_idx][j])
                image = image.unsqueeze(0)
                
                image = image.to(device)
                output = fine_model(image)
                _, predicted = torch.max(output, 1)
                
                # save the predicted label to test_labels[j]
                test_labels[task_idx*25+j] = predicted.item()
    test_acc.append(sup_acc/(25))
    test_loss.append(sup_loss/(25))

#plot the loss and accuracy
plt.figure(figsize=(10, 10))
plt.plot(test_loss, label='sup')
plt.xlabel('Epoch')
plt.ylabel('Sup image loss')
plt.title('Loss')
plt.savefig('Sup_Loss.png')
plt.legend()

plt.figure(figsize=(10, 10))
plt.plot(test_acc, label='sup')
plt.title('Sup image accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('Sup_Accuracy.png')
plt.legend()

with open('test_labels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Category'])
    for i in range(15000):
        writer.writerow([i, int(test_labels[i])])

