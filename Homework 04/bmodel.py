# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
from PIL import Image
import cv2
import glob
from tqdm import tqdm
# pytorch image generator
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
# torchvision import
import random
from bay_loss import Bay_Loss
from post_prob import Post_Prob
from torch.nn import functional as F
from torch.utils.data.dataloader import default_collate

import warnings

warnings.filterwarnings("ignore")

# %%
def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area

class Crowd(Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):
        self.root_path = root_path
        self.im_list = sorted(glob.glob(os.path.join(self.root_path, '*.jpg')))
        if method not in ['train', 'test']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'test':
            #keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, name

    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)
        assert st_size >= self.c_size # assert the crop size is smaller than the original image
        #assert len(keypoints) > 0 # assert there is at least one person in the image
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = transforms.functional.crop(img, i, j, h, w)
        
        #nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

        # nearest_dis = np.minimum(keypoints[:, 0], keypoints[:, 1])
        # nearest_dis = np.clip(nearest_dis, 0.0, st_size)

        # points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        # points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        # bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        # inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
        # origin_area = nearest_dis * nearest_dis
        # # ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        # mask = (ratio >= 0.5)
        # keypoints = keypoints[mask]

        if len(keypoints) > 0:
            idx_mask = (keypoints[:, 0] >= j) * (keypoints[:, 0] <= j + 512) * (keypoints[:, 1] >= i) * (keypoints[:, 1] <= i + 512)
            keypoints = keypoints[idx_mask]
            keypoints = keypoints - [j, i]  # change coodinate
        target = np.ones(len(keypoints))

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size

# %%
class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    return model

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = vgg19().to(device)

# %%
batch_size = 8
num_workers = 4
learning_rate = 8e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=8e-6, verbose=True)

# %%
def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

datasets = {x: Crowd(os.path.join('./', x),
                                  crop_size=512,
                                  downsample_ratio=8,
                                  is_gray=False) for x in ['train', 'test']}

dataloaders = {x: DataLoader(datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=num_workers,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'test']}
dataloaders = {x: DataLoader(datasets[x], collate_fn=train_collate, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
            for x in ['train', 'test']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'test']}


# %%
post_prob = Post_Prob(sigma=8.0, c_size=512, stride=8, background_ratio=1, use_background=True, device=device)
criterion = Bay_Loss(use_background=True, device=device)

# %%
total_epoch = 1000
best_loss = 1e6
best_mae = 1e6
best_mse = 1e6
best_rmse = 1e6

for epoch in range(total_epoch):
    temp_loss = 0.0
    temp_mae = 0.0
    temp_mse = 0.0
    temp_rmse = 0.0
    print('Epoch {}/{}'.format(epoch, total_epoch - 1))
    print('-' * 40)
    model.train()
    for steps, (inputs, points, targets, st_sizes) in enumerate(dataloaders['train']):
        inputs = inputs.to(device)
        points = [point.to(device) for point in points]
        targets = [target.to(device) for target in targets]
        st_sizes = st_sizes.to(device)
        gd_count = np.array([len(p) for p in points], dtype=np.float32)

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            prob_list = post_prob(points, st_sizes)
            loss = criterion(prob_list, targets, outputs)
            temp_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            N = inputs.size(0)
            pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
            res = pre_count - gd_count
            temp_mae += np.mean(np.fabs(res))
            temp_mse += np.mean(res ** 2)
            temp_rmse += np.sqrt(np.mean(res ** 2))

    print('Best Loss: {:4f}, Current Loss: {:4f}'.format(best_loss, temp_loss / len(dataloaders['train'])))
    print('Best MAE: {:4f}, Current MAE: {:4f}'.format(best_mae, temp_mae / len(dataloaders['train'])))
    print('Best MSE: {:4f}, Current MSE: {:4f}'.format(best_mse, np.sqrt(temp_mse / len(dataloaders['train']))))
    print('Best RMSE: {:4f}, Current RMSE: {:4f}'.format(best_rmse, temp_rmse / len(dataloaders['train'])))

    if (temp_loss / len(dataloaders['train'])) < best_loss:
        best_loss = temp_loss / len(dataloaders['train'])
        best_mae = temp_mae / len(dataloaders['train'])
        best_mse = np.sqrt(temp_mse / len(dataloaders['train']))
        best_rmse = temp_rmse / len(dataloaders['train'])
        torch.save(model.state_dict(), 'best_model.pth')
        print('Model Saved!')
    scheduler.step(loss.item())
    print('-' * 40)
    print()

torch.save(model.state_dict(), 'final_model.pth')
    
# %%
# test

test_dataset = Crowd(os.path.join('./', 'test'), 512, 8, is_gray=False, method='test')

test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False,
                                             num_workers=0, pin_memory=False)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

pred = np.zeros(len(test_loader))
index = 0

for img, name in test_loader:
    img = img.to(device)
    with torch.no_grad():
        output = model(img)
        print("Name: {}, People: {}".format(name, torch.sum(output).item()))
        pred[index] = torch.sum(output).item()
    index += 1


# %%
import csv

with open('result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Count'])
    for i in range(len(pred)):
        writer.writerow([i+1, pred[i]])


