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

# model load pretrained model
# download pretrained model from https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
# pretrained_dict = torch.load('./vgg19-dcbb9e9d.pth')
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

# %%
batch_size = 8
num_workers = 4
learning_rate = 2e-5
total_epoch = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=8e-06) # T_max means total epoch

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

# dataloaders = {x: DataLoader(datasets[x],
#                                           collate_fn=(train_collate
#                                                       if x == 'train' else default_collate),
#                                           batch_size=(batch_size
#                                           if x == 'train' else 1),
#                                           shuffle=(True if x == 'train' else False),
#                                           num_workers=num_workers,
#                                           pin_memory=(True if x == 'train' else False))
#                             for x in ['train', 'test']}
# dataloaders = {x: DataLoader(datasets[x], collate_fn=train_collate, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
#             for x in ['train', 'test']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'test']}


# %%
post_prob = Post_Prob(sigma=8.0, c_size=512, stride=8, background_ratio=1, use_background=True, device=device)
criterion = Bay_Loss(use_background=True, device=device)

# %%
best_loss = 1e6
best_mae = 1e6
best_mse = 1e6
best_rmse = 1e6

# random spilt train and val from train dataset
train_idx, val_idx = torch.utils.data.random_split(range(dataset_sizes['train']), [int(dataset_sizes['train'] * 0.8), int(dataset_sizes['train'] * 0.2)])
train_loader = DataLoader(datasets['train'], collate_fn=train_collate, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(datasets['train'], collate_fn=train_collate, batch_size=batch_size, shuffle=True, num_workers=num_workers)

for epoch in range(total_epoch):
    
    train_loss, train_mae, train_mse, train_rmse = 0.0, 0.0, 0.0, 0.0
    val_loss, val_mae, val_mse, val_rmse = 0.0, 0.0, 0.0, 0.0

    print('Epoch {}/{}'.format(epoch, total_epoch - 1))
    print('-' * 40)
    model.train()
    for steps, (inputs, points, targets, st_sizes) in enumerate(train_loader):
        inputs = inputs.to(device)
        points = [point.to(device) for point in points]
        targets = [target.to(device) for target in targets]
        st_sizes = st_sizes.to(device)
        gd_count = np.array([len(p) for p in points], dtype=np.float32)

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            prob_list = post_prob(points, st_sizes)
            loss = criterion(prob_list, targets, outputs)

            train_loss += loss.item()
           
            N = inputs.size(0)
            pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
            res = pre_count - gd_count
            train_mae += np.mean(np.fabs(res))
            train_mse += np.mean(res ** 2)
            train_rmse += np.sqrt(np.mean(res ** 2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for steps, (inputs, points, targets, st_sizes) in enumerate(val_loader):
        inputs = inputs.to(device)
        points = [point.to(device) for point in points]
        targets = [target.to(device) for target in targets]
        st_sizes = st_sizes.to(device)
        gd_count = np.array([len(p) for p in points], dtype=np.float32)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            prob_list = post_prob(points, st_sizes)
            loss = criterion(prob_list, targets, outputs)

            val_loss += loss.item()
            
            N = inputs.size(0)
            pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
            res = pre_count - gd_count
            val_mae += np.mean(np.fabs(res))
            val_mse += np.mean(res ** 2)
            val_rmse += np.sqrt(np.mean(res ** 2))

    if (val_loss / len(val_loader)) < best_loss:
        best_loss = val_loss / len(val_loader)
        best_mae = val_mae / len(val_loader)
        best_mse = val_mse / len(val_loader)
        best_rmse = val_rmse / len(val_loader)
        torch.save(model.state_dict(), 'best_model.pth')
        print('Model Saved!')

    print('Train')
    print('Best Loss: {:.4f}, Current Loss: {:.4f}'.format(best_loss, train_loss / len(train_loader)))
    print('Best MAE: {:.4f}, Current MAE: {:.4f}'.format(best_mae, train_mae / len(train_loader)))
    print('Best MSE: {:.4f}, Current MSE: {:.4f}'.format(best_mse, train_mse / len(train_loader)))
    print('Best RMSE: {:.4f}, Current RMSE: {:.4f}'.format(best_rmse, train_rmse / len(train_loader)))

    print()
    
    print('Val')
    print('Best Loss: {:.4f}, Current Loss: {:.4f}'.format(best_loss, val_loss / len(val_loader)))
    print('Best MAE: {:.4f}, Current MAE: {:.4f}'.format(best_mae, val_mae / len(val_loader)))
    print('Best MSE: {:.4f}, Current MSE: {:.4f}'.format(best_mse, val_mse / len(val_loader)))
    print('Best RMSE: {:.4f}, Current RMSE: {:.4f}'.format(best_rmse, val_rmse / len(val_loader)))

    scheduler.step(val_loss / len(val_loader))
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


