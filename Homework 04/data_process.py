from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse
import shutil

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('.jpg', '.txt')
    points = []
    with open (mat_path, 'r') as f:
        while True:
            point = f.readline()
            if not point:
                break
            point = point.split(' ')
            points.append([float(point[0]), float(point[1])])

    points = np.array(points)

    if len(points>0):
        idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
        points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:

        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr

    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='data',
                        help='original data directory')
    parser.add_argument('--data-dir', default='data_processed',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048

    sub_dir = os.path.join(args.origin_dir, 'train')
    sub_save_dir = os.path.join(save_dir, 'train')
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)
    im_list = glob(os.path.join(sub_dir, '*jpg'))
    for im_path in im_list:
        name = os.path.basename(im_path)
        #print(name)
        im, points = generate_data(im_path)

        im_save_path = os.path.join(sub_save_dir, name)
        im.save(im_save_path, quality=95)

        gd_save_path = im_save_path.replace('jpg', 'npy')
        np.save(gd_save_path, points)
