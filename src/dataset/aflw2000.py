# coding:utf-8
from __future__ import print_function
from functools import partial
import sys
sys.path.append("..")
import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as albu
from torch.utils.data import DataLoader, Dataset
from utils.preprocess import preprocess, change_bbox
from albumentations.augmentations import functional as F
from utils.functional import get_pt_ypr_from_mat, euler2quat

class AFLW2000Dataset(Dataset):
    def __init__(self, base_dir=None, split='train', affine_augmenter=None, image_augmenter=None, 
            target_size=224, filename=None, use_bined=False, n_class=4, debug=False):
        self.base_dir = base_dir
        self.base_dir = Path(base_dir)
        self.split = split
        self.use_bined = use_bined
        self.n_class = n_class
        self.debug = debug

        self.img_paths = []
        self.bbox = []
        self.labels = []
        self.euler_binned = []

        with open(self.base_dir / filename) as f:
            for i, line in enumerate(f.readlines()):
                ls = line.strip()

                mat_path = self.base_dir / ls.replace('.jpg', '.mat')
                bbox, pose = get_pt_ypr_from_mat(mat_path, pt3d=True)

                if True and (abs(pose[0])>99 or abs(pose[1])>99 or abs(pose[2])>99):
                    continue

                if use_bined:
                    yaw_pitch_bins = np.array([-60, -40, -20, 20, 40, 60])
                    roll_bins = np.array(range(-81, 82, 9))
                    self.euler_binned.append([np.digitize(pose[0], yaw_pitch_bins),np.digitize(pose[1], yaw_pitch_bins),np.digitize(pose[2], roll_bins)])

                self.labels.append(np.array(pose))
                self.bbox.append(bbox)
                self.img_paths.append(ls)

        self.labels_sort_idx = np.argsort(-np.mean(np.abs(self.labels), axis=1))
        

        if 'train' in self.split:
            self.resizer = albu.Compose([albu.SmallestMaxSize(target_size, p=1.),
                                        albu.RandomScale(scale_limit=(-0.2, 0.2), p=0.1),
                                        albu.PadIfNeeded(min_height=target_size, min_width=target_size, value=0, p=1),
                                        albu.RandomCrop(target_size, target_size, p=1.)])
        else:
            # self.resizer = albu.Compose([albu.Resize(target_size[0], target_size[1], p=1.)])
            self.resizer = albu.Compose([albu.SmallestMaxSize(target_size, p=1.),
                                        albu.CenterCrop(target_size, target_size, p=1.)])

        self.affine_augmenter = affine_augmenter
        self.image_augmenter = image_augmenter

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.base_dir /  self.img_paths[index]
        bbox = change_bbox(self.bbox[index], 1.4, use_forehead=False)
        img = np.array(Image.open(img_path).crop(bbox))

        label = self.labels[index].copy()
        if self.use_bined:
            bined_label = self.euler_binned[index].copy()

        # ImageAugment (RandomBrightness, AddNoise...)
        if self.image_augmenter:
            augmented = self.image_augmenter(image=img)
            img = augmented['image']

        # Resize (Scale & Pad & Crop)
        if self.resizer:
            resized = self.resizer(image=img)
            img = resized['image']
        # AffineAugment (Horizontal Flip, Rotate...)
        if self.affine_augmenter:
            augmented = self.affine_augmenter(image=img)
            img = augmented['image']

        if self.split=='train':
            # 图片左右翻转
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                label[0] = -label[0]
                label[2] = -label[2]
                if self.use_bined:
                    bined_label[0] = -(bined_label[0]-3)+3
                    bined_label[2] = -(bined_label[2]-9)+10
            if random.random() < 0.5 and abs(label[0])<30 and abs(label[2])<30:
                if random.random() < 0.5:
                    factor = 1
                    label[2] += 90
                    if self.use_bined:
                        bined_label[2] = min(bined_label[2] + 10, 20)
                else:
                    factor = 3
                    label[2] -= 90
                    if self.use_bined:
                        bined_label[2] = max(bined_label[2] - 10, 0)

                img = np.ascontiguousarray(np.rot90(img, factor))

        if self.n_class == 4:
            label = euler2quat(*label)

        if self.debug:
            print(self.bbox[index])
            print(label)
        else:
            img = preprocess(img)
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor(img)
            label = torch.FloatTensor(label)
        if self.use_bined:
            return img, label, bined_label[0], bined_label[1], bined_label[2]
        else:
            return img, label

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    affine_augmenter = None
    image_augmenter = albu.Compose([albu.GaussNoise((0, 25), p=.5),
                                    albu.RandomBrightnessContrast(0.4, 0.3, p=1),
                                    albu.JpegCompression(90, 100, p=0.5)])
    #image_augmenter = None
    image_augmenter = albu.Compose([albu.RandomBrightnessContrast(0.4,0.3,p=0.5),
                                    albu.RandomGamma(p=0.3),
                                    albu.CLAHE(p=0.1),
                                    albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20,p=0.2),
                                    ])
    dataset = AFLW2000Dataset(base_dir="", affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                             filename='aflw2000_filename.txt', split='valid', target_size=224, debug=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(dataset))

    for i, batched in enumerate(dataloader):
        images, labels = batched
        print(images.shape)
        for j in range(8):
            img = images[j].numpy()
            img = img.astype('uint8')
            img = Image.fromarray(img)
            img.save('tmp/%d_%d.jpg'%(i, j))
        if i > 2:
            break





