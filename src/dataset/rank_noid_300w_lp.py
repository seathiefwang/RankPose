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


class RankNoIDDataset(Dataset):
    def __init__(self, base_dir=None, split='train', affine_augmenter=None, image_augmenter=None, 
            target_size=224, filename=None, use_bined=False, n_class=3, debug=False):
        self.base_dir = base_dir
        self.base_dir = Path(base_dir)
        self.split = split
        self.use_bined = use_bined
        self.n_class = n_class
        self.debug = debug

        self.img_paths = []
        self.bboxs = []
        self.labels = []
        self.euler_binned = []

        with open(self.base_dir / filename) as f:
            for i, line in enumerate(f.readlines()):
                ls = line.strip()

                mat_path = self.base_dir / ls.replace('.jpg', '.mat')
                bbox, pose = get_pt_ypr_from_mat(mat_path)

                if False and (abs(pose[0])>90 or abs(pose[1])>90 or abs(pose[2])>90):
                    continue

                if use_bined:
                    yaw_pitch_bins = np.array([-60, -40, -20, 20, 40, 60])
                    roll_bins = np.array(range(-81, 82, 9))
                    self.euler_binned.append([np.digitize(pose[0], yaw_pitch_bins),np.digitize(pose[1], yaw_pitch_bins),np.digitize(pose[2], roll_bins)])

                self.labels.append(np.array(pose))
                self.bboxs.append(bbox)
                self.img_paths.append(ls)
                if i > 500:
                    break

        self.labels_sort_idx = np.argsort(-np.mean(np.abs(self.labels), axis=1))
        

        if 'train' in self.split:
            self.resizer = albu.Compose([albu.SmallestMaxSize(int(target_size * 1.1), p=1.),
                                        albu.RandomScale(scale_limit=(-0.1, 0.1), p=1),
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
        idxs = np.random.choice(len(self.img_paths), 2, replace=False)
        if index not in idxs:
            idxs[0] = index

        img_path1 = self.base_dir /  self.img_paths[idxs[0]]
        img_path2 = self.base_dir /  self.img_paths[idxs[1]]

        scale = np.random.random_sample() * 0.2 + 0.1
        bbox1 = change_bbox(self.bboxs[idxs[0]], scale=scale, use_forehead=True)
        bbox2 = change_bbox(self.bboxs[idxs[1]], scale=scale, use_forehead=True)
        img1 = np.array(Image.open(img_path1).crop(bbox1))
        img2 = np.array(Image.open(img_path2).crop(bbox2))

        lbl1 = self.labels[idxs[0]]
        lbl2 = self.labels[idxs[1]]
        if self.use_bined:
            bined_lbl1 = self.euler_binned[idxs[0]]
            bined_lbl2 = self.euler_binned[idxs[1]]


        # ImageAugment (RandomBrightness, AddNoise...)
        if self.image_augmenter:
            augmented = self.image_augmenter(image=img1)
            img1 = augmented['image']
            augmented = self.image_augmenter(image=img2)
            img2 = augmented['image']

        # Resize (Scale & Pad & Crop)
        if self.resizer:
            resized = self.resizer(image=img1)
            img1 = resized['image']
            resized = self.resizer(image=img2)
            img2 = resized['image']
        # AffineAugment (Horizontal Flip, Rotate...)
        if self.affine_augmenter:
            augmented = self.affine_augmenter(image=img1)
            img1 = augmented['image']
            augmented = self.affine_augmenter(image=img2)
            img2 = augmented['image']

        # label = (abs(lbl1) > abs(lbl2)) * 2 - 1
        label = np.sign(abs(lbl1) - abs(lbl2))

        if self.n_class==4:
            lbl1 = euler2quat(*lbl1)
            lbl2 = euler2quat(*lbl2)

        if self.debug:
            print(label)
            return img1, img2
        else:
            img1 = preprocess(img1)
            img1 = torch.FloatTensor(img1).permute(2, 0, 1)
            img2 = preprocess(img2)
            img2 = torch.FloatTensor(img2).permute(2, 0, 1)

            label = torch.FloatTensor(label.astype(np.float32))
            lbl1 = torch.FloatTensor(lbl1)
            lbl2 = torch.FloatTensor(lbl2)
            if self.use_bined:
                return img1, img2, lbl1, lbl2, label, int(bined_lbl1[0]), int(bined_lbl1[1]), int(bined_lbl1[2]),\
                        int(bined_lbl2[0]), int(bined_lbl2[1]), int(bined_lbl2[2])
            else:
                return img1, img2, lbl1, lbl2, label

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.custum_aug import Rotate

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
    dataset = RankNoIDDataset(base_dir="F:\\pingan\\headPose\\data", affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                             filename='300w_lp_filename.txt', split='train', target_size=224, debug=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(dataset))

    for i, batched in enumerate(dataloader):
        img1, img2 = batched
        for j in range(8):
            img = img1[j].numpy()
            img = img.astype('uint8')
            img = Image.fromarray(img)
            img.save('tmp/img1_%d_%d.jpg'%(i, j))
            img = img2[j].numpy()
            img = img.astype('uint8')
            img = Image.fromarray(img)
            img.save('tmp/img2_%d_%d.jpg'%(i, j))
        if i > 2:
            break


