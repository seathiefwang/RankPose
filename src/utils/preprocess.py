from __future__ import division
import numpy as np
import cv2


def minmax_normalize(img, norm_range=(0, 1), orig_range=(0, 255)):
    # range(0, 1)
    norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
    # range(min_value, max_value)
    norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
    return norm_img


def meanstd_normalize(img, mean, std):
    mean = np.asarray(mean)
    std = np.asarray(std)
    norm_img = (img - mean) / std
    return norm_img

def preprocess(img):
    img = img / 256.
    img = (img - np.asarray([0.485, 0.456, 0.406])) / np.asarray([0.229, 0.224, 0.225])
    return img

def change_bbox(bbox, scale=1, use_forehead=True):
    x_min, y_min, x_max, y_max = bbox
    if use_forehead:
        k = scale
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
    else:
        w, h = x_max-x_min, y_max-y_min
        h_w = max(h, w) * scale
        x_min -= (h_w-w)//2
        y_min -= (h_w-h)//2
        x_max = x_min + h_w
        y_max = y_min + h_w
    return (int(x_min), int(y_min), int(x_max), int(y_max))

def padding(img, pad, constant_values=0):
    pad_img = np.pad(img, pad, 'constant', constant_values=constant_values)
    return pad_img


def clahe(img, clip=2, grid=8):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    img_yuv[:, :, 0] = _clahe.apply(img_yuv[:, :, 0])
    img_equ = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2BGR)
    return img_equ

