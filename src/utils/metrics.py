from __future__ import division, print_function, absolute_import
from __future__ import print_function
from utils.functional import quat2euler
import numpy as np
import torch
import cv2
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

__all__ = ['calculate_diff']

def calculate_diff(output, target, mean=True):
    _, ypr_or_qua = target.shape
    if ypr_or_qua == 4:
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        output_ypr = []
        target_ypr = []
        for i in range(len(output)):
            ypr1 = quat2euler(*output[i])
            ypr2 = quat2euler(*target[i])
            output_ypr.append(ypr1)
            target_ypr.append(ypr2)

        dif = np.abs(np.asarray(output_ypr) - np.asarray(target_ypr))
        if mean:
            dif = np.mean(dif)
    else:
        dif = torch.abs(output - target)
        if mean:
            dif = torch.mean(dif).item()
        else:
            dif = dif.detach().cpu().numpy()
    return dif

if __name__=='__main__':
    output = torch.randn((32, 3))
    target = torch.randn((32, 3))
    print(calculate_diff(output, target))