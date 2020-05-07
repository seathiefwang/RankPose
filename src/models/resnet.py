import math
import torch
import torch.nn as nn
import torchvision.models as models
from .cus_layer import l2_norm

class ResNet(nn.Module):
    def __init__(self, n_class=4, use_norm=True):
        super().__init__()
        self.n_class = n_class
        self.use_norm = use_norm
        resnet = models.resnet50(pretrained=True)


        self.model = nn.Sequential(
                            resnet.conv1,
                            resnet.bn1,
                            resnet.relu,
                            resnet.maxpool,

                            resnet.layer1,
                            resnet.layer2,
                            resnet.layer3,
                            resnet.layer4,

                            resnet.avgpool,
                            )
        
        self.dropout = nn.Dropout(0.2)
        if self.use_norm:
            self.w = nn.Parameter(torch.Tensor(2048, n_class))
        else:
            self.fc_angles = nn.Linear(2048, n_class)
        self.fc_y = nn.Linear(2048, 7) # -60, -40, -20, 20, 40, 60
        self.fc_p = nn.Linear(2048, 7) # -60, -40, -20, 20, 40, 60
        self.fc_r = nn.Linear(2048, 20) # -81, -72, -63, -54, -45, -36, -27, -18, -9, 0, ... 81

    def forward(self, x, use_bined=False):
        x = self.model(x)
        x = torch.flatten(x, 1)

        if self.use_norm:
            return torch.matmul(l2_norm(x, 1), l2_norm(self.w, 0)) * 180

        x = self.dropout(x)
        
        angles = self.fc_angles(x)

        if self.n_class == 4:
            angles = l2_norm(angles)
        
        if use_bined:
            yaw_lbl = self.fc_y(x)
            pitch_lbl = self.fc_p(x)
            roll_lbl = self.fc_r(x)
            return angles, yaw_lbl, pitch_lbl, roll_lbl
        else:
            return angles
