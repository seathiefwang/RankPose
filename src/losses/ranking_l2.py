import torch.nn as nn
import torch
import torch.nn.functional as F

class RankingL2loss(nn.Module):
    def __init__(self, margin=0):
        super().__init__()
        self.margin = margin
        

    def forward(self, output1, output2, targets):
        # l2_dis = torch.pow(output1-output2, 2)

        loss = torch.mean(torch.pow(torch.clamp(targets * (output1-output2), min=0.0), 2))
        return loss
