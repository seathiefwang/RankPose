import torch.nn as nn
import math
import torch

class Wingloss(nn.Module):
    def __init__(self, w=2., epsilon=2.):
        super().__init__()
        # self.criterion = nn.L1Loss(reduction='none')
        self.w = w
        self.epsilon = epsilon
        self.c = self.w * (1.0 - math.log(1. + self.w/self.epsilon))

    def forward(self, preds, targets):
        loss = torch.mean(torch.abs(preds - targets), 1)

        absolute_x = torch.abs(loss)
        losses = torch.where(
            self.w > absolute_x,
            self.w * torch.log(1. + absolute_x/self.epsilon),
            absolute_x - self.c
            )

        # losses = torch.sum(losses) / losses.size()[0]
        losses = torch.mean(losses)
        return losses
