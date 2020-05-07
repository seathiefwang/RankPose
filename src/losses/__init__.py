import torch
import torch.nn as nn
from .wingloss import Wingloss
from models.cus_layer import torch_quat2euler, torch_euler2quat
from .ranking_l2 import RankingL2loss

class Criterion(nn.Module):
    def __init__(self, loss_type='MSE', **kwargs):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'MSE':
            self.criterion = nn.MSELoss(**kwargs)
        elif loss_type == 'L1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'SMOOTHL1':
            self.criterion = nn.SmoothL1Loss()
        elif loss_type == 'WING':
            self.criterion = Wingloss()
        elif loss_type == 'RANK':
            # self.criterion = Wingloss()
            self.criterion = nn.MSELoss(**kwargs)
            self.rank_criterion = nn.MarginRankingLoss(margin=0., **kwargs)
            # self.rank_criterion = RankingL2loss(margin=0., **kwargs)
        else:
            raise NotImplementedError

        self.class_criterion = nn.CrossEntropyLoss()

    def forward(self, preds, labels, use_bined=False):
        if self.loss_type == 'RANK':
            if use_bined:
                class_loss = self.class_criterion(preds[2], labels[2]) + self.class_criterion(preds[5], labels[5])
                class_loss += self.class_criterion(preds[3], labels[3]) + self.class_criterion(preds[6], labels[6])
                class_loss += self.class_criterion(preds[4], labels[4]) + self.class_criterion(preds[7], labels[7])
            else:
                class_loss = 0
            rank_loss = self.rank_criterion(torch.abs(preds[0][:,0]),torch.abs(preds[1][:,0]),labels[-1][:,0])
            rank_loss += self.rank_criterion(torch.abs(preds[0][:,1]),torch.abs(preds[1][:,1]),labels[-1][:,1])
            rank_loss += self.rank_criterion(torch.abs(preds[0][:,2]),torch.abs(preds[1][:,2]),labels[-1][:,2])
            
            angles_loss = self.criterion(preds[0],labels[0]) + self.criterion(preds[1],labels[1])
            
            return angles_loss + 1 * rank_loss + 0.1 * class_loss
        elif use_bined:
            angles_loss = self.criterion(preds[0],labels[0])

            class_loss = self.class_criterion(preds[1], labels[1])
            class_loss += self.class_criterion(preds[2], labels[2])
            class_loss += self.class_criterion(preds[3], labels[3])

            return angles_loss + 0.1 * class_loss
        else:
            angles_loss = self.criterion(preds[0],labels[0])
            return angles_loss
