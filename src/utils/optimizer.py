import torch.optim as optim
import adabound
from .scheduler import CosineWithRestarts
from torch.optim.lr_scheduler import StepLR


def create_optimizer(params, mode='adam', base_lr=1e-3, t_max=10, final_lr=0.1):
    if mode == 'adam':
        optimizer = optim.Adam(params, base_lr)
    elif mode == 'sgd':
        optimizer = optim.SGD(params, base_lr, momentum=0.9, weight_decay=4e-5)
    elif mode == 'adabound':
        optimizer = adabound.AdaBound(params, lr=base_lr, final_lr=final_lr)
    else:
        raise NotImplementedError(mode)

    scheduler = CosineWithRestarts(optimizer, t_max)

    return optimizer, scheduler
