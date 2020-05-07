import torch.nn as nn
import torch
from .biwi import BIWIDataset
from .aflw2000 import AFLW2000Dataset
from .ft_300w_lp import Pose300wDataset
from .rank_300w_lp import Rank300wDataset
# from .rank_noid_300w_lp import RankNoIDDataset

def laod_dataset(data_type='AFLW2000', **kwargs):
    if data_type == 'BIWI':
        return BIWIDataset(**kwargs)
    elif data_type == 'AFLW2000':
        return AFLW2000Dataset(**kwargs)
    elif data_type == '300W':
        return Pose300wDataset(**kwargs)
    elif data_type == 'RANK_300W':
        return Rank300wDataset(**kwargs)
    # elif data_type == 'RANK_NOID':
    #     return RankNoIDDataset(**kwargs)
    else:
        raise NotImplementedError
