import torch.nn as nn

# from torchvision.models import MobileNetV3
# from .ShuffleNetV2 import ShuffleNetV2
# from .MobileNetV1 import MobileNet
# from .MobileNetV2 import MobilenetV2
from .resnet import ResNet
# from .googlenet import GoogleNet
# from .MobileNetV3 import MobileNetV3_Small

# from .resnet_64 import ResNet64
# from .mbv2_64 import MobilenetV264

def load_model(net_type='ShuffleNetV2', **kwargs):
    if net_type == 'ShuffleNetV2':
        pass
        # model = ShuffleNetV2(**kwargs)
    # elif net_type == 'MobileNet':
    #     model = MobileNet(**kwargs)
    # elif net_type == 'MobileNetV2':
    #     model = MobilenetV2(**kwargs)
    # elif net_type == 'MobileNetV3':
    #     model = MobileNetV3_Small(**kwargs)
    elif net_type == 'ResNet':
        model = ResNet(**kwargs)
    # elif net_type == 'GoogleNet':
    #     model = GoogleNet(**kwargs)
    # elif net_type == 'ResNet64':
    #     model = ResNet64(**kwargs)
    # elif net_type == 'MobileNetV264':
    #     model = MobilenetV264(**kwargs)
    else:
        raise NotImplementedError

    return model