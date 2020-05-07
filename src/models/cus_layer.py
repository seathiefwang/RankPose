import torch
import math


def torch_euler2quat(input_):
    input_ = input_ * math.pi / 180
    x = torch.sin(input_[:,1] / 2) * torch.sin(input_[:,0] / 2) * torch.cos(input_[:,2] / 2) + \
        torch.cos(input_[:,1] / 2) * torch.cos(input_[:,0] / 2) * torch.sin(input_[:,2] / 2)
    x = torch.unsqueeze(x, 1)
    y = torch.sin(input_[:,1] / 2) * torch.cos(input_[:,0] / 2) * torch.cos(input_[:,2] / 2) + \
        torch.cos(input_[:,1] / 2) * torch.sin(input_[:,0] / 2) * torch.sin(input_[:,2] / 2)
    y = torch.unsqueeze(y, 1)
    z = torch.cos(input_[:,1] / 2) * torch.sin(input_[:,0] / 2) * torch.cos(input_[:,2] / 2) + \
        torch.sin(input_[:,1] / 2) * torch.cos(input_[:,0] / 2) * torch.sin(input_[:,2] / 2)
    z = torch.unsqueeze(z, 1)
    w = torch.cos(input_[:,1] / 2) * torch.cos(input_[:,0] / 2) * torch.cos(input_[:,2] / 2) + \
        torch.sin(input_[:,1] / 2) * torch.sin(input_[:,0] / 2) * torch.sin(input_[:,2] / 2)
    w = torch.unsqueeze(w, 1)
    output = torch.cat((w, x, y, z), 1)
    return output

def torch_quat2euler(input_):
    roll = torch.atan2(2 * (torch.mul(input_[:,0], input_[:,1]) + torch.mul(input_[:,2], input_[:,3])), \
            1 - 2*(torch.pow(input_[:,1], 2) + torch.pow(input_[:,2], 2)))
    roll = torch.unsqueeze(roll, 1) * 180 / math.pi

    pitch = torch.asin(2 * (torch.mul(input_[:,0], input_[:,2]) - torch.mul(input_[:,3], input_[:,1])))
    roll = torch.unsqueeze(roll, 1) * 180 / math.pi

    yaw = torch.atan2(2 * (torch.mul(input_[:,0], input_[:,3]) + torch.mul(input_[:,2], input_[:,1])), \
            1 - 2*(torch.pow(input_[:,3], 2) + torch.pow(input_[:,2], 2)))
    roll = torch.unsqueeze(roll, 1) * 180 / math.pi
    
    output = torch.cat((yaw, pitch, roll), 1)
    return output

def l2_norm(input_, axis=1):
    norm = torch.norm(input_, 2, axis, True)
    return torch.div(input_, norm)
