import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import cv2
import sys
import argparse
import numpy as np
from shutil import copyfile

from models import load_model
from torch.utils.data import DataLoader
from dataset import laod_dataset
from utils.metrics import calculate_diff

from utils.preprocess import preprocess
from utils.functional import quat2euler

# 打印两位小数
np.set_printoptions(precision=2)

target_size = 224
batch_size = 16
n_class = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def error_ploter(diff_list, path):
    title = "Error Distribution"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(diff_list)) * 2
    x_name = [i for i in range(0, 31, 5)]

    y = diff_list[:, 0]
    ax.plot(x, y, label='Yaw')
    y = diff_list[:, 1]
    ax.plot(x, y, label='Pitch')
    y = diff_list[:, 2]
    ax.plot(x, y, label='Roll')
    ax.legend()

    ax.set_title(title)
    plt.xlabel('Fraction of the num')
    plt.ylabel('Pose estimation error')
    plt.ylim(0,1)
    plt.xlim(0,30)
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid(True, linestyle='--', linewidth=0.5) # 打开网格
    plt.savefig(str(path))
    plt.close()

def error_bar_ploter(x_name, diff_list, path):
    title = "Error Distribution"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = list(range(len(diff_list)))  

    total_width, n = 0.8, 3
    width = total_width / n 

    y = diff_list[:, 0]
    ax.bar(x, y, width=width, label='Yaw',fc = 'y')
    for i in range(len(x)):  
        x[i] = x[i] + width  
    y = diff_list[:, 1]
    ax.bar(x, y, width=width, label='Pitch',fc = 'g')   
    for i in range(len(x)):  
        x[i] = x[i] + width  
    y = diff_list[:, 2]
    ax.bar(x, y, width=width, label='Roll',fc = 'r',tick_label = x_name)
    ax.legend()

    ax.set_title(title)
    plt.savefig(str(path))
    plt.close()


def main(args):
    model = load_model(net_type=args.net_type, n_class=n_class)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.modelpath))
    model.eval()

    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    pos_dir = os.path.join(args.dst, 'positive')
    neg_dir = os.path.join(args.dst, 'negative')
    if not os.path.exists(pos_dir):
        os.mkdir(pos_dir)
    if not os.path.exists(neg_dir):
        os.mkdir(neg_dir)

    dataset = laod_dataset(data_type=args.data_type, base_dir=args.base_dir, filename=args.filename, target_size=target_size, n_class=n_class, split='val')
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    total_size = len(dataset)
    # print(total_size)
    diff_list = []

    thre_list = [_ for _ in range(0, 31, 2)]
    lager_diff = [[0, 0, 0] for _ in range(len(thre_list))]


    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels_np = labels.detach().numpy()
        labels = labels.to(device)
        
        preds = model(images)
        preds_np = preds.detach().cpu().numpy()

        diff_total = calculate_diff(preds, labels, mean=False)
        for j in range(len(images)):
            diff = diff_total[j]
            diff_list.append(diff)

            if n_class == 4:
                ypr_lbl = quat2euler(*labels_np[j])
                ypr_pre = quat2euler(*preds_np[j])
            else:
                ypr_lbl = labels_np[j]
                ypr_pre = preds_np[j]

            for m in range(3):
                for n, thre in enumerate(thre_list):
                    if diff[m] < thre:
                        lager_diff[n][m] += 1
                        print(dataset.img_paths[i*batch_size+j], ypr_lbl, ypr_pre, diff)
                        img = cv2.imread(os.path.join(args.base_dir, dataset.img_paths[i*batch_size+j]))
                        bx = dataset.bbox[i*batch_size+j]
                        img = img[int(bx[1]):int(bx[3]), int(bx[0]):int(bx[2])]
                        cv2.imwrite(os.path.join(neg_dir, os.path.basename(dataset.img_paths[i])), img)
                        copyfile(os.path.join(args.base_dir, dataset.img_paths[i*batch_size+j]), os.path.join(neg_dir, os.path.basename(dataset.img_paths[i])))
                        break
            
            print("[ %d/%d ]" % (i*batch_size + j + 1, total_size), end='\r')
            if args.maxsize > 0 and (i*batch_size + j + 1) >= args.maxsize:
                break
    print("\n[ INFO ] TEST end")
    diff_list = np.array(diff_list)
    lager_diff = np.array(lager_diff)

    error_ploter(lager_diff / total_size, os.path.join(args.dst, args.modelpath.split('/')[-2] + "_error_show.jpg"))
    # error_bar_ploter(thre_list, diff_list, os.path.join(args.dst, args.modelpath.split('/')[-2] + "_error_show.jpg"))
    
    print(lager_diff)
    print(np.mean(diff_list, axis=0), np.mean(diff_list))

def parseArg(args):
    parser = argparse.ArgumentParser(description='test data')
    parser.add_argument('--dst', dest='dst', type=str, default='tmp', help='save file.')
    parser.add_argument('--base_dir', dest='base_dir', type=str, default='../data/', help='save file.')
    parser.add_argument('--filename', dest='filename', type=str, default='data/aflw2000_filename.txt', help='')
    parser.add_argument('--data_type', dest='data_type', type=str, default='AFLW2000', help='')
    parser.add_argument('--net_type', dest='net_type', type=str, default='ResNet', help='')
    parser.add_argument('--modelpath', dest='modelpath', type=str, default='model/headpose_resnet/model.pth')
    parser.add_argument('--maxsize', dest='maxsize', type=int, default=0)

    return parser.parse_args(args)

if __name__=='__main__':
    args = parseArg(sys.argv[1:])
    main(args)