# -*- coding: utf-8
import yaml
import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from losses import Criterion
from models import load_model
import albumentations as albu
from dataset import laod_dataset
from collections import OrderedDict
from logger.log import debug_logger
from logger.plot import history_ploter
from torch.utils.data import DataLoader
from utils.metrics import calculate_diff
from utils.optimizer import create_optimizer
from torch.utils.data.sampler import  WeightedRandomSampler

seed = 2020
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed) #both CPU and CUDA

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()

def main():
    config_path = Path(args.config_path)
    config = yaml.safe_load(open(config_path))

    net_config = config['Net']
    data_config = config['Data']
    train_config = config['Train']
    loss_config = config['Loss']
    opt_config = config['Optimizer']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_class = net_config['n_class']
    max_epoch = train_config['max_epoch']
    batch_size = train_config['batch_size']
    num_workers = train_config['num_workers']
    test_every = train_config['test_every']
    resume = train_config['resume']
    pretrained_path = train_config['pretrained_path']
    use_rank = train_config['use_rank']
    use_bined = train_config['use_bined']

    train_dir = data_config['train_dir']
    val_dir = data_config['val_dir']
    train_name = data_config['train_name']
    val_name = data_config['val_name']
    train_type = data_config['train_type']
    val_type = data_config['val_type']

    model = load_model(**net_config)

    # To device
    model = model.to(device)

    modelname = config_path.stem
    output_dir = Path(data_config['model_save_path']) / modelname
    output_dir.mkdir(exist_ok=True)
    log_dir = Path(data_config['logs_path']) / modelname
    log_dir.mkdir(exist_ok=True)

    logger = debug_logger(log_dir)
    logger.debug(config)
    logger.info(f'Device: {device}')
    logger.info(f'Max Epoch: {max_epoch}')

    loss_fn = Criterion(**loss_config).to(device)
    params = model.parameters()
    optimizer, scheduler = create_optimizer(params, **opt_config)

    # history
    if resume:
        with open(log_dir.joinpath('history.pkl'), 'rb') as f:
            history_dict = pickle.load(f)
            best_metrics = history_dict['best_metrics']
            loss_history = history_dict['loss']
            diff_history = history_dict['diff']
            start_epoch = len(diff_history)
            for _ in range(start_epoch):
                scheduler.step()
    else:
        start_epoch = 0
        best_metrics = float('inf')
        loss_history = []
        diff_history = []

    # Dataset
    affine_augmenter = albu.Compose([albu.GaussNoise(var_limit=(0,25),p=0.05),
                                    albu.GaussianBlur(3, p=0.01),
                                    albu.JpegCompression(50, 100, p=0.01)])

    image_augmenter = albu.Compose([
                                    albu.OneOf([
                                        albu.RandomBrightnessContrast(0.25,0.25),
                                        # albu.CLAHE(clip_limit=2),
                                        # albu.RandomGamma(),
                                        ], p=0.2),
                                    albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,p=0.05),
                                    albu.RGBShift(p=0.01),
                                    albu.ToGray(p=0.01)
                                    ])
    # image_augmenter = None
    train_dataset = laod_dataset(data_type=train_type, affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                            base_dir=train_dir, filename=train_name, use_bined=use_bined, n_class=n_class, **data_config)

    valid_dataset = laod_dataset(data_type=val_type, split='valid', base_dir=val_dir, filename=val_name, 
                            use_bined=use_bined, n_class=n_class, **data_config)

    # top_10 = len(train_dataset) // 10
    # top_30 = len(train_dataset) // 3.33
    # train_weights = [ 3 if idx<top_10 else 2 if idx<top_30 else 1 for idx in train_dataset.labels_sort_idx]
    # train_sample = WeightedRandomSampler(train_weights, num_samples=len(train_dataset), replacement=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sample, num_workers=num_workers,
    #                           pin_memory=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)


    # Pretrained model
    if pretrained_path:
        logger.info(f'Resume from {pretrained_path}')
        param = torch.load(pretrained_path, map_location='cpu')
        if "state_dict" in param:
            model.load_state_dict(param['state_dict'], strict=False)
        else:
            model.load_state_dict(param)
        del param

    # Restore model
    if resume:
        model_path = output_dir.joinpath(f'model_tmp.pth')
        logger.info(f'Resume from {model_path}')
        param = torch.load(model_path, map_location='cpu')
        model.load_state_dict(param)
        del param
        opt_path = output_dir.joinpath(f'opt_tmp.pth')
        param = torch.load(opt_path)
        optimizer.load_state_dict(param)
        del param

    if torch.cuda.is_available():
        model = nn.DataParallel(model)


    # Train
    for i_epoch in range(start_epoch, max_epoch):
        logger.info(f'Epoch: {i_epoch}')
        logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')

        train_losses = []
        train_diffs = []
        model.train()
        with tqdm(train_loader) as _tqdm:
            for batched in _tqdm:
                optimizer.zero_grad()

                if use_rank:
                    if use_bined:
                        img1, img2, lbl1, lbl2, labels, yaw_lbl1, pitch_lbl1, roll_lbl1, yaw_lbl2, pitch_lbl2, roll_lbl2 = batched
                        # print(lbl1, lbl2)
                        # print( yaw_lbl1, pitch_lbl1, roll_lbl1, yaw_lbl2, pitch_lbl2, roll_lbl2)
                        img1, img2, lbl1, lbl2, labels = img1.to(device),img2.to(device),lbl1.to(device),lbl2.to(device),labels.to(device)
                        yaw_lbl1, pitch_lbl1, roll_lbl1 = yaw_lbl1.to(device), pitch_lbl1.to(device), roll_lbl1.to(device)
                        yaw_lbl2, pitch_lbl2, roll_lbl2 = yaw_lbl2.to(device), pitch_lbl2.to(device), roll_lbl2.to(device)
                        
                        preds1, y_pres1, p_pres1, r_pres1 = model(img1, True)
                        preds2, y_pres2, p_pres2, r_pres2 = model(img2, True)
                        
                        pre_list = [preds1,preds2,y_pres1,p_pres1,r_pres1,y_pres2,p_pres2,r_pres2]
                        lbl_list = [lbl1,lbl2,yaw_lbl1,pitch_lbl1,roll_lbl1,yaw_lbl2,pitch_lbl2,roll_lbl2,labels]
                        loss = loss_fn(pre_list, lbl_list, use_bined=True)
                    else:
                        img1, img2, lbl1, lbl2, labels = batched
                        # print(lbl1, lbl2, labels)
                        img1, img2, lbl1, lbl2, labels = img1.to(device),img2.to(device),lbl1.to(device),lbl2.to(device),labels.to(device)

                        preds1 = model(img1, False)
                        preds2 = model(img2, False)
                        # print(preds1)
                        loss = loss_fn([preds1,preds2], [lbl1,lbl2,labels], use_bined=False)

                    diff = calculate_diff(preds1, lbl1)
                    diff += calculate_diff(preds2, lbl2)
                    diff /= 2
                elif use_bined:
                    images, labels, yaw_labels, pitch_labels, roll_labels = batched
                
                    images, labels = images.to(device), labels.to(device)
                    yaw_labels, pitch_labels, roll_labels = yaw_labels.to(device), pitch_labels.to(device), roll_labels.to(device)

                    preds, y_pres, p_pres, r_pres = model(images, use_bined)
                
                    loss = loss_fn([preds, y_pres, p_pres, r_pres], [labels, yaw_labels, pitch_labels, roll_labels], use_bined)

                    diff = calculate_diff(preds, labels)
                else:
                    images, labels = batched
                
                    images, labels = images.to(device), labels.to(device)

                    preds = model(images, use_bined)
                
                    loss = loss_fn([preds], [labels])

                    diff = calculate_diff(preds, labels, mean=True)

                _tqdm.set_postfix(OrderedDict(loss=f'{loss.item():.3f}', mae=f'{diff:.1f}'))
                train_losses.append(loss.item())
                train_diffs.append(diff)

                loss.backward()
                optimizer.step()

        scheduler.step()

        train_loss = np.mean(train_losses)
        train_diff = np.nanmean(train_diffs)
        logger.info(f'train loss: {train_loss}')
        logger.info(f'train diff: {train_diff}')

        torch.save(model.module.state_dict(), output_dir.joinpath('model_tmp.pth'))
        torch.save(optimizer.state_dict(), output_dir.joinpath('opt_tmp.pth'))

        if (i_epoch + 1) % test_every == 0:
            valid_losses = []
            valid_diffs = []
            model.eval()
            with torch.no_grad():
                with tqdm(valid_loader) as _tqdm:
                    for batched in _tqdm:
                        if use_bined:
                            images, labels, yaw_labels, pitch_labels, roll_labels = batched
                        
                            images, labels = images.to(device), labels.to(device)
                            # yaw_labels, pitch_labels, roll_labels = yaw_labels.to(device), pitch_labels.to(device), roll_labels.to(device)

                            preds, y_pres, p_pres, r_pres = model(images, use_bined)
                        
                            # loss = loss_fn([preds, y_pres, p_pres, r_pres], [labels, yaw_labels, pitch_labels, roll_labels])

                            diff = calculate_diff(preds, labels)
                        else:
                            images, labels = batched
                        
                            images, labels = images.to(device), labels.to(device)

                            preds = model(images, use_bined)
                        
                            # loss = loss_fn([preds], [labels])

                            diff = calculate_diff(preds, labels)
                        
                        _tqdm.set_postfix(OrderedDict(loss=f'{loss.item():.3f}', mae=f'{diff:.2f}'))
                        # _tqdm.set_postfix(OrderedDict(loss=f'{loss.item():.3f}', d_y=f'{np.mean(diff[:,0]):.1f}', d_p=f'{np.mean(diff[:,1]):.1f}', d_r=f'{np.mean(diff[:,2]):.1f}'))
                        valid_losses.append(0)
                        valid_diffs.append(diff)

            valid_loss = np.mean(valid_losses)
            valid_diff = np.mean(valid_diffs)
            logger.info(f'valid seg loss: {valid_loss}')
            logger.info(f'valid diff: {valid_diff}')

            if best_metrics >= valid_diff:
                best_metrics = valid_diff
                logger.info('Best Model!\n')
                torch.save(model.state_dict(), output_dir.joinpath('model.pth'))
                torch.save(optimizer.state_dict(), output_dir.joinpath('opt.pth'))
        else:
            valid_loss = None
            valid_diff = None

        loss_history.append([train_loss, valid_loss])
        diff_history.append([train_diff, valid_diff])
        history_ploter(loss_history, log_dir.joinpath('loss.png'))
        history_ploter(diff_history, log_dir.joinpath('diff.png'))

        history_dict = {'loss': loss_history,
                        'diff': diff_history,
                        'best_metrics': best_metrics}
        with open(log_dir.joinpath('history.pkl'), 'wb') as f:
            pickle.dump(history_dict, f)

if __name__=='__main__':
    main()