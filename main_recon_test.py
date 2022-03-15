#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import os
from utils.options import args_parser
from models.evaluation import test_recon_save
from data.mri_data import SliceData, DataTransform
from data.subsample import create_mask_for_mask_type
from models.Recurrent_Transformer import ReconFormer
import pathlib
from torch.utils.data import DataLoader

if __name__ == '__main__':
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    # parse args
    args, parser = args_parser()
    path_dict = {'F': pathlib.Path(args.F_path)}
    resolution_dict = {'F': 320}
    rate_dict = {'F': 1.0}
    args.device = torch.device('cuda:{}'.format(args.gpu[0]) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.resolution = resolution_dict[args.test_dataset]

    # data loader
    def _create_dataset(data_path,data_transform, data_partition, sequence, bs, shuffle, sample_rate=None, display=False):
        sample_rate = sample_rate or args.sample_rate
        dataset = SliceData(
            root=data_path / data_partition,
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=args.challenge,
            sequence=sequence
        )
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=False, num_workers=8)


    # load dataset and split users
    if args.challenge == 'singlecoil':
        mask = create_mask_for_mask_type(args.mask_type, args.center_fractions,
                                         args.accelerations)
        val_data_transform = DataTransform(args.resolution, args.challenge, mask, use_seed=True)

        if args.phase == 'test':
            dataset_val = _create_dataset(path_dict[args.test_dataset]/args.sequence,val_data_transform, 'val', args.sequence, 8, False, 1.0)
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'ReconFormer':
        net = ReconFormer(in_channels=2, out_channels=2, num_ch=(96, 48, 24),num_iter=5,
        down_scales=(2,1,1.5), img_size=args.resolution, num_heads=(6,6,6), depths=(2,1,1),
        window_sizes=(8,8,8), mlp_ratio=2., resi_connection ='1conv',
        use_checkpoint=(False, False, False, False, False, False)
        ).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net)

    # copy weights
    if len(args.gpu) > 1:
        net = torch.nn.DataParallel(net, args.gpu)

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.phase == 'test':
        if len(args.gpu) > 1:
            net.module.load_state_dict(torch.load(args.checkpoint))
        else:
            net.load_state_dict(torch.load(args.checkpoint))
        print('Load checkpoint :', args.checkpoint)
        test_recon_save(net, dataset_val, args)




