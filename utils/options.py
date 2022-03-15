#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import pathlib

def args_parser():
    parser = argparse.ArgumentParser()
    # basic arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--bs', type=int, default=16, help="batch size")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--lr-step-size', type=int, default=50, help="SGD momentum (default: 0.5)")
    parser.add_argument('--lr-gamma', type=float, default=0.1, help="SGD momentum (default: 0.5)")
    # data arguments
    parser.add_argument('--F_path', type=str, default='', help='Path to the dataset')
    parser.add_argument('--val_sample_rate', type=float, default=1.0, help='Path to the dataset')
    parser.add_argument('--train_dataset', type=str, default='', help='Path to the dataset')
    parser.add_argument('--test_dataset', type=str, default='', help='Path to the dataset')
    parser.add_argument('--sequence', type=str, default='T1', help='Path to the dataset')
    parser.add_argument('--save_dir', type=pathlib.Path, help='Path to the save dir')
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--challenge', choices=['singlecoil', 'multicoil'], help='Which challenge')
    parser.add_argument('--sample-rate', type=float, default=1.,help='Fraction of total volumes to include')
    # Mask parameters
    parser.add_argument('--mask-type', choices=['random', 'equispaced'], default='random',help='The type of mask function to use')
    parser.add_argument('--accelerations', nargs='+', default=[4], type=int,
                      help='Ratio of k-space columns to be sampled. If multiple values are '
                           'provided, then one of those is chosen uniformly at random for '
                           'each volume.')
    parser.add_argument('--center-fractions', nargs='+', default=[0.08], type=float,
                      help='Fraction of low-frequency k-space columns to be sampled. Should '
                           'have the same length as accelerations')
    # model arguments
    parser.add_argument('--model', type=str, default='', help='model name')
    # other arguments
    parser.add_argument('--phase', type=str, default='train', help="name of phase")
    parser.add_argument('--checkpoint', type=str, default='', help="name of ck")
    parser.add_argument('--gpu', nargs='+', default=[0,1],type=int, help="GPU ID, -1 for CPU")
    parser.add_argument('--continues', action='store_true', help='continues training')
    parser.add_argument('--return_h', action='store_true', help='return hidden state/ feature maps for plot')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    args = parser.parse_args()
    return args, parser
