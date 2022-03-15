"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import numpy as np
import h5py
from torch.utils.data import Dataset
from data import transforms
import torch


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sequence, sample_rate, seed=42):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'

        phase = root.parts[-1]
        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        print('Loading dataset :', root)
        random.seed(seed)
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            data = h5py.File(fname, 'r')
            padding_left = None
            padding_right = None
            kspace = data['kspace']
            num_slices = kspace.shape[0]
            num_start = 0
            self.examples += [(fname, slice, padding_left, padding_right) for slice in range(num_start, num_slices)]
        if phase == 'train' and sample_rate > 1:
            self.paths_for_run = []
            for element in self.examples:
                for i in range(int(sample_rate)):
                    self.paths_for_run.append(element)
            self.examples = self.paths_for_run


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, padding_left, padding_right = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            mask = np.asarray(data['mask']) if 'mask' in data else None
            target = data[self.recons_key][slice] if self.recons_key in data else None
            attrs = dict(data.attrs)
            attrs['padding_left'] = padding_left
            attrs['padding_right'] = padding_right
            return self.transform(kspace, mask, target, attrs, fname.name, slice)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(
                f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            mask (numpy.array): Mask from the test dataset
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        kspace = transforms.to_tensor(kspace)

        # Apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(
                kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)

        target = transforms.ifft2(kspace)
        # Absolute value
        abs_image = transforms.complex_abs(image)
        mean = torch.tensor(0.0)
        std = abs_image.mean()
        # Normalize input
        image = image.permute(2, 0, 1)
        target = target.permute(2, 0, 1)
        image = transforms.normalize(image, mean, std, eps=0)
        masked_kspace = masked_kspace.permute(2, 0, 1)
        masked_kspace = transforms.normalize(masked_kspace, mean, std, eps=0)
        # Normalize target
        target = transforms.normalize(target, mean, std, eps=0)
        mask = mask.repeat(image.shape[1], 1, 1).squeeze().unsqueeze(0)
        return image, target, mean, std, attrs['norm'].astype(np.float32), fname, slice, attrs['max'].astype(np.float32), mask, masked_kspace

