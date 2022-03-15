"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def to_numpy(data):
    """
    Convert PyTorch tensor to numpy array. For complex tensor with two channels, the complex numpy arrays are used.

    Args:
        data (torch.Tensor): Input torch tensor

    Returns:
        np.array numpy arrays
    """
    if data.shape[-1] == 2:
        out = np.zeros(data.shape[:-1], dtype=np.complex64)
        real = data[..., 0].numpy()
        imag = data[..., 1].numpy()

        out.real = real
        out.imag = imag
    else:
        out = data.numpy()
    return out


def apply_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    return data * mask, mask


def fft2(data, normalized=True):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=normalized)
    data = fftshift(data, dim=(-3, -2))
    return data


def rfft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    data = ifftshift(data, dim=(-2, -1))
    data = torch.rfft(data, 2, normalized=True, onesided=False)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data, normalized=True):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=normalized)
    data = fftshift(data, dim=(-3, -2))
    return data


def irfft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    data = ifftshift(data, dim=(-3, -2))
    data = torch.irfft(data, 2, normalized=True, onesided=False)
    data = fftshift(data, dim=(-2, -1))
    return data


def complex_to_mag_phase(data):
    """
    :param data (torch.Tensor): A complex valued tensor, where the size of the third last dimension should be 2
    :return: Mag and Phase (torch.Tensor): tensor of same size as input
    """

    assert data.size(-3) == 2
    mag = (data ** 2).sum(dim=-3).sqrt()
    phase = torch.atan2(data[:, 1, :, :], data[:, 0, :, :])
    return torch.stack((mag, phase), dim=-3)


def mag_phase_to_complex(data):
    """
    :param data (torch.Tensor): Mag and Phase (torch.Tensor):
    :return: A complex valued tensor, where the size of the third last dimension is 2
    """

    assert data.size(-3) == 2
    real = data[:, 0, :, :] * torch.cos(data[:, 1, :, :])
    imag = data[:, 0, :, :] * torch.sin(data[:, 1, :, :])
    return torch.stack((real, imag), dim=-3)


def partial_fourier(data):
    """
    :param data:
    :return:
    """



def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2 or data.size(-3) == 2
    return (data ** 2).sum(dim=-1).sqrt() if data.size(-1) == 2 else (data ** 2).sum(dim=-3).sqrt()


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


def normalize_volume(data, mean, std, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are provided and computed from volume.

        Args:
            data (torch.Tensor): Input data to be normalized
            mean: mean of whole volume
            std: std of whole volume
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    return normalize(data, mean, std, eps), mean, std


def normalize_complex(data, eps=0.):
    """
        Normalize the given complex tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from magnitude of data.

        Note that data is centered by complex mean so that the result centered data have average zero magnitude.

        Args:
            data (torch.Tensor): Input data to be normalized (*, 2)
            mean: mean of image magnitude
            std: std of image magnitude
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized complex tensor with 2 channels (*, 2)
        """
    mag = complex_abs(data)
    mag_mean = mag.mean()
    mag_std = mag.std()

    temp = mag_mean/mag

    mean_real = data[..., 0] * temp
    mean_imag = data[..., 1] * temp

    mean_complex = torch.stack((mean_real, mean_imag), dim=-1)

    stddev = mag_std

    return (data - mean_complex) / (stddev + eps), mag_mean, stddev


# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)
