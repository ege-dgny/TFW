import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


def dwt_transform(matrix, wavelet='haar', level=1):
    wavelet = wavelet.lower()
    if wavelet in ['haar', 'db4']:
        coeffs = pywt.wavedec2(matrix, wavelet, level=level)
        return coeffs
    elif wavelet == 'shannon':
        return dwt_shannon(matrix)
    else:
        raise ValueError("Unsupported wavelet type. Choose 'haar', 'db4', or 'shannon'.")


def dwt_shannon(matrix):
    M, N = matrix.shape

    F = np.fft.fftshift(np.fft.fft2(matrix))

    # Create frequency grid indices.
    u = np.arange(-M // 2, M // 2)
    v = np.arange(-N // 2, N // 2)
    U, V = np.meshgrid(u, v, indexing='ij')

    cutoff_M = M // 4
    cutoff_N = N // 4

    # Create ideal (rectangular) masks.
    low_mask = (np.abs(U) < cutoff_M) & (np.abs(V) < cutoff_N)
    LH_mask = (np.abs(U) < cutoff_M) & (np.abs(V) >= cutoff_N)
    HL_mask = (np.abs(U) >= cutoff_M) & (np.abs(V) < cutoff_N)
    HH_mask = (np.abs(U) >= cutoff_M) & (np.abs(V) >= cutoff_N)

    LL = np.real(np.fft.ifft2(np.fft.ifftshift(F * low_mask)))
    LH = np.real(np.fft.ifft2(np.fft.ifftshift(F * LH_mask)))
    HL = np.real(np.fft.ifft2(np.fft.ifftshift(F * HL_mask)))
    HH = np.real(np.fft.ifft2(np.fft.ifftshift(F * HH_mask)))

    coeffs = {
        'LL': LL[::2, ::2],
        'LH': LH[::2, ::2],
        'HL': HL[::2, ::2],
        'HH': HH[::2, ::2],
    }
    return coeffs


def idwt_transform(coeffs, wavelet='haar'):
    wavelet = wavelet.lower()
    if wavelet in ['haar', 'db4']:
        return pywt.waverec2(coeffs, wavelet)
    elif wavelet == 'shannon':
        raise NotImplementedError("Inverse transform for the Shannon wavelet is not implemented.")
    else:
        raise ValueError("Unsupported wavelet type. Choose 'haar', 'db4', or 'shannon'.")


import torch.nn.functional as F


def haar_wavelet_transform(x):
    """
    Performs a single-level 2D Haar wavelet transform on a batch of images and returns
    the coefficients combined into a single matrix.

    Input:
      x: Tensor of shape (B, 1, H, W)
    Output:
      combined: Tensor of shape (B, 1, H, W) where the 2x2 sub-bands [LL, LH, HL, HH]
                are rearranged into a single matrix.
    """
    device = x.device
    dtype = x.dtype
    # Define the 2x2 Haar filters with normalization 1/2
    ll = torch.tensor([[1, 1],
                       [1, 1]], dtype=dtype, device=device) / 2.0
    lh = torch.tensor([[1, 1],
                       [-1, -1]], dtype=dtype, device=device) / 2.0
    hl = torch.tensor([[1, -1],
                       [1, -1]], dtype=dtype, device=device) / 2.0
    hh = torch.tensor([[1, -1],
                       [-1, 1]], dtype=dtype, device=device) / 2.0

    # Stack filters into one weight tensor of shape (4, 1, 2, 2)
    kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
    # Apply convolution with stride 2 to both downsample and extract coefficients
    coeffs = F.conv2d(x, kernel, stride=2)  # shape: (B, 4, H//2, W//2)

    # Rearrange the coefficients into a single channel image.
    # pixel_shuffle takes a tensor of shape (B, C*r^2, H, W) and rearranges it to (B, C, H*r, W*r).
    # Here, with C=1 and r=2, the 4 channels are rearranged into (B, 1, H, W).
    combined = F.pixel_shuffle(coeffs, upscale_factor=2)
    return combined