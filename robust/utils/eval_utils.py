from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from robust.utils.loss_utils import L2Loss

from robust.utils.general_utils import TINY_NUMBER

# SSIM implementation is from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
# Calculate the one-dimensional Gaussian distribution vector


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# Create a Gaussian kernel, obtained by matrix multiplication of two one-dimensional Gaussian distribution vectors
# You can set the channel parameter to expand to 3 channels
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

""" # Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=13, size_average=True, val_range=None, reduce='mean', padd=0):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.padd = padd

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
        self.reduce = reduce

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average,
                    reduce=self.reduce, padd=self.padd)
"""

def mse2psnr(x): return -10. * np.log10(x + TINY_NUMBER)


def img2psnr(x, y, mask=None):
    return mse2psnr(L2Loss().loss_fn(x, y, mask).item())
