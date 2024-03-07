import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable
from functools import partial
from robust.utils.general_utils import TINY_NUMBER
# from robust.utils.eval_utils import ssim_loss
from math import exp

class RGBCriterion(nn.Module):
    name = 'rgb_loss'

    def __init__(self, args):
        super().__init__()
        self.args = args

    def loss_fn(self, pred, gt, mask):
        pass

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """

        pred_rgb  = outputs.rgb
        pred_mask = outputs.mask.float()
        gt_rgb    = ray_batch['rgb']

        if self.args.process_loss:
            pred_rgb = Delinearize().de_linearize(pred_rgb, ray_batch['white_level'])
            gt_rgb   = Delinearize().de_linearize(gt_rgb, ray_batch['white_level'])

        loss = self.loss_fn(pred_rgb, gt_rgb, pred_mask)
        scalars_to_log[self.name] = loss
        return loss

    @staticmethod
    def patch_view(x):
        assert x.shape[-1] in [1, 3]
        assert int(x.shape[0] ** 0.5) ** 2 == x.shape[0]
        crop_size = int(x.shape[0] ** 0.5)
        x = x.reshape((crop_size, crop_size, -1))
        return x

    def mean_with_mask(self, x, mask):
        return torch.sum(x * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)

    def loss_with_mask(self, fn, x, y, mask=None):
        """
        @param x: img 1, [(...), 3]
        @param y: img 2, [(...), 3]
        @param mask: optional, [(...)]
        @param fn: loss function
        """
        loss = fn(x, y)
        if mask is None:
            return torch.mean(loss)
        else:
            return self.mean_with_mask(loss, mask)



class L2Loss(RGBCriterion):
    name = 'rgb_l2'
    
    def l2_loss(self) -> Callable:
        return partial(self.loss_with_mask, lambda x, y: (x - y) ** 2) # return a function definition
    
    def loss_fn(self, pred, gt, mask):
        return self.l2_loss()(pred, gt, mask)


class L1Loss(RGBCriterion):
    name = 'rgb_l1'
    def l1_loss(self) -> Callable:
        return partial(self.loss_with_mask, lambda x, y: torch.abs(x - y)) # return a function definition
    def loss_fn(self, pred, gt, mask):
        return self.l1_loss()(pred, gt, mask)

class L1GradLoss(RGBCriterion):
    '''
    Loss that tests smoothness
    '''
    name  = 'grad_l1'
    alpha = 2

    def gen_loss(self) -> Callable:
        return partial(self.loss_with_mask, lambda x, y: x, 0) # return a function definition
    
    def loss_fn(self, pred, gt, mask):
        # gradient
        pred = self.patch_view(pred)
        gt   = self.patch_view(gt)
        mask = self.patch_view(mask.unsqueeze(-1))[..., 0]

        pred_dx, pred_dy = self.gradient(pred)
        gt_dx,   gt_dy   = self.gradient(gt)
        #
        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)

        # condense into one tensor and avg
        return self.gen_loss()((grad_diff_x ** self.alpha + grad_diff_y ** self.alpha + TINY_NUMBER) ** (1 / self.alpha), mask)

    @staticmethod
    def gradient(x):
        # From https://discuss.pytorch.org/t/how-to-calculate-the-gradient-of-images/1407/6
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (h, w, c), float32 or float64
        # dx, dy: (h, w, c)

        left   = x
        right  = F.pad(x, [0, 0, 0, 1])[:, 1:, :]
        top    = x
        bottom = F.pad(x, [0, 0, 0, 0, 0, 1])[1:, :, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[..., :, -1, :] = 0
        dy[..., -1, :, :] = 0

        return dx, dy

class SSIMLoss(RGBCriterion):
    name = 'ssim'
    
    def loss_fn(self, pred, gt, mask=None):
        return ssim_loss(pred, gt, mask)


def ssim_loss(rgb1, rgb2, mask=None):
    if mask is not None:
        raise NotImplementedError
    crop_size = int(rgb1.shape[-2] ** 0.5)
    img1 = rgb1.reshape((-1, crop_size, crop_size, 3)).permute((0, 3, 1, 2))
    img2 = rgb2.reshape((-1, crop_size, crop_size, 3)).permute((0, 3, 1, 2))

    return 1 - ssim(img1, img2, window_size=11)

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

def ssim(img1, img2, window_size=13, window=None, size_average=True, full=False, val_range=None, reduce='mean', padd=0):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if reduce is None:
        return ssim_map
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean().mean()

    if full:
        return ret, cs
    return ret

loss_mapping = {'l2': L2Loss,
                'l1': L1Loss,
                'l1_grad': L1GradLoss,
                'ssim': SSIMLoss}

class NANLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.losses_list = []
        self.weights_list = []
        for loss_type, weight in zip(args.losses, args.losses_weights):
            if weight == 0:
                continue
            self.losses_list.append(loss_mapping[loss_type](args))
            self.weights_list.append(weight)

    def forward(self, outputs, ray_batch, scalars_to_log):
        return sum([w * loss(outputs, ray_batch, scalars_to_log) for w, loss in zip(self.weights_list, self.losses_list)])

class Delinearize:
    t = 0.0031308
    gamma = 2.4
    a = 1. / (1. / (t ** (1 / gamma) * (1. - (1 / gamma))) - 1.)  # 0.055
    # a = 0.055
    k0 = (1 + a) * (1 / gamma) * t ** ((1 / gamma) - 1.)  # 12.92
    # k0 = 12.92
    inv_t = t * k0

    def de_linearize(self, rgb, wl=1.):
        """
        Process the RGB values in the inverse process of the approximate linearization, in a differential format
        @param rgb:
        @param wl:
        @return:
        """
        rgb = rgb / wl
        srgb = torch.where(rgb > self.t, (1 + self.a) * torch.clamp(rgb, min=self.t) ** (1 / self.gamma) - self.a, self.k0 * rgb)

        k1 = (1 + self.a) * (1 / self.gamma)
        srgb = torch.where(rgb > 1, k1 * rgb - k1 + 1, srgb)
        return srgb
