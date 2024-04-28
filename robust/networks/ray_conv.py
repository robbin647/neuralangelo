import pdb
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import tensor as Tensor

class RayConvModule(nn.Module):
    def __init__(self, input_size): # 获取input_size = list(x.shape[-4:])
       super().__init__()
       self.input_size = input_size #[N_SRC, kernel_x, kernel_y, 3] 
       self.n_src, self.kernel_x, self.kernel_y, _ = self.input_size 
       
       self.input_feat_dim = [self.n_src, self.kernel_x*self.kernel_y]
       self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=tuple(self.input_feat_dim), stride=1, padding=0),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(64, 3, kernel_size=(1,1), stride=1, padding=0),
                                  nn.LeakyReLU(),
                                  nn.Flatten())
       self.color_mlp = nn.Sequential(
                        nn.Linear(6, 256),
                        nn.ReLU(),
                        nn.Linear(256, 3)
                        )
        
    def forward(self, input_rgb, neighbor_rgbs)-> Tensor:
        """
        :param input_rgb: (Tensor) the neuralangelo RGB input [B, R, S, 3] 
        :param neighbor_rgbs: (Tensor) [B, R, S, N_SRC, k, k, 3] 
        :return pred_rgb: (Tensor) [B, R, S, 3]
        """
        assert list(neighbor_rgbs.shape[-4:]) == self.input_size[-4:], "Error: neighbor_rgbs shape did not match input_size"
        n_batch, n_rays, n_samples, _ = input_rgb.shape
        input = neighbor_rgbs.permute(0,1,2,6,3,4,5).reshape(n_batch, n_rays, n_samples, 3, self.n_src, -1) # [B, R, S, 3, N_SRC, k*k]
        input = input.contiguous().reshape(-1, *input.shape[-3:])
        neighbor_means = self.conv1(input) # [B* R* S, 3]
        neighbor_means = neighbor_means.reshape(n_batch, n_rays, n_samples, 3) # [B, R, S, 3]

        x = self.color_mlp(torch.cat([input_rgb, neighbor_means], dim=-1)) # [B, R, S, 3] 
        return x


if __name__ == '__main__':
    input_rgb = torch.ones(2, 512, 64, 3)
    neighbor_rgbs = torch.randn(2, 512, 64, 4, 3, 3, 3)*1. 
    pdb.set_trace()
    rcm = RayConvModule(list(neighbor_rgbs.shape[-4:]))
    pred_rgb = rcm(input_rgb, neighbor_rgbs)
    print(pred_rgb.shape)
    