import pdb
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
from torch import tensor as Tensor
# Tensor = TypeVar('torch.tensor')

class NoiseVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs):
        super().__init__()
        self.in_channels = 3
        self.latent_dim = latent_dim
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.in_channels, 64,
                                    kernel_size=8, stride=8, padding=0),
                          nn.BatchNorm2d(64),
                          nn.MaxPool2d(kernel_size=(2,2), stride=2),
                          nn.LeakyReLU()),
            
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3,3), stride=3, padding=0),
                          nn.BatchNorm2d(128),
                          nn.MaxPool2d(kernel_size=(2,2), stride=2),
                          nn.LeakyReLU()),
            
            nn.Sequential(nn.Conv2d(128, 128, kernel_size=(4,4), stride=4, padding=0),
                   nn.BatchNorm2d(128),
                   nn.LeakyReLU())
        ])
        self.latent_layer = nn.Linear(128, 6)
        self.add_noise_mlp = nn.ModuleList()
        # self state
        self.shot_sigma = None
        self.read_sigma = None
    
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Learn noise mean and variance 
        :param input: (Tensor) [B, 3, H, W]
        :return: (Tensor) [B, 3] for shot noise sigma and [B, 3] for read noise sigma
        """
        x = input
        for enc_block in self.encoder:
            x =  enc_block(x)
        assert x.shape[-1] == 128, "Error, encoded feature does not match [..., 128]"
        latent_feat = self.latent_layer(x)
        shot_sigma, read_sigma = torch.split(latent_feat, 2)
        self.shot_sigma = torch.mean(self.shot_sigma, dim=0) #[3]
        self.read_sigma = torch.mean(self.read_sigma, dim=0) #[3]
        return shot_sigma, read_sigma

    def add_noise(self, input_rgb: Tensor, 
                  neighbor_mean: Tensor = None,
                  neighbor_variance: Tensor = None) -> Tensor:
        """
        Using learned noise mean and variance to create noise and add to input_rgb   
        :param input_rgb: (Tensor) [B, R, S, 3]
        :param neighbor_mean: (None | Tensor) [B, R, S, 3]
        :param neighbor_variance: (None | Tensor) [B, R, S, 3] 
        :return: (Tensor) [B, R, S, 3]
        """
        assert self.shot_sigma is not None, "Error: shot_sigma used in add_nosie() before encode() "
        assert self.read_sigma is not None, "Error: read_sigma used in add_nosie() before encode() "    
        B, R, S, _ = input_rgb.shape
        if neighbor_mean is None or neighbor_variance is None:
            _shot_sig = self.shot_sigma[None, None, None, ...].expand(*input_rgb.shape)
            _read_sig = torch.sqrt(neighbor_variance - self.shot_sigma**2 * neighbor_mean)
            _shot_noise = torch.sqrt(neighbor_mean) *_shot_sig * torch.randn_like(input_rgb)
            _read_noise = _read_sig * torch.randn_like(input_rgb)
            result_rgb = torch.clamp(input_rgb + _shot_noise + _read_noise, min=0., max=1.)

        else:
            _shot_sig = self.shot_sigma[None, None, None, ...].expand(*input_rgb.shape)
            _read_sig = self.read_sigma[None, None, None, ...].expand(*input_rgb.shape)
            result_rgb = torch.clamp(input_rgb 
                                     + torch.sqrt(input_rgb) *_shot_sig * torch.randn_like(input_rgb)
                                     + torch.randn_like(input_rgb) * _read_sig, min=0., max=1.)
        return result_rgb
    

if __name__ == '__main__':
    vae = NoiseVAE(3, 128, [64, 128, 128,])
    pdb.set_trace()
    input_rgb = torch.ones(2, 512, 64, 3) * 0.5 # [B, R, S , 3]
    neighbor_mean = torch.ones(2, 512, 64, 3) * 0.5
    neighbor_variance = torch.ones(2, 512, 64, 3)
    vae.shot_sigma = torch.zeros(3) * 0.5
    vae.read_sigma = torch.ones(3) * 0.5
    result_rgb = vae.add_noise(input_rgb, neighbor_mean, neighbor_variance)