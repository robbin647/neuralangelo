import pdb
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import tensor as Tensor
from projects.nerf.utils.nerf_util import MLPwithSkipConnection

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
        self.add_noise_mlp = nn.Sequential(MLPwithSkipConnection(layer_dims=[131,256,256,256,256,3],
                                                   skip_connection=[1,3]),
                                            nn.Sigmoid())
        # self state
        with torch.no_grad():
            # sd = self.state_dict()
            # sd['latent_dict'] = None
            # self.load_state_dict(sd)
            self.register_parameter("latent_dict", nn.parameter.Parameter(torch.ones(8, 128)))
    
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Learn noise mean and variance 
        :param input: (Tensor) [B, 3, H, W]
        :return: (Tensor) [B, 128] learned latent state 
        """
        x = input
        for enc_block in self.encoder:
            x =  enc_block(x)
        x = x.flatten(start_dim=-3) # [B, 128, 1, 1] => [B, 128]
        assert x.shape[-1] == 128, "Error, encoded feature does not match [..., 128]"
        # self.latent_dict.data = nn.parameter.Parameter(x, requires_grad=True)
        with torch.no_grad():
            self.latent_dict.data = nn.parameter.Parameter(x)
            # sd = self.state_dict()
            # sd['latent_dict'].fill_(x)
        return x
    
    """Override"""
    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.state_dict()['latent_dict'] = state_dict['latent_dict']
        return super().load_state_dict(state_dict, strict, assign)
    
    def add_noise(self, input_rgb: Tensor, 
                  latent: Tensor,
                  neighbor_mean: Tensor = None,
                  neighbor_variance: Tensor = None,
                  is_inference= False) -> Tensor:
        """
        Using learned noise mean and variance to create noise and add to input_rgb   
        :param input_rgb: (Tensor) [B, R, S, 3]
        :param latent: (Tensor) [B, 128] the latent_dict returned by `encode`
        :param neighbor_mean: (None | Tensor) [B, R, S, 3]
        :param neighbor_variance: (None | Tensor) [B, R, S, 3] 
        :param is_inference: (bool) controls if `latent` should be loaded from state dict
        :return: (Tensor) [B, R, S, 3]
        """
        if is_inference: # during inference, load from state dict
            latent = torch.tensor(self.state_dict()['latent_dict'])
        assert latent.dim() == 2 , "Error. self.latent_state used before created"
        B, R, S, _ = input_rgb.shape
        
        _l = latent.view(B, -1, 128) # [B, N_SRC, 128]
        _l = torch.mean(_l, dim=1) # [B, 128]
        _l = _l[:,None, None,:].tile(1,R,S,1) # [B, R, S, 128]
        latent_rgb_vec = torch.cat([_l, input_rgb], dim=-1)
        pred_rgb = self.add_noise_mlp(latent_rgb_vec)
        return pred_rgb
        # assert self.shot_sigma is not None, "Error: shot_sigma used in add_nosie() before encode() "
        # assert self.read_sigma is not None, "Error: read_sigma used in add_nosie() before encode() "    
        # B, R, S, _ = input_rgb.shape
        # if neighbor_mean is None or neighbor_variance is None:
        #     _shot_sig = self.shot_sigma[None, None, None, ...].expand(*input_rgb.shape)
        #     _read_sig = self.read_sigma[None, None, None, ...].expand(*input_rgb.shape)
        #     result_rgb = torch.clamp(input_rgb 
        #                              + torch.sqrt(input_rgb) *_shot_sig * torch.randn_like(input_rgb)
        #                              + torch.randn_like(input_rgb) * _read_sig, min=0., max=1.)
    
        # else:
        #     _shot_sig = self.shot_sigma[None, None, None, ...].expand(*input_rgb.shape)
        #     _read_sig = torch.sqrt(neighbor_variance - self.shot_sigma**2 * neighbor_mean)
        #     _shot_noise = torch.sqrt(neighbor_mean) *_shot_sig * torch.randn_like(input_rgb)
        #     _read_noise = _read_sig * torch.randn_like(input_rgb)
        #     result_rgb = torch.clamp(input_rgb + _shot_noise + _read_noise, min=0., max=1.)
    

if __name__ == '__main__':

    
    vae = NoiseVAE(3, 128, [64, 128, 128,])
    pdb.set_trace()
    encode_input = torch.ones(8, 3, 400, 400)
    l = vae.encode(encode_input)

    input_rgb = torch.ones(2, 512, 64, 3) * 0.5 # [B, R, S , 3]
    neighbor_mean = torch.ones(2, 512, 64, 3) * 0.5
    neighbor_variance = torch.ones(2, 512, 64, 3)
    vae.shot_sigma = torch.zeros(3) * 0.5
    vae.read_sigma = torch.ones(3) * 0.5
    result_rgb = vae.add_noise(input_rgb)