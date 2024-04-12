import sys
sys.path.insert(0, '/root/my_code/neuralangelo')

import torch
import torch.nn as nn
import kornia
import math

# import from project root
from robust.configs.config import OUT_DIR
from functools import partial
from typing import Dict, Tuple
from pathlib import Path

# import from current folder
from robust.utils.io_utils import get_latest_file, print_link
from attention import EncoderLayer
from feature_encoder import ResUNet
from robust.networks.vision_transformer import ViTModel
from robust.configs.dotdict_class import DotDict

from projects.neuralangelo.utils.modules import NeuralSDF, NeuralRGB

import pdb

class Gaussian2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], sigma: Tuple[float, float]):
        super().__init__(in_channels, out_channels, kernel_size, padding='same')

        gauss_kernel: torch.Tensor = kornia.filters.get_gaussian_kernel2d(kernel_size, sigma)
        new_weight = torch.zeros_like(self.weight)
        new_weight[0, 0] = gauss_kernel
        new_weight[1, 1] = gauss_kernel
        new_weight[2, 2] = gauss_kernel

        with torch.no_grad():
            self.weight.copy_(new_weight)
        nn.init.zeros_(self.bias.data)


def parallel(model, local_rank):
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

def softmax3d(x, dim):
    R, S, k, _, V, C = x.shape
    return nn.functional.softmax(x.reshape((R, S, -1, C)), dim=-2).view(x.shape)

"""Borrowed from NanMLP  """
def blend_rgb_fn(x, mask, rgb_in):
    """
    x [512, 64, 1, 1, 8, 27] 
    mask [512, 64, 1, 1, 8, 1], 
    rgb_in [512, 64, 3, 3, 8, 3]
    """
    R, S, k, _, V, C = rgb_in.shape
    # masked_fill: Fills elements of self tensor with value where mask is True.
    # squeeze()把所有为1的dim干掉
    w = x.masked_fill((~mask), -1e9).squeeze().view((R, S, V, k, k, C)) # [512, 64, 8, 3, 3, 3]
    w = w.permute((0, 1, 3, 4, 2, 5)) # [512, 64, 3, 3, 8, 3]
    # masked_fill作用是把令mask为False的x最后一维填充负无穷 (这样softmax后会变成接近0)
    blending_weights_valid = softmax3d(w, dim=(2, 3, 4))  # 把w的第2,3,4维合并起来做softmax
    # blending_weights_valid [512, 64, 3, 3, 8, 3]
    rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=(2, 3, 4))
    return rgb_out, blending_weights_valid # [512, 64, 3], [512, 64, 3, 3, 8, 3]

class RobustModel(nn.Module):
    @classmethod
    def create(cls, args):
        model = cls(args)
        if args.distributed:
            model = parallel(model, args.local_rank)
        return model

    def __init__(self, args):
        super().__init__()
        self.args = args
        device = torch.device(f'cuda:{args.local_rank}')

        # Read model configuration parameters
        assert self.args.kernel_size[0] == self.args.kernel_size[1]
        self.k_mid = int(self.args.kernel_size[0] // 2)
        if self.args.anti_alias_pooling:
            self.anti_alias_s = nn.Parameter(torch.tensor(0.2), requires_grad=True)


        # create feature extraction network
        self.feature_net = ResUNet(coarse_out_ch=args.coarse_feat_dim,
                                   fine_out_ch=args.fine_feat_dim,
                                   coarse_only=args.coarse_only).to(device)
        # Gaussian filter to put before feature_net
        if args.pre_net:
            self.pre_net = Gaussian2D(in_channels=3, out_channels=3, kernel_size=(3, 3), sigma=(1.5, 1.5)).to(device)
        else:
            self.pre_net = None

        ViT_RAY_CONFIG = DotDict({
        "num_encoder_block": 1,
        "n_head": 5,    
        "embedding_size": 35,
        "model_k_size": 7,
        "model_v_size": 7,
        "mlp_dim": 256, #  dimension of the intermediate output between last two linear layers 
        "dropout_rate": 0.1
        })
        
        self.ray_transformer = ViTModel(ViT_RAY_CONFIG).to(device)

        ViT_RGB_CONFIG = DotDict({
        "num_encoder_block": 1,
        "n_head": 9,
        "embedding_size": 108,
        "model_k_size": 12,
        "model_v_size": 3,
        "mlp_dim": 256,
        "dropout_rate": 0.1
        })

        self.rgb_transformer = ViTModel(ViT_RGB_CONFIG).to(device)
        self.rgb_transformer_tail_fc = nn.Linear(ViT_RGB_CONFIG.embedding_size, 27).to(device) # 特事特办：因为出来的rgb_trans_out最后一维要是27，才能跟rgb_in做blending
        

        self.sdf_net = nn.Sequential(nn.Linear(280, 257), nn.ReLU(False)).to(device)
        # RGB_NET_CONFIG = DotDict({
        #         "encoding_view": DotDict({
        #             "levels": 3,
        #             "type": "spherical",
        #         }),
        #         "mlp": DotDict({
        #             "activ": "relu_",
        #             "activ_params": {},
        #             "hidden_dim": 256,
        #             "num_layers": 4,
        #             "skip": [],
        #             "weight_norm": True
        #         }),
        #         "mode": "idr"
        # })
        #
        # self.rgb_net = NeuralRGB(cfg_rgb=RGB_NET_CONFIG, 
        #                          feat_dim=RGB_NET_CONFIG.mlp.hidden_dim, 
        #                          appear_embed=DotDict({"dim":8, "enable": False}))
        
        self.mlps = {'coarse': self.ray_transformer, 'fine': None}

        out_folder = OUT_DIR

        # optimizer and learning rate scheduler
        self.optimizer, self.scheduler = self.create_optimizer()

        self.start_step = self.load_from_ckpt(out_folder)

                    
        # self.neural_sdf = NeuralSDF(cfg_model.object.sdf)
        # self.neural_rgb = NeuralRGB(cfg_model.object.rgb, feat_dim=cfg_model.object.sdf.mlp.hidden_dim,
        #                             appear_embed=cfg_model.appear_embed)
        # if cfg_model.background.enabled:
        #     self.background_nerf = BackgroundNeRF(cfg_model.background, appear_embed=cfg_model.appear_embed)
        # else:
        #     self.background_nerf = None
        # self.s_var = torch.nn.Parameter(torch.tensor(cfg_model.object.s_var.init_val, dtype=torch.float32))

    def forward(self, data):
        """
        data = Dict{
          "rgb_feat",
          "mask",
          "ray_diff",
          "sigma_estimate",
          "rgb_in"
        }
        """
        for _v in data.values():
            assert _v.device.type == "cuda"
        rgb_feat = data['rgb_feat'] # [R, S, 3, 3, 8, 32+3=35]
        rgb_feat = rgb_feat[:,:,self.k_mid:self.k_mid+1, self.k_mid:self.k_mid+1] # [R, S, 1, 1, 8, 35] 
        ray_out, weights = self.ray_transformer(rgb_feat) # [R, S, 1, 1, 8, 35], [R, S, 1, 1, 5, 8, 8]
        R, S, _, _, _, _ = ray_out.shape
        ray_out_copy = ray_out.view(R, S, 1, -1) #[512, 64, 1, 280]
        sdf_imp = self.sdf_net(ray_out_copy) # [512, 64, 1, 257]
        # got sdf value as scalar, and encoded 1x256 vector
        sdf_val, sdf_vec = torch.split(sdf_imp, [1, 256], dim=-1) # [512, 64, 1, 1], [512, 64, 1, 256]
        
        mask = data['mask'] # [R, S, 1, 1, 8, 1]
        ray_diff = data['ray_diff'] # [R, S, 1, 1, 8, 4]
        
        weight = self.compute_weights(ray_diff, mask) # [R, S, 1, 1, 8, 1]
        def fused_mean_variance(x, weight):
            mean = torch.sum(x * weight, dim=-2, keepdim=True)
            var = torch.sum(weight * (x - mean) ** 2, dim=-2, keepdim=True)
            return mean, var
        # rgb_feat [R, S, 1, 1, 8, 35]
        mean, var = fused_mean_variance(rgb_feat, weight) # [R, S, 1, 1, 1, 35]
        mean = mean.expand(R, S, 1, 1, 8, 35) # [R, S, 1, 1, 8, 35]
        var = var.expand(R, S, 1, 1, 8, 35) # ibid
        sigma_estimate = data['sigma_estimate'] # [R, S, 3, 3, 8, 3]
        sigma_estimate = sigma_estimate[:, :, self.k_mid:self.k_mid+1, self.k_mid:self.k_mid+1] # [R, S, 1, 1, 8, 3]
        ext_feat = torch.cat([ray_out, mean, var, sigma_estimate], dim=-1) # [R, S, 1, 1, 8, 108]

        rgb_trans_out, _ = self.rgb_transformer(ext_feat) # [R, S, 1, 1, 8, 108]
        rgb_blending_w = self.rgb_transformer_tail_fc(rgb_trans_out) # [R, S, 1, 1, 8, 27]
        rgb_in = data['rgb_in'] # [R, S, 3, 3, 8, 3]
        # mask [R, S, 1, 1, 8, 1]
        rgb_out, blending_weights_rgb = blend_rgb_fn(rgb_blending_w, mask, rgb_in)
        
        return sdf_val, sdf_vec, rgb_out 
    
    def compute_weights(self, ray_diff, mask):
        if self.args.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)  # [n_rays, n_samples, 1, 1, n_views, 1]
            exp_dot_prod = torch.exp(torch.abs(self.anti_alias_s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=-2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=-2, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=-2, keepdim=True) + 1e-8)
        weight = weight / math.prod(self.args.kernel_size)
        return weight
    
    ### DEBUG 2024-04-04 ###
    def create_optimizer(self):
        params_list = [{'params': self.feature_net.parameters(), 'lr': self.args.lrate_feature},
                       {'params': self.mlps['coarse'].parameters(),  'lr': self.args.lrate_mlp}]
        if self.mlps['fine'] is not None:
            params_list.append({'params': self.mlps['fine'].parameters(), 'lr': self.args.lrate_mlp})

        if self.args.pre_net:
            params_list.append({'params': self.pre_net.parameters(), 'lr': self.args.lrate_feature})

        optimizer = torch.optim.Adam(params_list)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.args.lrate_decay_steps,
                                                    gamma=self.args.lrate_decay_factor)

        return optimizer, scheduler
    
    def load_from_ckpt(self, out_folder: Path):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        ckpt = None

        if out_folder.exists() and self.args.resume_training:
            print_link(out_folder, "[*] Resume training looking for ckpt in ")
            try:
                ckpt = get_latest_file(out_folder, "*.pth")
            except ValueError:
                pass

        if self.args.ckpt_path is not None and not self.args.resume_training:
            if not self.args.ckpt_path.exists():  # load the specified ckpt
                raise FileNotFoundError(f"requested ckpt_path does not exist: {self.args.ckpt_path}")
            ckpt = self.args.ckpt_path

        if ckpt is not None and not self.args.no_reload:
            step = int(ckpt.stem[-6:])
            print_link(ckpt, '[*] Reloading from', f"starting at step={step}")
            self.load_model(ckpt)
        else:
            if ckpt is None:
                print('[*] No ckpts found, training from scratch...')
                print(str(self.args.ckpt_path))
            if self.args.no_reload:
                print('[*] no_reload, training from scratch...')

            step = 0

        return step
    ### END DEBUG ###

if __name__ == '__main__':
    import numpy as np
    import json
    from robust.render_ray import RayRender
    from robust.sample_ray import RaySampler
    from robust.configs.config import CustomArgumentParser
    from robust.dataloaders.create_training_dataset import create_training_dataset
    from robust.utils.loss_utils import NANLoss
    gpu = "cuda:0"

    model_cfg = DotDict({
        "expname": "teststub",
        "render_stride": 2,
        "distributed": False,
        "train_dataset": "nerf_synthetic",
        "eval_dataset": "nerf_synthetic",
        "eval_scenes": [
            "lego",
            "chair"],
        "n_iters": 255000,
        "N_rand": 512,
        "lrate_feature": 0.001,
        "lrate_mlp": 0.0005,
        "lrate_decay_factor": 0.5,
        "lrate_decay_steps": 50000,
        "losses": [
            "l2",
            "l1",
            "l1_grad",
            "ssim"],
        "losses_weights": [
            0,
            1,
            0,
            0],
        "workers": 12,
        "chunk_size": 2048,
        "N_importance": 64,
        "N_samples": 64,
        "inv_uniform": True,
        "white_bkgd": False,
        "i_img": 1000,
        "i_print": 100,
        "i_tb": 20,
        "i_weights": 5000,
        "no_load_opt": False,
        "no_load_scheduler": False,
        "sup_clean": True,
        "include_target": True,
        "eval_gain": 16,
        "std": [
            -3,
            -0.5,
            -2,
            -0.5],
        "views_attn": True,
        "kernel_size": (
            3,
            3),
        "pre_net": True,
        "noise_feat": True,
        "rgb_weights": True,
        "local_rank": 0
    })
    parser = CustomArgumentParser.config_parser()
    cfg = parser.parse_args()
    # the cfg from parser is populated with default values
    # let's add our custom values from model_cfg
    for key in model_cfg.keys():
        setattr(cfg, key, model_cfg[key])

    model = RobustModel(cfg)
    _input = torch.randn(512, 64, 1, 1, 8, 35)
    _out = model(_input)
    
    
"""    
    criterion = NANLoss(cfg)
    scalars_to_log = {}
    print(cfg) # TODO Dump this in a nice way
    
    train_dataset, train_sampler = create_training_dataset(cfg)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                        worker_init_fn=lambda _: np.random.seed(),
                                                        num_workers=cfg.workers,
                                                        pin_memory=True,
                                                        sampler=train_sampler,
                                                        shuffle=True if train_sampler is None else False)
    B,N,H,W = 8, 10, 40, 60
    train_data = iter(train_loader).__next__()
    # train_data = dict([
    #     ('camera', torch.randn((1, 34))),
    #     ('src_rgbs_clean', torch.randn((B, N, H, W, 3))),
    #     ('src_rgbs', torch.randn((B, N, H, W, 3))),
    #     ('src_cameras', torch.randn((B, N, 34))),
    #     ('depth_range', torch.randn((1,2))),
    #     ('sigma_estimate', torch.randn((B, N, H, W, 3))),
    #     ('white_level', torch.randn((1,1))),
    #     ('rgb_clean', torch.randn((B, H, W, 3))),
    #     ('rgb', torch.randn((B, H, W, 3))),
    #     ('gt_depth', None),
    #     ('rgb_path', ["/dev/null" for i in range(B)])
    # ])
    print("camera: ", train_data['camera'])
    def single_training_loop(train_data):
        # Create object that generate and sample rays
        ray_sampler = RaySampler(train_data, gpu)
        N_rand = int(1.0 * cfg.N_rand * cfg.num_source_views / train_data['src_rgbs'][0].shape[0])

        # Sample subset (batch) of rays for the training loop
        ray_batch = ray_sampler.random_ray_batch(N_rand,
                                                 sample_mode=cfg.sample_mode,
                                                 center_ratio=cfg.center_ratio,
                                                 clean=cfg.sup_clean)
        ray_render = RayRender(model=model, args=model_cfg, device=gpu)
        # Calculate the feature maps of all views.
        # This step is seperated because in evaluation time we want to do it once for each image.
        org_src_rgbs = ray_sampler.src_rgbs.to(gpu)
        proc_src_rgbs, featmaps = ray_render.calc_featmaps(src_rgbs=org_src_rgbs)

        # Render the rgb values of the pixels that were sampled
        batch_out = ray_render.render_batch(ray_batch=ray_batch, proc_src_rgbs=proc_src_rgbs, featmaps=featmaps,
                                                 org_src_rgbs=org_src_rgbs,
                                                 sigma_estimate=ray_sampler.sigma_estimate.to(gpu))

        # compute loss
        model.optimizer.zero_grad()
        loss = criterion(batch_out['coarse'], ray_batch, scalars_to_log)

        if batch_out['fine'] is not None:
            loss += criterion(batch_out['fine'], ray_batch, scalars_to_log)

        loss.backward()
        scalars_to_log['loss'] = loss.item()
        model.optimizer.step()
        model.scheduler.step()

        scalars_to_log['lr_features'] = model.scheduler.get_last_lr()[0]
        scalars_to_log['lr_mlp'] = model.scheduler.get_last_lr()[1]

        return batch_out, ray_batch
        ###################
    ### Debugging begin 24-04-05
    # pdb.set_trace()
    ray_batch_out, ray_batch_in = single_training_loop(train_data)
"""