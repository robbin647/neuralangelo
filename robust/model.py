import sys
sys.path.insert(0, '/root/my_code/neuralangelo')

import torch
import torch.nn as nn
import kornia

# import from project root
from robust.configs.config import OUT_DIR
from functools import partial
from typing import Dict, Tuple
from pathlib import Path

# import from current folder
from robust.utils.io_utils import get_latest_file, print_link
from attention import EncoderLayer
from feature_encoder import ResUNet
from nan_mlp import NanMLP
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

        # create feature extraction network
        self.feature_net = ResUNet(coarse_out_ch=args.coarse_feat_dim,
                                   fine_out_ch=args.fine_feat_dim,
                                   coarse_only=args.coarse_only).to(device)

        # create coarse NAN mlps
        self.net_coarse = self.nan_factory('coarse', device)
        self.net_fine = None

        if not args.coarse_only:
            # create fine NAN mlps
            self.net_fine = self.nan_factory('fine', device)

        self.mlps: Dict[str, NanMLP] = {'coarse': self.net_coarse, 'fine': self.net_fine}

        if args.pre_net:
            self.pre_net = Gaussian2D(in_channels=3, out_channels=3, kernel_size=(3, 3), sigma=(1.5, 1.5)).to(device)
        else:
            self.pre_net = None

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
    
    def nan_factory(self, net_type, device) -> NanMLP:
        if net_type == 'coarse':
            feat_dim = self.args.coarse_feat_dim
            n_samples = self.args.N_samples
        elif net_type == 'fine':
            feat_dim = self.args.fine_feat_dim
            n_samples = self.args.N_samples + self.args.N_importance
        else:
            raise NotImplementedError

        return NanMLP(self.args,
                      in_feat_ch=feat_dim,
                      n_samples=n_samples).to(device)
    
    ### DEBUG 2024-04-04 ###
    def create_optimizer(self):
        params_list = [{'params': self.feature_net.parameters(), 'lr': self.args.lrate_feature},
                       {'params': self.net_coarse.parameters(),  'lr': self.args.lrate_mlp}]
        if self.net_fine is not None:
            params_list.append({'params': self.net_fine.parameters(), 'lr': self.args.lrate_mlp})

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

    def forward(self, data):
        x = self.encoderNetwork(data)
        pdb.set_trace()
        return x
        # [dist, x] = self.neural_sdf(x)
        # x = self.neural_rgb(torch.cat(x, encode(viewpoint), ))

if __name__ == '__main__':
    from robust.render_ray import RayRender
    from robust.sample_ray import RaySampler
    from robust.configs.config import CustomArgumentParser
    gpu = "cuda:0"

    class DotDict(dict):
        """Custom dictionary class that allows attribute access using the dot operator."""
        def __getattr__(self, item):
            return self.get(item)

        def __setattr__(self, key, value):
            self[key] = value

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
        "workers": 0,
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
        "kernel_size": [
            3,
            3],
        "pre_net": True,
        "noise_feat": True,
        "rgb_weights": True,
        "local_rank": 0
    })
    parser = CustomArgumentParser.config_parser()
    model_cfg = parser.parse_args()
    model = RobustModel(model_cfg)
    B,N,H,W = 8, 10, 40, 60
    train_data = dict([
        ('camera', torch.randn((B, 34))),
        ('src_rgbs_clean', torch.randn((B, N, H, W, 3))),
        ('src_rgbs', torch.randn((B, N, H, W, 3))),
        ('src_cameras', torch.randn((B, N, 34))),
        ('depth_range', torch.randn((1,2))),
        ('sigma_estimate', torch.randn((B, N, H, W, 3))),
        ('white_level', torch.randn((1,1))),
        ('rgb_clean', torch.randn((B, H, W, 3))),
        ('rgb', torch.randn((B, H, W, 3))),
        ('gt_depth', None),
        ('rgb_path', ["/dev/null" for i in range(B)])
    ])
    ray_sampler = RaySampler(train_data, gpu)
    ray_render = RayRender(model=model, args=model_cfg, device=gpu)
    pdb.set_trace()