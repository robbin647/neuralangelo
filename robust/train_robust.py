import sys
sys.path.insert(0, '/root/autodl-tmp/code/neuralangelo')

import torch
import torch.nn as nn
import kornia

# import from project root
from robust.configs.config import OUT_DIR
from functools import partial
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
from robust._model import RobustModel
from robust.render_ray import RayRender
from robust.sample_ray import RaySampler
from robust.configs.config import CustomArgumentParser
from robust.dataloaders.create_training_dataset import create_training_dataset
from robust.utils.loss_utils import NANLoss
from projects.neuralangelo.model import Model as NeuraModel
from projects.neuralangelo.trainer import Trainer as NeuraTrainer

import pdb
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
    "train_dataset": "colmap_dataset",
    "eval_dataset": "colmap_dataset",
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
    # "workers": 12, temporarily set to 0 for debugging
    "workers": 0,
    "chunk_size": 2048,
    # "N_importance": 64, 暂时设为0，因为不为零的话会调用RobustModel.mlps['fine']网络(未定义！)，等fine网络实现好了再设回去
    "N_importance": 0,
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
    "eval_gain": 1,
    "sig_read": 9.3e-2,
    "sig_shot": 3.7e-2,
    # "eval_gain": 2,
    # "sig_read": 3.72e-1,
    # "sig_shot": 1.48e-1,
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

def single_training_loop(train_data):
    # Create object that generate and sample rays
    ray_sampler = RaySampler(train_data, gpu)
    N_rand = int(1.0 * cfg.N_rand * cfg.num_source_views / train_data['src_rgbs'][0].shape[0])

    # Sample subset (batch) of rays for the training loop
    ray_batch = ray_sampler.random_ray_batch(N_rand,
                                                sample_mode=cfg.sample_mode,
                                                center_ratio=cfg.center_ratio,
                                                clean=cfg.sup_clean)
    ray_render = RayRender(model=model, neura_model=neura_model, args=model_cfg, device=gpu)
    # Calculate the feature maps of all views.
    # This step is seperated because in evaluation time we want to do it once for each image.
    org_src_rgbs = ray_sampler.src_rgbs.to(gpu)
    proc_src_rgbs, featmaps = ray_render.calc_featmaps(src_rgbs=org_src_rgbs)

    # Render the rgb values of the pixels that were sampled
    batch_out = ray_render.render_batch(ray_batch=ray_batch, proc_src_rgbs=proc_src_rgbs, featmaps=featmaps,
                                                org_src_rgbs=org_src_rgbs,
                                                sigma_estimate=ray_sampler.sigma_estimate.to(gpu))
    
    # batch_out {rgb: [N_rays, 3], depth: [N_rays,], weights: [N_rays,]}

    # compute loss
    model.optimizer.zero_grad()
    loss = criterion(batch_out)

    loss.backward()
    scalars_to_log['loss'] = loss.item()
    model.optimizer.step()
    model.scheduler.step()

    scalars_to_log['lr_features'] = model.scheduler.get_last_lr()[0]
    scalars_to_log['lr_mlp'] = model.scheduler.get_last_lr()[1]

    return batch_out, ray_batch
    ###################
if __name__ == '__main__':
    parser = CustomArgumentParser.config_parser()
    cfg = parser.parse_args()
    # the cfg from parser is populated with default values
    # let's add our custom values from model_cfg
    for key in model_cfg.keys():
        setattr(cfg, key, model_cfg[key])

    model = RobustModel(cfg)
    from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments
    neuralangelo_yaml_cfg = Config('/root/autodl-tmp/data/neuralangelo_log/synthetic_chair/config.yaml')
    neura_trainer = NeuraTrainer(neuralangelo_yaml_cfg, is_inference=False)
    neura_model = neura_trainer.model.module 
    neura_model.neural_sdf.set_active_levels(current_iter=0)
    neura_model.neural_sdf.set_normal_epsilon()
    
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
    for train_data in train_loader:
        print("camera: ", train_data['camera'])
        ray_batch_out, ray_batch_in = single_training_loop(train_data)
        break
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

