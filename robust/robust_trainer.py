'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import math
import pdb
import torch
import torch.nn.functional as torch_F
import tqdm
import wandb
import os
import os.path as osp
import pdb

from torch.autograd import profiler
from torch.cuda.amp import GradScaler, autocast

from imaginaire.utils.misc import to_cuda, requires_grad, to_cpu, Timer
from imaginaire.utils.distributed import master_only
from imaginaire.utils.visualization import wandb_image
from projects.nerf.trainers.base import BaseTrainer
from projects.neuralangelo.utils.misc import get_scheduler, eikonal_loss, curvature_loss
from torch.utils.tensorboard import SummaryWriter
tensorboard_log_interval = 10

class Trainer(BaseTrainer):

    def __init__(self, cfg, is_inference=True, seed=0):
        super().__init__(cfg, is_inference=is_inference, seed=seed)
        self.metrics = dict()
        self.warm_up_end = cfg.optim.sched.warm_up_end
        self.cfg_gradient = cfg.model.object.sdf.gradient
        if cfg.model.object.sdf.encoding.type == "hashgrid" and cfg.model.object.sdf.encoding.coarse2fine.enabled:
            self.c2f_step = cfg.model.object.sdf.encoding.coarse2fine.step
            self.model.module.neural_sdf.warm_up_end = self.warm_up_end
        if cfg.logdir is None:
            raise Exception("cfg.logdir!")
        else:
            global writer # Accessible throughout this module
            writer = SummaryWriter(log_dir=osp.abspath(cfg.logdir))
    
    def _init_loss(self, cfg):
        self.criteria["render"] = torch.nn.L1Loss()

    def setup_scheduler(self, cfg, optim):
        return get_scheduler(cfg.optim, optim)

    def _compute_loss(self, data, mode=None):
        if mode == "train":
            # Compute loss only on randomly sampled rays.
            self.losses["render"] = self.criteria["render"](data["rgb"], data["image_sampled"]) * 3  # ==L1Loss FIXME:sumRGB?!
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb"], data["image_sampled"]).log10()
            if "eikonal" in self.weights.keys():
                self.losses["eikonal"] = eikonal_loss(data["gradients"], outside=data["outside"])
            if "curvature" in self.weights:
                self.losses["curvature"] = curvature_loss(data["hessians"], outside=data["outside"])
        else:
            # Compute loss on the entire image.
            self.losses["render"] = self.criteria["render"](data["rgb_map"], data["image"])
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb_map"], data["image"]).log10()

    def get_curvature_weight(self, current_iteration, init_weight, decay_factor):
        if "curvature" in self.weights:
            weight = (min(current_iteration / self.warm_up_end, 1.) if self.warm_up_end > 0 else 1.) * init_weight
            self.weights["curvature"] = weight / decay_factor

    def _start_of_iteration(self, data, current_iteration):
        model = self.model_module
        self.progress = model.progress = current_iteration / self.cfg.max_iter
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            model.neural_sdf.set_active_levels(current_iteration)
            if self.cfg_gradient.mode == "numerical":
                model.neural_sdf.set_normal_epsilon()
                decay_factor = model.neural_sdf.growth_rate ** model.neural_sdf.add_levels  # TODO: verify?
                self.get_curvature_weight(current_iteration, self.cfg.trainer.loss_weight.curvature, decay_factor)
        elif self.cfg_gradient.mode == "numerical":
            model.neural_sdf.set_normal_epsilon()

        return super()._start_of_iteration(data, current_iteration)

    @master_only
    def log_wandb_scalars(self, data, mode=None):
        super().log_wandb_scalars(data, mode=mode)
        scalars = {
            f"{mode}/PSNR": self.metrics["psnr"].detach(),
            f"{mode}/s-var": self.model_module.s_var.item(),
        }
        if "curvature" in self.weights:
            scalars[f"{mode}/curvature_weight"] = self.weights["curvature"]
        if "eikonal" in self.weights:
            scalars[f"{mode}/eikonal_weight"] = self.weights["eikonal"]
        if mode == "train" and self.cfg_gradient.mode == "numerical":
            scalars[f"{mode}/epsilon"] = self.model.module.neural_sdf.normal_eps
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            scalars[f"{mode}/active_levels"] = self.model.module.neural_sdf.active_levels
        wandb.log(scalars, step=self.current_iteration)

    @master_only
    def log_wandb_images(self, data, mode=None, max_samples=None):
        images = {"iteration": self.current_iteration, "epoch": self.current_epoch}
        if mode == "val":
            images_error = (data["rgb_map"] - data["image"]).abs()
            images.update({
                f"{mode}/vis/rgb_target": wandb_image(data["image"]),
                f"{mode}/vis/rgb_render": wandb_image(data["rgb_map"]),
                f"{mode}/vis/rgb_error": wandb_image(images_error),
                f"{mode}/vis/normal": wandb_image(data["normal_map"], from_range=(-1, 1)),
                f"{mode}/vis/inv_depth": wandb_image(1 / (data["depth_map"] + 1e-8) * self.cfg.trainer.depth_vis_scale),
                f"{mode}/vis/opacity": wandb_image(data["opacity_map"]),
            })
        wandb.log(images, step=self.current_iteration)
    
    """Override"""
    def train(self, cfg, data_loader, single_gpu=False, profile=False, show_pbar=False):
        self.progress = self.model_module.progress = self.current_iteration / self.cfg.max_iter
        # super().train(cfg, data_loader, single_gpu, profile, show_pbar)
        start_epoch = self.checkpointer.resume_epoch or self.current_epoch  # The epoch to start with.
        current_iteration = self.checkpointer.resume_iteration or self.current_iteration  # The starting iteration.

        self.timer.checkpoint_tic()  # start timer
        self.timer.reset_timeout_counter()
        freeze_weights = None
        requires_grad(self.model_module, True)
        for current_epoch in range(start_epoch, cfg.max_epoch):
            if not single_gpu:
                data_loader.sampler.set_epoch(current_epoch)
            self.start_of_epoch(current_epoch)
            if show_pbar:
                data_loader_wrapper = tqdm(data_loader, desc=f"Training epoch {current_epoch + 1}", leave=False)
            else:
                data_loader_wrapper = data_loader
            for it, data in enumerate(data_loader_wrapper):
                with profiler.profile(enabled=profile,
                                      use_cuda=True,
                                      profile_memory=True,
                                      record_shapes=True) as prof:
                    data = self.start_of_iteration(data, current_iteration)
                    """Case 1: neuralangelo 和 nan始终一起train"""
                    pass
                    """Case 2: 0-100 iteration间先冻住neuralangelo只是train nan
                    100个iteration以后开始交替fine tune两个部分"""
                    # if current_iteration < 100: 
                    #     if freeze_weights != "neuralangelo":
                    #         self.model_fine_tune(data, freeze_part="neuralangelo")
                    #         freeze_weights = "neuralangelo"
                    # else:
                    #     if (current_iteration // 100) % 2 == 0 and freeze_weights != "neuralangelo": # 偶数*100 ~ 奇数*100个iteration 训练nan
                    #         self.model_fine_tune(data, freeze_part="neuralangelo")
                    #         freeze_weights = "neuralangelo"
                    #     elif (current_iteration // 100) % 2 == 1 and freeze_weights != "nan": # 奇数*100 ~ 偶数*100个iteration 训练neuralangelo
                    #         self.model_fine_tune(data, freeze_part="nan")
                    #         freeze_weights = "nan"
                    self.train_step(data, current_iteration, last_iter_in_epoch=(it == len(data_loader) - 1))

                    current_iteration += 1
                    if show_pbar:
                        data_loader_wrapper.set_postfix(iter=current_iteration)
                    if it == len(data_loader) - 1:
                        self.end_of_iteration(data, current_epoch + 1, current_iteration)
                    else:
                        self.end_of_iteration(data, current_epoch, current_iteration)
                    if current_iteration >= cfg.max_iter:
                        print('Done with training!!!')
                        return
                if profile:
                    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                    prof.export_chrome_trace(os.path.join(cfg.logdir, "trace.json"))

            self.end_of_epoch(data, current_epoch + 1, current_iteration)
        print('Done with training!!!')

    """Override"""
    def train_step(self, data, global_iter, last_iter_in_epoch=False):
        r"""One training step.

        Args:
            data (dict): Data used for the current iteration.
        """
        # Set requires_grad flags.
        # requires_grad(self.model_module, True)

        # Compute the loss.
        self.timer._time_before_forward()

        autocast_dtype = getattr(self.cfg.trainer.amp_config, 'dtype', 'float16')
        autocast_dtype = torch.bfloat16 if autocast_dtype == 'bfloat16' else torch.float16
        amp_kwargs = {
            'enabled': self.cfg.trainer.amp_config.enabled,
            'dtype': autocast_dtype
        }
        with autocast(**amp_kwargs):
            total_loss, rgb_contrast = self.model_forward(data)

            # Scale down the loss w.r.t. gradient accumulation iterations.
            total_loss = total_loss / float(self.cfg.trainer.grad_accum_iter)
            
            writer.add_scalars("losses", self.losses, global_step=global_iter)
            writer.add_scalars("metrics", self.metrics, global_step=global_iter)
            # if global_iter % 50 == 0:
            #     writer.add_embedding(rgb_contrast[0][0].reshape(1, -1), metadata=["pred_{0}"], global_step=global_iter)
            #     writer.add_embedding(rgb_contrast[1][0].reshape(1, -1), metadata=["gt_{0}"], global_step=global_iter)
        # Backpropagate the loss.
        self.timer._time_before_backward()
        self.scaler.scale(total_loss).backward()

        self._extra_step(data)

        # Perform an optimizer step. This enables gradient accumulation when grad_accum_iter is not 1.
        if (self.current_iteration + 1) % self.cfg.trainer.grad_accum_iter == 0 or last_iter_in_epoch:
            self.timer._time_before_step()
            self.scaler.step(self.optim)
            self.scaler.update()
            # Zero out the gradients.
            self.optim.zero_grad(**self.optim_zero_grad_kwargs)

        # Update model average.
        self.timer._time_before_model_avg()
        if self.cfg.trainer.ema_config.enabled:
            self.model.module.update_average()

        self._detach_losses()
        self.timer._time_before_leave_gen()

    """Freeze neuralangelo's weights and train noise-aware module"""
    def model_fine_tune(self, data, freeze_part):
        assert freeze_part in ["neuralangelo", "nan"]
        neura_modules = ["appear_embed", # may be None
                         "appear_embed_outside", # may be None
                         "neural_sdf",
                         "neural_rgb",
                         "background_nerf"
        ]
        nan_modules = ["ray_transformer",
        ]

        def disable_grads_recursive(mod: torch.nn.Module):
            submodule_list = [c for c in mod.children()]
            if len(submodule_list) == 0: 
                mod.requires_grad_(False)
                return
            for submodule in submodule_list:
                disable_grads_recursive(submodule)

        def enable_grads_recursive(mod: torch.nn.Module):
            submodule_list = [c for c in mod.children()]
            if len(submodule_list) == 0: 
                mod.requires_grad_(True)
                return
            for submodule in submodule_list:
                enable_grads_recursive(submodule)

        if freeze_part == "neuralangelo": 
            for _mod_name in neura_modules:
                if _mod_name == "neural_sdf":
                    # special handling of tcnn_encoding
                    _neural_sdf = self.model.module.get_submodule(_mod_name)
                    for subname, submodule in _neural_sdf.named_children():
                        if subname == "tcnn_encoding":
                            continue # tcnn hash table state has already disabled gradients
                        disable_grads_recursive(submodule)
                else:                     
                    try:  
                        disable_grads_recursive(self.model.module.get_submodule(_mod_name))
                    except AttributeError: # okay when named module does not exist, just skip it
                        continue
            for _mod_name in nan_modules:
                enable_grads_recursive(self.model.module.get_submodule(_mod_name))
        elif freeze_part == "nan":
            for _mod_name in nan_modules:
                disable_grads_recursive(self.model.module.get_submodule(_mod_name))
            for _mod_name in neura_modules:
                if _mod_name == "neural_sdf":
                    # special handling of tcnn_encoding
                    _neural_sdf = self.model.module.get_submodule(_mod_name)
                    for subname, submodule in _neural_sdf.named_children():
                        if subname == "tcnn_encoding":
                            continue # tcnn hash table state has already disabled gradients
                        enable_grads_recursive(submodule)
                else:                        
                    try:
                        enable_grads_recursive(self.model.module.get_submodule(_mod_name))
                    except AttributeError: # okay when named module does not exist, just skip it
                        continue

    """Optional to override parents' model_forward """
    def model_forward(self, data): 
        # Model forward.
        output = self.model(data)  # data = self.model(data) will not return the same data in the case of DDP.
        data.update(output)
        rgb_contrast = [data["rgb"], data["image_sampled"]]
        # Compute loss.
        self.timer._time_before_loss()
        self._compute_loss(data, mode="train")
        total_loss = self._get_total_loss()
        return total_loss, rgb_contrast
    