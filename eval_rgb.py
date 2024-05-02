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

import argparse
import os
import pdb
import torch
import numpy as np
import cv2
import glob
import pyiqa
from torchvision.transforms import v2
from copy import deepcopy

import imaginaire.config
from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.distributed import init_dist, get_world_size, master_only_print as print, is_master
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.utils.misc import to_device
from imaginaire.datasets.utils.get_dataloader import get_val_dataloader
from imaginaire.trainers.utils.logging import init_logging
from imaginaire.trainers.utils.get_trainer import get_trainer
from imaginaire.utils.set_random_seed import set_random_seed
from pdb import set_trace

def parse_args():
    parser = argparse.ArgumentParser(description='Test neuralangelo')
    parser.add_argument('--config', help='Path to the training config file.', required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.', default=None)
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--show_pbar', action='store_true')
    parser.add_argument('--wandb', action='store_true', help="Enable using Weights & Biases as the logger")
    parser.add_argument('--wandb_name', default='default', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument("--dump_dir",required=True, type=str, help="The directory to dump test results")
    parser.add_argument("--nan", action="store_true", help="A flag when specified will feed nan with noise data, but validate using clean data")
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd

def dump_data(batch_data, output_dir, seq_idx):
    """
    PARAMETERS
    -------------
    all_data: dict. A dictionary where k-v pairs store the test result for this batch
    output_dir: str. The path to save the following data:
        1) ground truth RGB image
        2) RGB rendered from the same view as 1)
        3) camera pose matrix associated with 2) 
    seq_idx: int. A unique integer that identifies the dumped test sample
    """
    gt_imgs = np.clip(batch_data["image"].cpu().numpy().transpose(0,2,3,1)* 255, 0, 255)
    rendered_imgs = np.clip(batch_data["rgb_map"].cpu().numpy().transpose(0,2,3,1)* 255, 0, 255)
    pose_mtx = batch_data["pose"].cpu().numpy()
    # iterate through each sample
    for sample_idx in range(gt_imgs.shape[0]):
        cv2.imwrite(os.path.join(output_dir, f"gt_rgb_{seq_idx + sample_idx}.png"), cv2.cvtColor(gt_imgs[sample_idx], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"rendered_rgb_{seq_idx + sample_idx}.png"), cv2.cvtColor(rendered_imgs[sample_idx], cv2.COLOR_RGB2BGR))
        np.savez(os.path.join(output_dir, f"pose_{seq_idx + sample_idx}.npz"), pose=pose_mtx[sample_idx])
    # print(f"Saved {seq_idx} test sample to {output_dir}")
    
def eval_metrics(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform_function = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    psnr = pyiqa.create_metric("psnr", color_space='rgb').to(device)
    ssim = pyiqa.create_metric('ssim', device=device)
    lpips = pyiqa.create_metric('lpips', device=device)
    dists = pyiqa.create_metric('dists', device=device)

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    dists_scores = []

    for item in glob.glob(args.dump_dir + "/gt_rgb_*.png"):
        file_name = os.path.basename(item)
        gt_img = cv2.imread(os.path.join(args.dump_dir, file_name), cv2.COLOR_BGR2RGB)
        rendered_img = cv2.imread(os.path.join(args.dump_dir, "rendered_rgb_" + file_name.split("_")[-1]), cv2.COLOR_BGR2RGB)
        _psnr = psnr(transform_function(gt_img).to(device).unsqueeze(0), transform_function(rendered_img).to(device).unsqueeze(0)).cpu().item()
        _ssim = ssim(transform_function(gt_img).to(device).unsqueeze(0), transform_function(rendered_img).to(device).unsqueeze(0)).cpu().item()
        _lpips = lpips(transform_function(gt_img).to(device).unsqueeze(0), transform_function(rendered_img).to(device).unsqueeze(0)).cpu().item()
        _dists = dists(transform_function(gt_img).to(device).unsqueeze(0), transform_function(rendered_img).to(device).unsqueeze(0)).cpu().item()
        psnr_scores.append(_psnr)
        ssim_scores.append(_ssim)
        lpips_scores.append(_lpips)
        dists_scores.append(_dists)

    print(psnr_scores)
    print("Mean PSNR: ", np.mean(np.array(psnr_scores)))
    print(ssim_scores)
    print("Mean SSIM: ", np.mean(np.array(ssim_scores)))
    print(lpips_scores)
    print("Mean LPIPS: ", np.mean(np.array(lpips_scores)))
    print(dists_scores)
    print("Mean DISTS: ", np.mean(np.array(dists_scores)))


def main():
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)

    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)

    # If args.single_gpu is set to True, we will disable distributed data parallel.
    if not args.single_gpu:
        # this disables nccl timeout
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Training with {get_world_size()} GPUs.")

    # set random seed by rank
    set_random_seed(args.seed, by_rank=True)

    # Global arguments.
    imaginaire.config.DEBUG = args.debug

    # Create log directory for storing training results.
    # cfg.logdir = init_logging(args.config, args.logdir, makedir=True)

    # Print and save final config
    if is_master():
        cfg.print_config()
        # cfg.save_config(cfg.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    trainer = get_trainer(cfg, is_inference=True, seed=args.seed)
    
    ## TODO: Figure out config for validation data loader
    trainer.set_data_loader(cfg, split="val")

    # Make a copy of cfg and provide additional gt dataloader for nan
    pdb.set_trace()
    if args.nan:
        print("*" * 40, "\n",">"*10 + "NAN evaluation enabled" + "<"*10,"\n", "*"*40)
        gt_cfg = deepcopy(cfg) # Caution: deepcopy failed for Config object!
        recursive_update_strict(gt_cfg, cfg)
        gt_cfg.data.image_folder_name = "images"
        trainer.set_data_loader(gt_cfg, split="gt")
        # trainer.gt_data_loader will be available
    
    trainer.checkpointer.load(args.checkpoint, args.resume, load_sch=True, load_opt=True)
    trainer.mode = 'val'
    

    with torch.no_grad():
        # Initialize testing loop
        model = trainer.model.module
        model.eval()
        test_loader = trainer.eval_data_loader
        if args.nan:
            gt_loader = trainer.gt_data_loader
            gt_iterator = iter(gt_loader)
        if not os.path.exists(args.dump_dir):
            os.makedirs(args.dump_dir, exist_ok=True)

        checkpoint_iteration = 833
        seq_idx = 0
        for it, data in enumerate(test_loader):
            if args.nan:
                gt_batch = next(gt_iterator)
            print("===="*2, "Iteration {}".format(it+1), "===="*2)
            data = to_device(data, "cuda")
            
            data = trainer.start_of_iteration(data, current_iteration=checkpoint_iteration)
            
            output = model.inference(data)
            data.update(output)
            
            if args.nan: # nan only: replace the noisy validation data with real gt data
                data['image'] = gt_batch['image']
            dump_data(data, args.dump_dir, seq_idx)
            seq_idx += data["idx"].size()[0]
    
        eval_metrics(args)


if __name__ == "__main__":
    main()
