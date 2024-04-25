import sys
sys.path.insert(0, '/root/my_code/neuralangelo')
import torch
import os
import os.path as osp
import matplotlib.pyplot as plt
import PIL.Image as Image
import torchvision.transforms.functional as F
import cv2
import torchvision
import argparse
import pdb
from glob import glob
from pathlib import Path

def add_noise(images, read_std=0.037, shot_std=0.093):
  """
  Add per-channel read and shot noise to each pixel of the images

  INPUT
  ----------
  images: Tensor (N, H, W, C) where N is the number of images
  shot_std: the multiplicative scale for poisson (shot) noise
  read_std: the standard deviation of Gaussian (read) noise.

  OUTPUT
  ----------
  A tensor (N,H,W,C) of noise-added images
  """
  result = []
  if len(images.size()) <= 3:
    images = images.unsqueeze(0)
  for n_idx in range(images.size(0)):
    img = images[n_idx].clone()
    poisson_noise = shot_std * torch.sqrt(img) * torch.randn(img.size())
    gaussian_noise = read_std * torch.randn(img.size())
    img = img.float() + poisson_noise + gaussian_noise
    img = torch.clamp(img, min=0., max=1.0)
    result.append(img)
  return torch.stack(result)

def parse_args():
    myargparser = argparse.ArgumentParser()
    myargparser.add_argument("--root", type=str, default=None, action="append", help="A list of colmap scene roots")    
    myargparser.add_argument("--image_ext", type=str, default="png", help="Image file type. Possible values: png, jpg")
    return myargparser.parse_args()

if __name__ == '__main__':
    '''
    Create noised images for each multiview scene
    '''
    args = parse_args()
    print(args)

    if args.root is not None:
      scene_roots = args.root
    else: 
      scene_roots = ['/root/autodl-tmp/data/nerf_synthetic/lego',
                  # '/root/autodl-tmp/data/nerf_synthetic/chair',
                  # '/root/autodl-tmp/data/dtu/dtu_scan24',
                  # '/root/autodl-tmp/data/dtu/dtu_scan37'
                  ]
    noise_levels = {"noise_2": {"read_std": 1.48e-1, "shot_std":3.72e-1 },
                    "noise_1":{"read_std": 3.7e-2, "shot_std": 9.3e-2},
                    }
    for lv, lv_cfg in noise_levels.items():
        for root in scene_roots:
            rgb_files = []
            if args.image_ext == "jpg":
                rgb_files = glob(osp.join(root, 'images') + '/*.jpg')
            else:
                rgb_files = glob(osp.join(root, 'images') + '/*.png')
            
            noise_image_folder = osp.join(root, lv)
            os.makedirs(noise_image_folder, exist_ok = True)
            pdb.set_trace()
            for _file in rgb_files:
                noise_image_file = osp.join(noise_image_folder, osp.basename(_file))
                clean_array = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2RGB)
                clean_tensor = torchvision.transforms.ToTensor()(clean_array)
                noisy_tensor = add_noise(clean_tensor, read_std=lv_cfg['read_std'], shot_std=lv_cfg['shot_std'])[0]
                noisy_img = F.to_pil_image(noisy_tensor)
                noisy_img.save(noise_image_file)
            print(f"Finished writing noise to {noise_image_folder}")