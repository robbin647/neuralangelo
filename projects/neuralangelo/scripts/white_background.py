# Attempt to change the png file's background to white

import argparse
import os
import os.path as osp
import pdb
import numpy as np
import imageio.v2 as imageio

from glob import glob

def parse_args():
    myarg = argparse.ArgumentParser()
    myarg.add_argument("--src", type=str, required=True, help="Path to load PNG images")
    myarg.add_argument("--out", type=str, required=True, help="Path to save modified PNG images")
    args = myarg.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    IN_FOLDER = args.src
    OUT_FOLDER = args.out
    if not osp.exists(IN_FOLDER):
        raise Exception(f"Input folder {IN_FOLDER} not found!")
    os.makedirs(osp.abspath(OUT_FOLDER), exist_ok=True)
    
    in_png_imgs = glob(osp.abspath(IN_FOLDER) + "/*.png")
    basenames = [osp.basename(f) for f in in_png_imgs]
    out_png_imgs = [osp.join(osp.abspath(OUT_FOLDER), bn) for bn in basenames]
    for in_img_path, out_img_path in list(zip(in_png_imgs, out_png_imgs)):
        src_img = imageio.imread(in_img_path)
        src_img = (src_img / 255.).astype(np.float32) # [W, H, RGBA]
        img_rgb = src_img[..., :3]*src_img[..., -1:] + (1.-src_img[..., -1:]) # Turn background to white
        img_alpha = np.ones_like(src_img[..., -1:])
        img_out = np.concatenate([img_rgb, img_alpha], axis=-1) # (H, W, 4)
        img_out = np.clip(img_out * 255, 0, 255).astype(np.uint8)

        imageio.imwrite(out_img_path, img_out)
        print(f"Saved image to {out_img_path}")

