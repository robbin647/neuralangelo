# Attempt to change the png file's background to white

import argparse
import os
import os.path as osp
import pdb
import numpy as np
import imageio.v2 as imageio

from glob import glob

def save_to_white_bg(in_folder: str, out_folder="white_bg"):
    """
    """
    if not osp.exists(in_folder):
        raise Exception(f"Input folder {in_folder} not found!")
    if out_folder == "white_bg":
        out_folder = osp.join(osp.dirname(in_folder), "white_bg") # by default out_folder is in the same parent directoy of in_folder
    os.makedirs(osp.abspath(out_folder), exist_ok=True)

    in_png_imgs = glob(osp.abspath(in_folder) + "/*.png")
    basenames = [osp.basename(f) for f in in_png_imgs]
    out_png_imgs = [osp.join(osp.abspath(out_folder), bn) for bn in basenames]
    for in_img_path, out_img_path in list(zip(in_png_imgs, out_png_imgs)):
        src_img = imageio.imread(in_img_path)
        src_img = (src_img / 255.).astype(np.float32) # [W, H, RGBA]
        img_rgb = src_img[..., :3]*src_img[..., -1:] + (1.-src_img[..., -1:]) # Turn background to white
        img_alpha = np.ones_like(src_img[..., -1:])
        img_out = np.concatenate([img_rgb, img_alpha], axis=-1) # (H, W, 4)
        img_out = np.clip(img_out * 255, 0, 255).astype(np.uint8)

        imageio.imwrite(out_img_path, img_out)
        print(f"Saved white background image to {out_img_path}")


def parse_args():
    myarg = argparse.ArgumentParser()
    myarg.add_argument("--src", type=str, required=True, help="Path to load PNG images")
    myarg.add_argument("--out", type=str, required=False, default="white_bg", help="Path to save modified PNG images. Default to `white_bg` and lie in the same parent directory as `src`")
    args = myarg.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    IN_FOLDER = args.src
    OUT_FOLDER = args.out
    save_to_white_bg(IN_FOLDER, OUT_FOLDER)
