from robust.dataloaders.basic_dataset import Mode
from robust.dataloaders.nerf_synthetic import NerfSyntheticDataset
from robust.dataloaders.data_utils import get_nearest_pose_ids

import os
import numpy as np
import random
import json
import imageio
import torch
import pdb

# def generate_unique_random_integers(n, start, end):
#     """
#     Generates a sequence of non-repeating random integers.

#     Args:
#         n (int): Number of unique integers to generate.
#         start (int): Start of the range (inclusive).
#         end (int): End of the range (inclusive).

#     Returns:
#         list: List of non-repeating random integers.
#     """
#     if end - start + 1 < n:
#         raise ValueError("Range is too small for the number of unique integers requested.")
#     return random.sample(range(start, end + 1), n)
def re_linearize(rgb, wl=1.):
    """
    Approximate re-linearization of RGB values by revert gamma correction and apply white level
    Revert gamma correction
    @param rgb:
    @param wl:
    @return:
    """
    # return rgb
    return wl * (rgb ** 2.2)

class COLMAPDataset(NerfSyntheticDataset):
    def __init__(self, args, mode, scenes=(), **kwargs):
        super().__init__(args, mode, scenes, **kwargs)
        rgb_files_num = len(self.render_rgb_files)
        # self.id_pool = generate_unique_random_integers(rgb_files_num, 0, rgb_files_num)

    def add_single_scene(self, _, scene_path):
        pose_file = os.path.join(scene_path, f'transforms.json')
        rgb_files, intrinsics, poses =self.read_cameras_colmap(pose_file)
        if self.mode != Mode.train:
            rgb_files = rgb_files[::self.testskip]
            intrinsics = intrinsics[::self.testskip]
            poses = poses[::self.testskip]
        self.render_rgb_files.extend(rgb_files)
        self.render_poses.extend(poses)
        self.render_intrinsics.extend(intrinsics)
    
    def read_cameras_colmap(self, pose_file):
        basedir = os.path.dirname(pose_file)
        with open(pose_file, 'r') as fp:
            meta = json.load(fp)

        camera_angle_x = float(meta['camera_angle_x'])
        rgb_files = []
        c2w_mats = []

        img = imageio.imread(os.path.join(basedir, meta['frames'][0]['file_path']))
        H, W = img.shape[:2]
        focal = .5 * W / np.tan(.5 * camera_angle_x) #摄像机焦距 0.5*W / tan(广度角/2)
        intrinsics = np.array([[focal, 0, 1.0 * W / 2, 0],
                     [0, focal, 1.0 * H / 2, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
        for i, frame in enumerate(meta['frames']):
            rgb_file = os.path.join(basedir, meta['frames'][i]['file_path'])
            rgb_files.append(rgb_file)
            c2w = np.array(frame['transform_matrix'])
            w2c_blender = np.linalg.inv(c2w)
            w2c_opencv = w2c_blender
            w2c_opencv[1:3] *= -1
            c2w_opencv = np.linalg.inv(w2c_opencv)
            c2w_mats.append(c2w_opencv)
        c2w_mats = np.array(c2w_mats)
        return rgb_files, np.array([intrinsics] * len(meta['frames'])), c2w_mats
    
    def __getitem__(self, idx):
        """
        注意：这里idx是指在所有scene组成的rgb图像、pose文件中的索引。比如包含两个场景：chair有99张图,lego有49张图，则idx是从0到147
        """
        
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx]

        train_pose_file = os.path.join('/'.join(rgb_file.split('/')[:-2]), 'transforms.json')
        train_rgb_files, train_intrinsics, train_poses = self.read_cameras_colmap(train_pose_file)

        if self.mode == Mode.train:
            # id_render = int(os.path.basename(rgb_file)[:-4]) # instead of parsing file name, use list index as id. As file names may not always contain a number 
            id_render = train_rgb_files.index(rgb_file) # 返回该元素第一个出现的下标（正常情况下存在且唯一）
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
        else:
            id_render = -1
            subsample_factor = 1

        rgb = self.read_image(rgb_file)
        rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]
        camera = self.create_camera_vector(rgb, render_intrinsics, render_pose)
        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                int(self.num_source_views * subsample_factor),
                                                tar_id=id_render,
                                                angular_dist_method='vector')
        nearest_pose_ids = self.choose_views(nearest_pose_ids, self.num_source_views, id_render)

        src_rgbs = []
        src_cameras = []
        for idx in nearest_pose_ids:
            src_rgb = self.read_image(train_rgb_files[idx])
            src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
            train_pose = train_poses[idx]
            train_intrinsics_ = train_intrinsics[idx]

            src_rgbs.append(src_rgb)
            src_camera = self.create_camera_vector(src_rgb, train_intrinsics_, train_pose)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        depth_range = self.final_depth_range()

        batch_dict = self.create_batch_from_numpy(rgb, camera, rgb_file, src_rgbs, src_cameras, depth_range)        
        # # 训练时可选的小伎俩：设置噪声的sigma_estimate值 (color blending用到)
        # max_sigma = (self.sig_read ** 2 + self.sig_shot ** 2 * 1.0) ** 0.5# 就是根据sigma_read, sigma_shot合并算出来的最大sigma
        # B, N, H, W, _ = batch_dict['src_rgbs_clean'].shape
        # # 因为是per-pixel per-color-channel, 所以sigma_estimate要跟原RGB有一样的维度
        # batch_dict['sigma_estimate'] = torch.tensor([max_sigma, self.sig_read, self.sig_shot]).expand(B, N, H, W, 3) 
        return batch_dict
    
    """@override
    重写了父父类NoiseDataset的该方法
    1. 主要是output里面删去rgb_clean和src_rgbs_clean。因为默认是直接读colmap跑出来的图片，噪声已经直接加到图片里了。
    即：rgb 以及src_rgbs 才是直接读取的有噪声的图片
    2. 自己重新构造了sigma_estimate
    """
    def create_batch_from_numpy(self, rgb_clean, camera, rgb_file, src_rgbs_clean, src_cameras, depth_range,
                                gt_depth=None):
        if self.mode in [Mode.train, Mode.validation]:
            white_level = 10 ** -torch.rand(1)
        else:
            white_level = torch.Tensor([1])

        if rgb_clean is not None:
            rgb = re_linearize(torch.from_numpy(rgb_clean[..., :3]), white_level)
        else:
            rgb = None
        src_rgbs = re_linearize(torch.from_numpy(src_rgbs_clean[..., :3]), white_level)

        # 训练时可选的小伎俩：设置噪声的sigma_estimate值 (color blending用到)
        max_sigma = (self.sig_read ** 2 + self.sig_shot ** 2 * 1.0) ** 0.5# 就是根据sigma_read, sigma_shot合并算出来的最大sigma
        N, H, W, _ = src_rgbs_clean.shape
        # 因为是per-pixel per-color-channel, 所以sigma_estimate要跟原RGB有一样的维度
        sigma_estimate = torch.tensor([max_sigma, self.sig_read, self.sig_shot]).expand(N, H, W, 3) 

        batch_dict = {'camera'        : torch.from_numpy(camera),
                      'rgb_path'      : str(rgb_file),
                      'src_rgbs'      : src_rgbs,
                      'src_cameras'   : torch.from_numpy(src_cameras),
                      'depth_range'   : depth_range,
                      'sigma_estimate': sigma_estimate,
                      'white_level'   : white_level}

        if rgb_clean is not None:
            batch_dict['rgb'] = rgb

        if gt_depth is not None:
            batch_dict['gt_depth'] = gt_depth

        return batch_dict

    def __len__(self):
        return super().__len__()
    