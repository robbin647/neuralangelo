"""
This dataset was borrowed from projects.neuralangelo.data
"""


import json
import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
import pdb
from PIL import Image, ImageFile

from projects.nerf.datasets import base
from projects.nerf.utils import camera

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(base.Dataset):

    def __init__(self, cfg, is_inference=False):
        super().__init__(cfg, is_inference=is_inference, is_test=False)
        cfg_data = cfg.data
        self.root = cfg_data.root
        self.preload = cfg_data.preload
        self.H, self.W = cfg_data.val.image_size if is_inference else cfg_data.train.image_size
        meta_fname = f"{cfg_data.root}/transforms.json"
        with open(meta_fname) as file:
            self.meta = json.load(file)
        self.list = self.meta["frames"]
        if cfg_data[self.split].subset:
            subset = cfg_data[self.split].subset
            subset_idx = np.linspace(0, len(self.list), subset+1)[:-1].astype(int)
            self.list = [self.list[i] for i in subset_idx]
        self.num_rays = cfg.model.render.rand_rays
        self.readjust = getattr(cfg_data, "readjust", None)
        # Preload dataset if possible.
        if cfg_data.preload:
            self.images = self.preload_threading(self.get_image, cfg_data.num_workers)
            self.cameras = self.preload_threading(self.get_camera, cfg_data.num_workers, data_str="cameras")

    def __getitem__(self, idx):
        """Process raw data and return processed data in a dictionary.

        Args:
            idx: The index of the sample of the dataset.
        Returns: A dictionary containing the data.
                 idx (scalar): The index of the sample of the dataset.
                 ray_idx (R tensor): ray index ranging from 0 to H*W. [R,]
                 image_sampled (Rx3 tensor): pixels from the image visited in the same order as ray_idx. [R, 3]
                 intr (3x3 tensor): The camera intrinsics of `image`.
                 pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
                 neighbor_rgbs: (available only in training) the C*W*H pixel values from N_SRC many nearest neighbor views. [N_SRC, C, W, H] 
                 neighbor_poses: [N_SRC, 34] 34=img_size(2) + intrinsics(16) + extrinsics(16)
        """
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        # Get the images.
        image, image_size_raw = self.images[idx] if self.preload else self.get_image(idx)
        # resize image to self.W, self.H
        image = self.preprocess_image(image)
        # Get the cameras (intrinsics and pose).
        intr, pose = self.cameras[idx] if self.preload else self.get_camera(idx)
        intr, pose = self.preprocess_camera(intr, pose, image_size_raw) # Adjust the intrinsics according to the resized image
       

        # Pre-sample ray indices.
        if self.split == "train":
            ray_idx = torch.randperm(self.H * self.W)[:self.num_rays]  # [R]
            image_sampled = image.flatten(1, 2)[:, ray_idx].t()  # [R,3]
            sample.update(
                ray_idx=ray_idx,
                image_sampled=image_sampled,
                intr=intr,
                pose=pose,
            )
        
        else:  # keep image during inference
            sample.update(
                image=image, # [C, W, H]
                intr=intr,
                pose=pose,
            )
        """
        Collect N_SRC neighbor views and poses
        Needed both for training phase and eval phase
        """        

        src_pose = pose
        
        train_poses = [self.get_camera(idx)[1] for idx in range(self.__len__())] # list[N_train, tensor[3, 4]]
        
        train_poses = torch.cat([cam.unsqueeze(0) for cam in train_poses], dim=0) # tensor [N_train, 3, 4]
        
        nearest_views_ids = self.get_nearest_views_ids(src_pose[:3,:], train_poses[:,:3,:], num_select=4, tar_id=idx)
        neighbor_rgbs = [] 
        neighbor_poses = []
        for _id in nearest_views_ids:
            _raw_img, _raw_size = self.get_image(_id) # Image, tuple(int, int)
            _raw_img = self.preprocess_image(_raw_img) # tensor [C, W, H]
            neighbor_rgbs.append(_raw_img.unsqueeze(0)) 
            _, pose_w2c = self.get_camera(_id) # tensor[3, 4] rotation & translation
            pose_vec = self.to_camera_vector(intr, pose_w2c, self.W, self.H) # [34,]
            neighbor_poses.append(pose_vec.unsqueeze(0)) # [1, 34]

        neighbor_rgbs = torch.cat(neighbor_rgbs, dim=0) # [N_SRC, C, W, H]
        neighbor_poses = torch.cat(neighbor_poses, dim=0) # [N_SRC, 34]
        sample.update(
            neighbor_rgbs=neighbor_rgbs,
            neighbor_poses=neighbor_poses,
            neighbor_ids=nearest_views_ids
        )

        return sample
    
    @staticmethod
    def to_camera_vector(intr_3x3, pose_w2c, W, H): # Return [B,34]=img_size(2) + intrinsics(16) + extrinsics(16)
        """
        BATCHED !!!  
        Input:
        intr_3x3: [3, 3]
        pose_w2c: [B, 3, 4]
        Return: a camera vector (H, W, intr.flatten(), (R|t).flatten())

        """
        device = intr_3x3.device
        intr_3x4 = torch.cat([intr_3x3, torch.tensor([0.,0.,0.])[..., None].to(device)], dim=-1) # [3,4]
        intr_4x4 = torch.cat([intr_3x4, torch.tensor([0.,0.,0.,1.])[None,...].to(device)], dim=-2) # [4,4]
        pose_c2w = camera.Pose().invert(pose_w2c[:3]) # [B, 3, 4]
        pose_4x4_c2w = torch.cat([pose_c2w, 
                                  torch.tensor([0.,0.,0.,1.]).expand(*pose_c2w.shape[:-2], 1, -1).to(device)]
                                  ,dim=-2) # [B, 4, 4]
        
        return torch.concat([torch.tensor([H, W]).expand(*pose_w2c.shape[:-2], -1).to(device),  # [B, 2]
                             intr_4x4.flatten().expand(*pose_w2c.shape[:-2], -1), #[B, 16]
                             pose_4x4_c2w.view(*pose_w2c.shape[:-2], -1)], #[B,16]
                             dim=-1).to(torch.float32) # [B,34]

    def get_image(self, idx):
        fpath = self.list[idx]["file_path"]
        image_fname = f"{self.root}/{fpath}"
        image = Image.open(image_fname)
        image.load()
        image_size_raw = image.size
        return image, image_size_raw

    def preprocess_image(self, image):
        # Resize the image.
        image = image.resize((self.W, self.H))
        image = torchvision_F.to_tensor(image)
        rgb = image[:3]
        return rgb

    def get_camera(self, idx):
        # Camera intrinsics.
        intr = torch.tensor([[self.meta["fl_x"], self.meta["sk_x"], self.meta["cx"]],
                             [self.meta["sk_y"], self.meta["fl_y"], self.meta["cy"]],
                             [0, 0, 1]]).float()
        # Camera pose.
        c2w_gl = torch.tensor(self.list[idx]["transform_matrix"], dtype=torch.float32)
        c2w = self._gl_to_cv(c2w_gl)
        # center scene
        center = np.array(self.meta["sphere_center"])
        center += np.array(getattr(self.readjust, "center", [0])) if self.readjust else 0.
        c2w[:3, -1] -= center
        # scale scene
        scale = np.array(self.meta["sphere_radius"])
        scale *= getattr(self.readjust, "scale", 1.) if self.readjust else 1.
        c2w[:3, -1] /= scale
        w2c = camera.Pose().invert(c2w[:3]) # (..., 3, 4)
        return intr, w2c
    
    def get_nearest_views_ids(self, tar_pose, all_poses, num_select, tar_id):
        """
        Args:
            tar_pose: target pose : [3, 4]
            all_poses: all poses from the training set : [N, 3, 4]
            num_select: the number of nearest views to select
            tar_id: index of target pose in `all_poses`
        Returns: a list of indices in the training set : [num_select, ]
        """
        def angular_dist_between_2_vectors(vec1, vec2):
            """
            计算两个向量之间的cosine夹角
            """
            TINY_NUMBER = 1e-6
            vec1_unit = vec1 / (torch.norm(vec1, dim=1, keepdim=True) + TINY_NUMBER)
            vec2_unit = vec2 / (torch.norm(vec2, dim=1, keepdim=True) + TINY_NUMBER)
            angular_dists = torch.acos(torch.clamp(torch.sum(vec1_unit * vec2_unit, dim=-1), -1.0, 1.0))
            return angular_dists
        
        # get ids of the nearest
        total_num_cams = len(all_poses) # N_train
        num_select = min(num_select, total_num_cams - 1)
        batched_tar_pose = tar_pose[None, ...].expand(total_num_cams, -1, -1) # [N_trainset, 3, 3]

        tar_cam_locs = batched_tar_pose[:, :3, 3] # [N_trainset, 3]
        ref_cam_locs = all_poses[:, :3, 3] # [N_trainset, 3]
        scene_center = torch.tensor((0.,0.,0.))[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)

        if tar_id is not None:
            assert tar_id < total_num_cams 
            dists[tar_id] = 1e3  # make sure not to select the target id itself
        
        sorted_ids = torch.argsort(dists)
        selected_ids = sorted_ids[:num_select]
        return selected_ids


    def preprocess_camera(self, intr, pose, image_size_raw):
        # Adjust the intrinsics according to the resized image.
        intr = intr.clone()
        raw_W, raw_H = image_size_raw
        intr[0] *= self.W / raw_W
        intr[1] *= self.H / raw_H
        return intr, pose

    def _gl_to_cv(self, gl):
        # convert to CV convention used in Imaginaire
        cv = gl * torch.tensor([1, -1, -1, 1])
        return cv

