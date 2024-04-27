
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF

from robust.configs.dotdict_class import DotDict
from robust.networks.vision_transformer import NanViTModel
from robust.networks.autoencoder import NoiseVAE
from robust.projection import Projector 
from robust.robust_dataset import Dataset as RobustDataset
from projects.neuralangelo.model import Model as NeuraModel

from projects.nerf.utils import nerf_util, camera, render


# from robust.projector.Projector
def inbound(pixel_locations, h, w):
        """
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        """
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) & \
               (pixel_locations[..., 1] >= 0)

def normalize_pixel_locations(pixel_locations, h, w):
        resize_factor = torch.tensor([w - 1., h - 1.]).to(pixel_locations.device)[None, None, :] # [1, 1, 2]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

def pixel_location_expander_factory(kernel_size):
        kx, ky = kernel_size
        assert kx % 2 == 1 and ky % 2 == 1

        x_range = torch.arange(-(kx // 2), kx // 2 + 1)
        y_range = torch.arange(-(ky // 2), ky // 2 + 1)
        # create a 3D matrix of (delta_x, delta_y), i.e. the coordinate offset to iterate through the neighborhood  
        kernel_expander = torch.stack(torch.meshgrid(x_range, y_range, indexing='ij')[::-1]).permute((1, 2, 0))

        def expander(pixel_locations): 
            '''
            Given N incident pixel locations, for each of them, compute the k*k neighbor locations around it

            Arguments
            ==========
            pixel_locations: Tensor(..., 2)
            Returns
            ==========
            Tensor(..., k , k, 2)
            '''
            expanded_pixels = pixel_locations.unsqueeze(-2).unsqueeze(-2) + kernel_expander.to(pixel_locations.device)
            return expanded_pixels

        def reshape_features(pixel_locations):
            '''
            Reshape the pixel_locations to (...,k,k,f) 
            Arguments
            ---------
            pixel_locations: Tensor of some dimensions
            Returns
            ---------
            Tensor (...,k,k,f) where f is for the feature embedding 
            '''
            return pixel_locations.reshape((pixel_locations.shape[0],) +
                                           (-1,) + kernel_size + pixel_locations.shape[2:])

        return expander, reshape_features


class Model(NeuraModel):
    def __init__(self, cfg_model, cfg_data):
        super().__init__(cfg_model, cfg_data)
        self.local_args = {
             "kernel_size": (3,3)} # kernel size for 3D reprojection 
        
        # TODO: tweak the ViTModel to make the final output [B,R,S,256]
        RAY_TRANSFOMRE_CFG = DotDict({
        "num_encoder_block": 1,
        "n_head": 3,    
        "embedding_size": math.prod(self.local_args["kernel_size"])*3, # k*k*3
        "final_out_size": 3, # controls the output size at the end of the whole transformer 
        "kernel_size": self.local_args["kernel_size"], # (optional) only when using RTMLP, the kernel size in feature extraction step
        "seq_length": 4, #(required for nan) the N_SRC a.k.a. number of views in feature extraction step
        "model_k_size": 9,
        "model_v_size": 9,
        "mlp_dim": 256,
        "dropout_rate": 0.1,
        "mlp_type": "ViTMLP"
        })
        self.ray_transformer = NanViTModel(RAY_TRANSFOMRE_CFG) 
        self.noise_vae = NoiseVAE(3, 128, [64, 128, 128,])
        self.expander, self.reshape_features = pixel_location_expander_factory(kernel_size=self.local_args['kernel_size'])
        
       
    def forward(self, data):
        
        w2c_tar_pose = data["pose"] # [B, 3, 4]
        intr = data["intr"][0]
        image_size = self.image_size_train
        sample_idx=data["idx"]
        ray_idx = data["ray_idx"] # [R,]
        stratified=self.cfg_render.stratified
        center, ray = camera.get_center_and_ray(w2c_tar_pose, intr, image_size)  # [B,HW,3]
        center = nerf_util.slice_by_ray_idx(center, ray_idx)  # [B,R,3]
        ray = nerf_util.slice_by_ray_idx(ray, ray_idx)  # [B,R,3]
        ray_unit = F.normalize(ray, dim=-1)  # [B,R,3]
        # render rays
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(center, ray_unit)
        app, app_outside = self.get_appearance_embedding(sample_idx, ray_unit.shape[1]) # [B,R,N,C], [B,R,N,C]
        # app=None, app_outside=None
        # render rays object 
        with torch.no_grad():
            dists = self.sample_dists_all(center, ray_unit, near, far, stratified=stratified)  # [B, R, N=128, 1]
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        ##################### NOISE AWARE MODULE #####################
        # Compute neighbor view RGB features and masks  
        tar_cam_vec = RobustDataset.to_camera_vector(intr, data["pose"], image_size[0], image_size[1] ) #[B, 34]      
        batched_neighbor_rgb, batched_neighbor_mask = [], []
        for _b in range(data["pose"].shape[0]): # batch index
            neighbor_rgb, neighbor_mask = self.projector_compute(data['neighbor_rgbs'][_b], data['neighbor_poses'][_b], tar_cam_vec[_b], xyz=points[_b])
            # neighbor_rgb [n_rays, n_samples, kernel_x, kernel_y, N_SRC, 3]
            # neighbor_mask [n_rays, n_samples, N_SRC, 1]<boolean>
            neighbor_rgb = neighbor_rgb.permute(0,1,4,2,3,5) # [n_rays, n_samples, N_SRC, kernel_x, kernel_y, 3]
            # To make sure the RGBs from invalid neighbor view is all 0
            neighbor_rgb *= neighbor_mask.unsqueeze(-1).unsqueeze(-1).expand(*neighbor_rgb.shape) # [n_rays, n_samples, N_SRC, kernel_x, kernel_y, 3]
            batched_neighbor_rgb.append(neighbor_rgb.unsqueeze(0))
            # batched_neighbor_mask.append(neighbor_mask)
        
        
        batched_neighbor_rgb = torch.cat(batched_neighbor_rgb, dim=0) # [n_batch, n_rays, n_samples, N_SRC, kernel_x, kernel_y, 3]
        batched_neighbor_mean = torch.mean(batched_neighbor_rgb, dim=(-4,-3,-2), keepdim=True) # [B, R, S, 1, 1, 1, 3]
        batched_neighbor_variance = torch.mean((batched_neighbor_rgb - batched_neighbor_mean)**2, dim=(-4,-3,-2), keepdim=True) # [B, R, S, 1, 1, 1, 3]
        n_batch, n_rays, n_samples, n_views ,_ ,_ , _ = batched_neighbor_rgb.shape
        """ Design 1: merge [N_SRC, k, k] into one single dimension """
        #  batched_neighbor_rgb = batched_neighbor_rgb.reshape((n_batch, n_rays, n_samples, -1, 3)) # [n_batch, n_rays, n_samples, N_SRC *kernel_x *kernel_y, 3]
        """Design 2: merge [k,k,3] into one single dimension """
        batched_neighbor_rgb = batched_neighbor_rgb.reshape((n_batch, n_rays, n_samples, n_views, -1)) # # [n_batch, n_rays, n_samples, N_SRC, *kernel_x *kernel_y *3]
        # rt_out, _ = self.ray_transformer(batched_neighbor_rgb) #[B,R,S,3]
        ############### END NOISE AWARE MODULE #################
        
        sdfs, sdf_feats = self.neural_sdf.forward(points)  # [B,R,S,1],[B,R,S,K=256]
        # outside [B=2, R=512, 1]<boolean>
        sdfs[outside[..., None].expand_as(sdfs)] = self.outside_val # [B, R, N, 1]
        # Compute 1st- and 2nd-order gradients.
        rays_unit = ray_unit[..., None, :].expand_as(points).contiguous()  # [B,R,N,3]
        gradients, hessians = self.neural_sdf.compute_gradients(points, training=self.training, sdf=sdfs) #[B, R, N, 3], None
        normals = F.normalize(gradients, dim=-1)  # [B,R,N,3]
        rgbs = self.neural_rgb.forward(points, normals, rays_unit, sdf_feats, app=app)  # [B,R,N,3] 
        
        ##### DEBUG 2024-4-27
        ray_src_image = data["image_sampled"] # [B,R,3]
        batched_neighbor_mean.squeeze_(dim=(-4,-3,-2)) # [B, R, S, 3]
        batched_neighbor_variance.squeeze_(dim=(-4,-3,-2)) # [B, R, S, 3]
        
        vae_latent = self.noise_vae.encode(VF.crop(data['neighbor_rgbs'].flatten(start_dim=0, end_dim=1),
                                             top=torch.randint(0, data['neighbor_rgbs'].shape[-2]-400, (1,)).item(),
                                             left=torch.randint(0, data['neighbor_rgbs'].shape[-1]-400, (1,)).item(),
                                             width=400,
                                             height=400)
                                    ) # input must be [B, C, H, W]
        vae_out = self.noise_vae.add_noise(rgbs, vae_latent) #[B,R,S,3] # 
        ##### DEBUG 2024-4-27

        rgbs = vae_out 
        # SDF volume rendering.
        alphas = self.compute_neus_alphas(ray_unit, sdfs, gradients, dists, dist_far=far[..., None],
                                          progress=self.progress)  # [B,R,N]
        if not self.training:
            weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
            opacity = render.composite(1., weights)  # [B,R,1]
            gradient = render.composite(gradients, weights)  # [B,R,3]
        else:
            opacity = None
            gradient = None
        
        # output_object = self.render_rays_object(center, ray_unit, near, far, outside, app, stratified=stratified)
        ### FROM HERE ON######
        if self.with_background:
            output_background = self.render_rays_background_nan(center, ray_unit, far, app_outside, 
                                                                vae_latent,
                                                                neighbor_rgbs=data['neighbor_rgbs'],
                                                                neighbor_poses=data['neighbor_poses'],
                                                                tar_cam_vec=tar_cam_vec,
                                                                stratified=stratified)
            # Concatenate object and background samples.
            # output_background["rgbs"]: [B, R, Nb=32, 3]
            rgbs = torch.cat([rgbs, output_background["rgbs"]], dim=2)  # [B,R,No+Nb,3]
            dists = torch.cat([dists, output_background["dists"]], dim=2)  # [B,R,No+Nb,1]
            alphas = torch.cat([alphas, output_background["alphas"]], dim=2)  # [B,R,No+Nb]
        else:
            pass
        weights = render.alpha_compositing_weights(alphas)  # [B,R,No+Nb=160,1]
        # Compute weights and composite samples.
        rgb = render.composite(rgbs, weights)  # [B,R,3] <=== Volume Rendering!!
        if self.white_background:
            opacity_all = render.composite(1., weights)  # [B,R,1]
            rgb = rgb + (1 - opacity_all)
        # Collect output.
        output = dict(
            rgb=rgb,  # [B,R,3]
            opacity=opacity,  # [B,R,1]/None
            outside=outside,  # [B,R,1]
            dists=dists,  # [B,R,No+Nb,1]
            weights=weights,  # [B,R,No+Nb,1]
            gradient=gradient,  # [B,R,3]/None
            gradients=gradients,  # [B,R,No,3]
            hessians=hessians,  # [B,R,No,3]/None
        )
        return output
    
    """Copied from projects.neurlangelo.model::Model.render_rays_background"""
    def render_rays_background_nan(self, center, ray_unit, far, app_outside,
                                   vae_latent,
                                   neighbor_rgbs, neighbor_poses, tar_cam_vec, stratified=False, is_inference=False):
        """
        Render rays of background pixels. Added ray transformer logic

        Additional params:
            neighbor_rgbs: [batch_size, n_views, 3, h, w]
            tar_cam_vec: [batch_size, 34]
            neighbor_poses: [batch_size, n_views, 34]
        """
        with torch.no_grad():
            dists = self.sample_dists_background(ray_unit, far, stratified=stratified)
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        rays_unit = ray_unit[..., None, :].expand_as(points)  # [B,R,N,3]
        rgbs, densities = self.background_nerf.forward(points, rays_unit, app_outside)  # [B,R,N,3]
        ############ BEGIN RAY TRANSFORMER ##########
        if not is_inference:
            batched_neighbor_rgb = []
            for _b in range(neighbor_rgbs.shape[0]):
                neighbor_rgb, neighbor_mask = self.projector_compute(neighbor_rgbs[_b], neighbor_poses[_b], tar_cam_vec[_b], xyz=points[_b])
                neighbor_rgb = neighbor_rgb.permute(0,1,4,2,3,5)
                neighbor_rgb *= neighbor_mask.unsqueeze(-1).unsqueeze(-1).expand(*neighbor_rgb.shape)
                batched_neighbor_rgb.append(neighbor_rgb.unsqueeze(0))
            batched_neighbor_rgb = torch.cat(batched_neighbor_rgb, dim=0)
            n_batch, n_rays, n_samples, n_views ,_ ,_ , _ = batched_neighbor_rgb.shape
            batched_neighbor_rgb = batched_neighbor_rgb.reshape((n_batch, n_rays, n_samples, n_views, -1)) 
            # rt_out, _ = self.ray_transformer(batched_neighbor_rgb)
        # 暂时解除vae
        rgbs = self.noise_vae.add_noise(rgbs, vae_latent)
        ############ END RAY TRANSFORMER ##########
        alphas = render.volume_rendering_alphas_dist(densities, dists)  # [B,R,N]
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,3]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
        )
        return output


    def render_pixels(self, pose, intr, image_size, stratified=False, sample_idx=None, ray_idx=None):
        return super().render_pixels(pose, intr, image_size, stratified, sample_idx, ray_idx)


    def render_rays(self, center, ray_unit, sample_idx=None, stratified=False, is_inference=False):
        # Will be called during inference
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(center, ray_unit)
        app, app_outside = self.get_appearance_embedding(sample_idx, ray_unit.shape[1]) # [B,R,N,C], [B,R,N,C]
        output_object = self.render_rays_object(center, ray_unit, near, far, outside, app, stratified=stratified)
        if self.with_background:
            output_background = self.render_rays_background_nan(center, ray_unit, far, app_outside, stratified=stratified, 
                                                                neighbor_rgbs=None, neighbor_poses=None,tar_cam_vec=None,
                                                                is_inference=True)
            # Concatenate object and background samples.
            # output_background["rgbs"]: [B, R, Nb=32, 3]
            rgbs = torch.cat([output_object["rgbs"], output_background["rgbs"]], dim=2)  # [B,R,No+Nb,3]
            dists = torch.cat([output_object["dists"], output_background["dists"]], dim=2)  # [B,R,No+Nb,1]
            alphas = torch.cat([output_object["alphas"], output_background["alphas"]], dim=2)  # [B,R,No+Nb]
        else:
            rgbs = output_object["rgbs"]  # [B,R,No,3]
            dists = output_object["dists"]  # [B,R,No,1]
            alphas = output_object["alphas"]  # [B,R,No]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,No+Nb=160,1]
        # Compute weights and composite samples.
        rgb = render.composite(rgbs, weights)  # [B,R,3] <=== Volume Rendering!!
        if self.white_background:
            opacity_all = render.composite(1., weights)  # [B,R,1]
            rgb = rgb + (1 - opacity_all)
        # Collect output.
        output = dict(
            rgb=rgb,  # [B,R,3]
            opacity=output_object["opacity"],  # [B,R,1]/None
            outside=outside,  # [B,R,1]
            dists=dists,  # [B,R,No+Nb,1]
            weights=weights,  # [B,R,No+Nb,1]
            gradient=output_object["gradient"],  # [B,R,3]/None
            gradients=output_object["gradients"],  # [B,R,No,3]
            hessians=output_object["hessians"],  # [B,R,No,3]/None
        )
        return output
        # raise NotImplementedError() # TODO: if you manipulated rgb in the forward(), you should do the same here! 
        
        
    """Override"""
    def render_rays_object(self, center, ray_unit, near, far, outside, app, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_all(center, ray_unit, near, far, stratified=stratified)  # [B, R, N=128, 1]
        # dists [B, R, N=128, 1]
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        sdfs, feats = self.neural_sdf.forward(points)  # [B,R,N,1],[B,R,N,K=256]
        # outside [B=2, R=512, 1]<boolean>
        sdfs[outside[..., None].expand_as(sdfs)] = self.outside_val # [B, R, N, 1]
        # Compute 1st- and 2nd-order gradients.
        rays_unit = ray_unit[..., None, :].expand_as(points).contiguous()  # [B,R,N,3]
        gradients, hessians = self.neural_sdf.compute_gradients(points, training=self.training, sdf=sdfs) #[B, R, N, 3], None
        normals = F.normalize(gradients, dim=-1)  # [B,R,N,3]
        rgbs = self.neural_rgb.forward(points, normals, rays_unit, feats, app=app)  # [B,R,N,3] app=None (appearance embedding)
        ###### ADD VAE #####
        vae_out = self.noise_vae.add_noise(rgbs)
        #################
        rgbs = vae_out
        # SDF volume rendering.
        alphas = self.compute_neus_alphas(ray_unit, sdfs, gradients, dists, dist_far=far[..., None],
                                          progress=self.progress)  # [B,R,N]
        if not self.training:
            weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
            opacity = render.composite(1., weights)  # [B,R,1]
            gradient = render.composite(gradients, weights)  # [B,R,3]
        else:
            opacity = None
            gradient = None
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,N,3]
            sdfs=sdfs[..., 0],  # [B,R,N]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
            opacity=opacity,  # [B,R,3]/None
            gradient=gradient,  # [B,R,3]/None
            gradients=gradients,  # [B,R,N,3]
            hessians=hessians,  # [B,R,N,3]/None
        )
        return output
      
    
    def projector_compute(self, neighbor_rgbs, neighbor_poses, tar_pose, xyz):
        """
        Given 3D points and the camera of the target and src views,
            computing the rgb values of the incident pixels and their kxk environment.
        Input
        neighbor_rgbs: [n_views, 3, h, w]
        neighbor_poses: [n_views, 34]
        tar_pose: [34, ] the pose of the target view (that we are sampling rays on)
        xyz: [R, S, 3] 3D points along the rays 

        Returns
        org_rgbs_sampled: [n_rays, n_samples, kernel_size_x, kernel_size_y, n_views, 3]
        mask: [n_rays, n_views, n_samples, 1]<boolean> the 3D point is invalid if it is behind 
                                    the camera or if the its projected location falls outside the a frame area
        """

        src_imgs       =  neighbor_rgbs # [n_views, 3, h, w]

        org_src_imgs   =   neighbor_rgbs # [n_views, 3, h, w]

        query_camera =  tar_pose # [34, ]
        src_cameras = neighbor_poses

        # compute the projection of the query points to each reference image
        xys, mask = self.compute_projections(xyz, src_cameras)  # xys: [n_views, n_rays, S * prod(kernel_size), 2]<float32>. mask: [n_rays, n_views, n_samples]<boolean>
        h, w = src_cameras[0][:2] # scalar, scalar. The height and width of input image
        norm_xys = normalize_pixel_locations(xys, h, w)  # [n_views, n_rays, S * prod(kernel_size), 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(src_imgs, norm_xys, align_corners=True) # [n_views, 3, n_rays, S*prod(kernel_size)]
        rgbs_sampled = rgbs_sampled.permute(2, 3, 0, 1)  # [n_rays, S * prod(kernel_size), n_views, 3]

        org_rgbs_sampled = F.grid_sample(org_src_imgs, norm_xys, align_corners=True) #[n_views, 3, n_rays, S * prod(kernel_size)]
        org_rgbs_sampled = org_rgbs_sampled.permute(2, 3, 0, 1)  # [n_rays, S * prod(kernel_size), n_views, 3]
        org_rgbs_sampled = self.reshape_features(org_rgbs_sampled) # [n_rays, S, kernel_size_x, kernel_size_y, n_views, 3]

        # mask
        mask = mask.permute(1, 2, 0)[..., None]  # [n_rays, n_samples, n_views, 1]
        return org_rgbs_sampled, mask
    
    def compute_projections(self, xyz, src_cameras):
        """
        project 3D points into cameras
        :param xyz: [n_rays, n_samples, 3]   
        :param src_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        """
        device = xyz.device
        n_rays, n_samples = xyz.shape[:2]
        xyz               = xyz.reshape(-1, 3)  # [n_points, 3], n_points = n_rays * n_samples
        xyz_h             = torch.cat([xyz, torch.ones_like(xyz[..., :1]).to(device)], dim=-1)  # [n_points, 4]

        num_views      = len(src_cameras)
        h, w           = src_cameras[0][:2]
        src_intrinsics = src_cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
        src_poses      = src_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]

        projections = src_intrinsics.bmm(torch.inverse(src_poses)) \
            .bmm(xyz_h.t()[None, ...].expand(num_views, 4, xyz.shape[0]))  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        mask_in_front = projections[..., 2] > 0  # a point is invalid if behind the camera,  [n_views, n_points]

        uv = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
        uv = torch.clamp(uv, min=-1e6, max=1e6)
        mask_inbound = inbound(uv, h, w)  # a point is invalid if out of the image
        uv = self.expander(uv)  # [n_views, n_points, kernel_size, 2]

        mask = mask_in_front * mask_inbound

        # split second dimension (n_points) into 2 dimensions: (n_rays, n_samples):
        # uv (because of F.grid_sample): [n_views, n_points, k, k, 2]  -->  [n_views, n_rays, n_samples*k*k, 2]
        # mask                         : [n_views, n_points]           -->  [n_views, n_rays, n_samples]
        k_by_k = math.prod(self.local_args['kernel_size'])
        return uv.view((num_views, n_rays, n_samples * k_by_k, 2)), mask.view((num_views, n_rays, n_samples))
