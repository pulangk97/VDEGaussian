#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from scene.envlight import EnvLight
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.waymo_loader import readWaymoInfo
from scene.kittimot_loader import readKittiMotInfo
import numpy as np
from scene.cameras import Camera
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import time
sceneLoadTypeCallbacks = {
    "Waymo": readWaymoInfo,
    "KittiMot": readKittiMotInfo
}

class Scene:

    gaussians : GaussianModel

    def __init__(self, args, gaussians : GaussianModel, load_iteration=None, shuffle=True):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.all_cameras = {}

        scene_info = sceneLoadTypeCallbacks[args.scene_type](args)
        
        self.time_interval = args.frame_interval
        self.gaussians.time_duration = scene_info.time_duration
        print("time duration: ", scene_info.time_duration)
        print("frame interval: ", self.time_interval)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.resolution_scales = args.resolution_scales
        self.scale_index = len(self.resolution_scales) - 1
        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
            self.all_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.all_cameras, resolution_scale, args)
            
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, 1)

    def upScale(self):
        self.scale_index = max(0, self.scale_index - 1)

    def getTrainCameras(self):
        return self.train_cameras[self.resolution_scales[self.scale_index]]
    
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


    def interpolate_one_view_diff(self, cam1: Camera, cam2: Camera, t):
        """
        修复：与 interpolate_views 保持一致，旋转用 Slerp，平移用线性插值。
        原实现对 world_view_transform 做线性插值，旋转矩阵线性插值不满足正交性，
        与 getPseudoCameras 里用 Slerp 建立的插值相机位姿不一致，导致相机错位。
        """
        from scipy.spatial.transform import Rotation as R_scipy
        from scipy.spatial.transform import Slerp

        t_val = t.detach().cpu().item()  # scalar in [0,1]

        time1 = cam1.timestamp
        time2 = cam2.timestamp
        timestamp = (1 - t_val) * time1 + t_val * time2

        # 平移线性插值（与 interpolate_views 一致）
        T1 = cam1.T  # (3,)
        T2 = cam2.T
        mid_T = (1 - t_val) * T1 + t_val * T2

        # 旋转 Slerp（与 interpolate_views 一致）
        R1 = cam1.R  # (3,3)
        R2 = cam2.R
        key_rots = R_scipy.from_matrix([R1, R2])
        slerp = Slerp([0, 1], key_rots)
        mid_R = slerp([t_val])[0].as_matrix()

        # 重建 w2c 用于渲染（world_view_transform）
        # R 在 Camera 里是转置存储的，w2c[:3,:3] = R.T
        w2c = np.zeros((4, 4))
        w2c[:3, :3] = mid_R.T
        w2c[:3,  3] = mid_T
        w2c[ 3,  3] = 1.0
        wv_inter = torch.tensor(w2c.T, dtype=torch.float32,
                                device=cam1.world_view_transform.device)

        new_cam = Camera(
                                colmap_id=1,
                                uid=1,
                                R=mid_R,
                                T=mid_T,
                                FoVx=cam1.FoVx,
                                FoVy=cam1.FoVy,
                                cx=cam1.cx,
                                cy=cam1.cy,
                                fx=cam1.fx,
                                fy=cam1.fy,
                                image=torch.zeros(1),
                                image_name=cam1.image_name+f"_{t}",
                                data_device=cam1.data_device,
                                timestamp=timestamp,
                                resolution=cam1.resolution,
                                image_path="",
                                pts_depth=torch.zeros(1),
                                sky_mask=torch.zeros(1),
                                image_full_scale=cam1.image_full_scale
                            )
        return new_cam, wv_inter


    def interpolate_views(self, cam1: Camera, cam2: Camera, interp_num):
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp
        T1 = cam1.T
        T2 = cam2.T
        R1 = cam1.R
        R2 = cam2.R

        time1 = cam1.timestamp
        time2 = cam2.timestamp
        interp_cam = []

        t_linear = np.linspace(0,1,interp_num)

        key_times = [0,1]

        key_rots = R.from_matrix([R1,R2])
        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(t_linear)
        for i in range(interp_num):

            mid_T = (1-t_linear[i])*T1 + t_linear[i]*T2
            mid_rots = interp_rots[i]

            mid_R = mid_rots.as_matrix()

            timestamp = (1-t_linear[i])*time1 + t_linear[i]*time2


            interp_cam.append(Camera(
                                colmap_id=i,
                                uid=i,
                                R=mid_R,
                                T=mid_T,
                                FoVx=cam1.FoVx,
                                FoVy=cam1.FoVy,
                                cx=cam1.cx,
                                cy=cam1.cy,
                                fx=cam1.fx,
                                fy=cam1.fy,
                                image=torch.zeros(1),
                                image_name=cam1.image_name+f"_{i}",
                                data_device=cam1.data_device,
                                timestamp=timestamp,
                                resolution=cam1.resolution,
                                image_path="",
                                pts_depth=torch.zeros(1),
                                sky_mask=torch.zeros(1),
                                image_full_scale=cam1.image_full_scale
                            ))
            
        return interp_cam

    def getPseudoCameras(self, scale, num_interpolate = 16, cam_num=1, total_cam=5):
        # colmap_id 硬编码为 idx * 5 + j，帧索引换算固定用 5
        CAM_STRIDE = 5

        resolution_scale = scale

        # 获取所有训练相机的 colmap_id，用于快速查找
        train_cam_dict = {cam.colmap_id: cam for cam in self.train_cameras[resolution_scale]}
        test_cams = self.test_cameras[resolution_scale]

        # 按帧索引（colmap_id // CAM_STRIDE）将测试相机分组，得到各测试块的帧索引集合
        # 每个块只取一对左右边界帧，生成一个伪帧
        test_frame_indices = sorted(set(cam.colmap_id // CAM_STRIDE for cam in test_cams))

        # 找连续块：将连续的帧索引聚合为块，每块取第一帧和最后一帧作为边界
        blocks = []
        if test_frame_indices:
            block_start = test_frame_indices[0]
            block_end   = test_frame_indices[0]
            for fi in test_frame_indices[1:]:
                if fi == block_end + 1:
                    block_end = fi
                else:
                    blocks.append((block_start, block_end))
                    block_start = block_end = fi
            blocks.append((block_start, block_end))

        views = []
        for (blk_start, blk_end) in blocks:
            # 左条件帧：块左边界的前一帧（在训练集中），取第0路相机
            # 右条件帧：块右边界的后一帧（在训练集中），取第0路相机
            left_frame_idx  = blk_start - 1
            right_frame_idx = blk_end   + 1

            left_colmap_id  = left_frame_idx  * CAM_STRIDE
            right_colmap_id = right_frame_idx * CAM_STRIDE

            left_cam  = train_cam_dict.get(left_colmap_id)
            right_cam = train_cam_dict.get(right_colmap_id)

            if left_cam is None or right_cam is None:
                print(f"Warning: boundary cam not found for block ({blk_start},{blk_end}), "
                      f"left_colmap_id={left_colmap_id}, right_colmap_id={right_colmap_id}, skipping.")
                continue

            interp_cam = self.interpolate_views(left_cam, right_cam, num_interpolate)

            view = {}
            view["leftcam"]   = left_cam
            view["rightcam"]  = right_cam
            view["interpcam"] = interp_cam
            views.append(view)

        return views


    

    def getPseudoImage(self, model, num_interpolate=16, cam_num=1, save_dir=None):

        video_size = (320, 512)
        transform = transforms.Compose([
            transforms.Resize(min(video_size)),
            transforms.CenterCrop(video_size)])

        if save_dir is not None:
            import os
            from torchvision.utils import save_image
            os.makedirs(save_dir, exist_ok=True)

        pseudoCameras = self.getPseudoCameras(scale=max(self.resolution_scales), num_interpolate=num_interpolate, cam_num=cam_num)
        batch_images = []
        batch_latents = []
        n=0
        for cam in tqdm(pseudoCameras):

            if num_interpolate%2==0:
                input_batch = torch.concat([torch.concat([cam["leftcam"].image_full_scale[None,...].to(cam["leftcam"].data_device)]*int(num_interpolate/2), dim=0), torch.concat([cam["rightcam"].image_full_scale[None,...].to(cam["rightcam"].data_device)]*int(num_interpolate/2), dim=0)], dim=0)
            else:
                input_batch = torch.concat([torch.concat([cam["leftcam"].image_full_scale[None,...].to(cam["leftcam"].data_device)]*int(np.floor(num_interpolate/2)), dim=0), torch.concat([cam["rightcam"].image_full_scale[None,...].to(cam["rightcam"].data_device)]*int(np.ceil(num_interpolate/2)), dim=0)], dim=0)
            input_batch = input_batch.permute(1,0,2,3)[None,...]

            left_frame_idx  = cam["leftcam"].colmap_id  // 5
            right_frame_idx = cam["rightcam"].colmap_id // 5
            frame_gap = right_frame_idx - left_frame_idx  
            infer_fs = 8
            batch_image, batch_latent = model.inference(input_batch.to(model.device), fs=infer_fs)

            batch_image[0][0,:,0,...] = (transform(cam["leftcam"].image_full_scale)*2-1)
            batch_image[0][0,:,2,...] = (transform(cam["rightcam"].image_full_scale)*2-1)

            if save_dir is not None:
                pseudo_frame = batch_image[0][0, :, 1, :, :]
                pseudo_frame = (pseudo_frame.clamp(-1, 1) + 1) / 2
                left_name  = cam["leftcam"].image_name
                right_name = cam["rightcam"].image_name
                save_image(pseudo_frame, os.path.join(
                    save_dir, f"{left_name}_{right_name}_pseudo.png"))

            n=n+1
            batch_images.append(batch_image)
            batch_latents.append(batch_latent)

        return batch_images, batch_latents