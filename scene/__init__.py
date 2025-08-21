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


        time1 = cam1.timestamp
        time2 = cam2.timestamp


        wv1 = cam1.world_view_transform
        wv2 = cam2.world_view_transform



        wv_inter = (1-t)*wv1 + t*wv2
        

        timestamp = (1-t.detach().cpu())*time1 + t.detach().cpu()*time2

        wv_inter_trans = wv_inter.transpose(0, 1)

        new_cam = Camera(
                                colmap_id=1,
                                uid=1,
                                R=wv_inter_trans[:3,:3].detach().cpu().numpy().transpose(),
                                T=wv_inter_trans[:3,-1].detach().cpu().numpy(),
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

        resolution_scale = scale
        num_cams = int(len(self.all_cameras[2])/cam_num*total_cam)

        test_idx = np.array([i.colmap_id for i in self.test_cameras[resolution_scale]])
        surrund_idx = np.stack([np.clip(test_idx-total_cam,0,num_cams), np.clip(test_idx+total_cam,0,num_cams)])

        views=[]

        for i in range(surrund_idx.shape[-1]):
            
            left_cam = [cam for cam in self.train_cameras[resolution_scale] if cam.colmap_id == surrund_idx[0,i]]
            right_cam = [cam for cam in self.train_cameras[resolution_scale] if cam.colmap_id == surrund_idx[1,i]]

            if len(left_cam) == 1 and len(right_cam) == 1:
                left_cam = left_cam[0]
                right_cam = right_cam[0]
            else:
                assert False, print("cam sum error") 
            interp_cam = self.interpolate_views(left_cam, right_cam, num_interpolate)
                
            view = {}
            view["leftcam"] = left_cam
            view["rightcam"] = right_cam
            view["interpcam"] = interp_cam
            views.append(view)

        return views


    

    def getPseudoImage(self, model, num_interpolate=16, cam_num=1):


        video_size = (320, 512)
        transform = transforms.Compose([
            transforms.Resize(min(video_size)),
            transforms.CenterCrop(video_size)])


        pseudoCameras = self.getPseudoCameras(scale=16, num_interpolate=num_interpolate, cam_num=cam_num)
        batch_images = []
        batch_latents = []
        n=0
        for cam in tqdm(pseudoCameras):

            if num_interpolate%2==0:

                input_batch = torch.concat([torch.concat([cam["leftcam"].image_full_scale[None,...].to(cam["leftcam"].data_device)]*int(num_interpolate/2), dim=0), torch.concat([cam["rightcam"].image_full_scale[None,...].to(cam["rightcam"].data_device)]*int(num_interpolate/2), dim=0)], dim=0)
            else:
                input_batch = torch.concat([torch.concat([cam["leftcam"].image_full_scale[None,...].to(cam["leftcam"].data_device)]*int(np.floor(num_interpolate/2)), dim=0), torch.concat([cam["rightcam"].image_full_scale[None,...].to(cam["rightcam"].data_device)]*int(np.ceil(num_interpolate/2)), dim=0)], dim=0)
            input_batch = input_batch.permute(1,0,2,3)[None,...]


            batch_image, batch_latent = model.inference(input_batch.to(model.device))


            batch_image[0][0,:,0,...] = (transform(cam["leftcam"].image_full_scale)*2-1)
            batch_image[0][0,:,2,...] = (transform(cam["rightcam"].image_full_scale)*2-1)

            n=n+1

            batch_images.append(batch_image)
            batch_latents.append(batch_latent)

        return batch_images, batch_latents


