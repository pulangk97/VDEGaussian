from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import random
def get_all_train_pairs(num_img, test_list):
     pair_list = []
     for i in range(num_img):
        left_idx, right_idx = i-1, i+1
        if not (i-1<0 or i+1>=num_img or ((i-1) in test_list) or ((i) in test_list) or ((i+1) in test_list)):
            pair_list.append([left_idx, i, right_idx])
     return pair_list

class Waymo(Dataset):
    """
    Waymo Dataset.

    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=3,
                 resolution=[512, 512],
                 frame_stride=2,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 down_rate = 0.5,
                 cam_num = 3,
                 testhold = 4 # -1
                 ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.testhold = testhold

        self.cam_num = cam_num

        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None


        
        car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(self.data_dir, "calib"))) if f.endswith('.txt')]

        transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),self.spatial_transform]) #outptu tensor in [-1,1]

        images_all = []
        for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):

            images = []
            image_paths = []
            HWs = []
            for subdir in ['image_0', 'image_1', 'image_2', 'image_3', 'image_4'][:self.cam_num]:

                if os.path.exists(os.path.join(self.data_dir, subdir, car_id + '.jpg')):
                    image_path = os.path.join(self.data_dir, subdir, car_id + '.jpg')
                elif os.path.exists(os.path.join(self.data_dir, subdir, car_id + '.png')):
                    image_path = os.path.join(self.data_dir, subdir, car_id + '.png')
                else:
                    raise NotImplementedError

                im_data = Image.open(image_path)

                W, H = im_data.size
                image = np.array(im_data) / 255.

                HWs.append((H, W))
                images.append(transform(torch.tensor(image,dtype=torch.float32)[...,:3].permute(2,0,1)).numpy())
                image_paths.append(image_path)

            images_all.append(images)

        num_img = len(images_all)

        if self.testhold == -1:
            test_idx = []
        else:
            test_idx = [idx for idx in range(num_img) if (idx) % self.testhold == 0 and (idx)>0]
        seed = 114514
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)



        self.pair_idx = torch.tensor(np.stack(get_all_train_pairs(num_img, test_idx)),dtype=int)


        self.scene = self.data_dir.split("/")[-2]

        self.imgs = torch.tensor(np.stack(images_all))


    
    def __getitem__(self, index):

        cam_idx = torch.randperm(self.cam_num)
        idx = self.pair_idx[index]

        frames = self.imgs[idx,cam_idx[0],...].squeeze().permute(1,0,2,3)

        caption = f"a driving scene"
        video_path = ""


        fps_ori = 10
        frame_stride = self.frame_stride

        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max


        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride}
        return data
    
    def __len__(self):

        return len(self.pair_idx)


if __name__== "__main__":


    meta_path =  None ## path to the meta file
    data_dir = "" ## path to the data directory
    save_dir = "" ## path to the save directory
    dataset = Waymo(meta_path,
                 data_dir,
                 subsample=None,
                 video_length=3,
                 resolution=[400,400],
                 frame_stride=1,
                 spatial_transform="resize_center_crop",
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=True
                 )
    dataloader = DataLoader(dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=False)

    
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
    from utils.save_video import tensor_to_mp4
    for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):

        video = batch['video']


