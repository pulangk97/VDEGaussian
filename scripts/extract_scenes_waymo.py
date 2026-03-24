import os
from os.path import join
from tqdm import tqdm

data_root = './data/waymo/kitti_format/training'

tags = ['image_0','image_1','image_2','image_3','image_4','calib','velodyne','pose']
posts = ['.jpg','.jpg','.jpg','.jpg','.jpg','.txt', '.bin','.txt']

out_dir = 'data/waymo_scenes'

scene_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
scene_nums = [
    [0, 198],
    [0, 196],
    [0, 198],
    [0, 198],
    [0, 198],
    [0, 197],
    [0, 198],
    [0, 198],
    [0, 198],
    [0, 197],
    [0, 197],
    [0, 197],
    [0, 198],
    [0, 196],
    [0, 196],
    [0, 198],
    [0, 196],
    [0, 197],
    [0, 196],
    [0, 198],
    [0, 197],
    [0, 197],
    [0, 197],
    [0, 198],
    [0, 198],
    [0, 197],
    [0, 197],
    [0, 198],
    [0, 198],
    [0, 198],
    [0, 197],
    [0, 198],
]
os.makedirs(out_dir, exist_ok=True)

for scene_idx, scene_id in enumerate(scene_ids):
    scene_dir = join(out_dir, f'{scene_id:04d}001')
    os.makedirs(scene_dir, exist_ok=True)

    for tag in tags:
        os.makedirs(join(scene_dir, tag), exist_ok=True)
    for post, tag in zip(posts,tags):
        for i in tqdm(range(scene_nums[scene_idx][0], scene_nums[scene_idx][1])):
            cmd = "cp {} {}".format(join(data_root,tag,f'{scene_id:04d}{i:03d}'+post), 
                                    join(scene_dir, tag, f'{scene_id:04d}{i:03d}'+post))
            os.system(cmd)



