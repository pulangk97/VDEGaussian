import numpy as np
import torch
import os
import math
from utils.graphics_utils import focal2fov
# from lib.datasets.base_readers import CameraInfo
from PIL import Image
from tqdm import tqdm


def to_cuda(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = [to_cuda(b) for b in batch]
        return batch
    elif isinstance(batch, torch.Tensor):
        return batch.cuda()
    elif isinstance(batch, np.ndarray):
        return torch.from_numpy(batch).cuda()
    elif isinstance(batch, dict):
        for k in batch:
            if k == "meta":
                continue
            batch[k] = to_cuda(batch[k])
        return batch
    else:
        raise NotImplementedError


def get_split_data(split_train, split_test, data):
    if split_train != -1:
        train_data = [d for idx, d in enumerate(data) if idx % split_train == 0]
        test_data = [d for idx, d in enumerate(data) if idx % split_train != 0]
    else:
        train_data = [d for idx, d in enumerate(data) if idx % split_test != 0]
        test_data = [d for idx, d in enumerate(data) if idx % split_test == 0]
    return train_data, test_data


def get_val_frames(num_frames: int, test_every: int, train_every: int):
    if train_every is None or train_every < 0:
        val_frames = set(np.arange(test_every, num_frames, test_every))
        train_frames = (set(np.arange(num_frames)) - val_frames) if test_every > 1 else set()
    else:
        train_frames = set(np.arange(0, num_frames, train_every))
        val_frames = (set(np.arange(num_frames)) - train_frames) if train_every > 1 else set()

    train_frames = sorted(list(train_frames))
    val_frames = sorted(list(val_frames))

    return train_frames, val_frames

def is_test_frame_multi(frame_idx, test_interval=10, test_offsets=[0], skip_first=False):
    """
    判断某一帧是否为测试帧（支持每间隔抽多帧，固定位置）
    
    Args:
        frame_idx: 当前帧索引
        test_interval: 测试间隔（每多少帧为一个块）
        test_offsets: 测试帧在每个块中的位置列表（0-based）
                     例如：[0, 5] 表示每个块的第0和第5帧为测试帧
        skip_first: 是否跳过第0帧（EmerNeRF兼容模式）
    
    Returns:
        bool: True=测试帧, False=训练帧
    
    Examples:
        # 每10帧抽2帧（位置3和7）
        >>> is_test_frame_multi(3, test_interval=10, test_offsets=[3, 7])
        True
        >>> is_test_frame_multi(7, test_interval=10, test_offsets=[3, 7])
        True
        >>> is_test_frame_multi(13, test_interval=10, test_offsets=[3, 7])
        True  # 13 % 10 = 3
        
        # EmerNeRF模式：每10帧抽1帧（位置0），跳过第0帧
        >>> is_test_frame_multi(0, test_interval=10, test_offsets=[0], skip_first=True)
        False  # 跳过
        >>> is_test_frame_multi(10, test_interval=10, test_offsets=[0], skip_first=True)
        True
        >>> is_test_frame_multi(20, test_interval=10, test_offsets=[0], skip_first=True)
        True
    """
    # 跳过第0帧（EmerNeRF兼容）
    if skip_first and frame_idx == 0:
        return False
    
    # 计算在块内的位置
    pos_in_block = frame_idx % test_interval
    
    # 判断是否在测试位置列表中
    return pos_in_block in test_offsets