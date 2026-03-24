import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

if __name__ == "__main__":
    # 使用绝对路径
    local_model_path = '/home/ubuntu/VDE/SegFormer'
    
    # 检查路径是否存在
    print(f"Checking model path: {local_model_path}")
    print(f"Path exists: {os.path.exists(local_model_path)}")
    
    # 列出路径中的文件
    if os.path.exists(local_model_path):
        print(f"Files in path: {os.listdir(local_model_path)[:10]}")  # 显示前10个文件
    
    # 加载图像处理器和模型 - 强制使用本地路径
    print("Loading SegFormer model from local path...")
    
    try:
        # 方法1：直接指定本地路径
        image_processor = SegformerImageProcessor.from_pretrained(local_model_path, local_files_only=True)
        model = SegformerForSemanticSegmentation.from_pretrained(local_model_path, local_files_only=True)
    except Exception as e:
        print(f"Error loading with local_files_only=True: {e}")
        print("Trying without local_files_only flag...")
        # 方法2：直接使用路径
        image_processor = SegformerImageProcessor.from_pretrained(local_model_path)
        model = SegformerForSemanticSegmentation.from_pretrained(local_model_path)
    
    model = model.to('cuda')
    model.eval()
    
    # Cityscapes数据集中天空的类别ID通常是10
    SKY_CLASS_ID = 10
    
    root = './PVG/data/waymo_scenes_streetsurf'
    scenes = sorted(os.listdir(root))
    
    for scene in scenes:
        for cam_id in range(5):
            image_dir = os.path.join(root, scene, f'image_{cam_id}')
            sky_dir = os.path.join(root, scene, f'sky_{cam_id}')
            os.makedirs(sky_dir, exist_ok=True)
            
            image_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith(".jpg")]
            for image_name in tqdm(image_files, desc=f"Processing {scene}/cam_{cam_id}"):
                image_path = os.path.join(image_dir, image_name)
                mask_path = os.path.join(sky_dir, image_name)
                
                # 使用PIL读取图像
                image = Image.open(image_path).convert("RGB")
                
                # 预处理图像
                inputs = image_processor(images=image, return_tensors="pt")
        
                # 将输入移到GPU
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                # 推理
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # 获取分割结果（上采样到原始图像大小）
                logits = outputs.logits
                
                # 上采样到原始图像大小
                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=image.size[::-1],  # (height, width)
                    mode='bilinear',
                    align_corners=False
                )
                
                # 获取预测的类别
                predicted = upsampled_logits.argmax(dim=1).cpu().numpy()[0]
                
                # 创建天空掩码（天空类别ID为10）
                sky_mask = (predicted == SKY_CLASS_ID).astype(np.float32) * 255
                sky_mask = sky_mask.astype(np.uint8)
                
                # 保存掩码
                cv2.imwrite(mask_path, sky_mask)

    print("Processing completed!")