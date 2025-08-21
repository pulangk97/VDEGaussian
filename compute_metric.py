import numpy as np
import os
import json


def compute_ave_metrics(source_path):
    target_path = source_path+"/results.json"

    scenes = os.listdir(source_path)
    psnr = []
    ssim = []
    lpips = []
    for scene in scenes:
        
        result_path = source_path+f"/{scene}/eval/test_25000_render/metrics.json"
        if not os.path.exists(result_path):
            continue
        with open(result_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            psnr.append(data["psnr"])
            ssim.append(data["ssim"])
            lpips.append(data["lpips"])

    ave_psnr = sum(psnr)/len(psnr)     
    ave_ssim = sum(ssim)/len(ssim)    
    ave_lpips = sum(lpips)/len(lpips)   

    data = {
        "PSNR": ave_psnr,
        "SSIM": ave_ssim,
        "LPIPS": ave_lpips,
    }
    with open(target_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

source_path = ""
compute_ave_metrics(source_path)

    

