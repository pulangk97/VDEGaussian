        
from omegaconf import OmegaConf
from collections import OrderedDict
import argparse, os, sys, glob
from submodules.DynamiCrafter.utils.utils import instantiate_from_config
import torch
import torch.nn.functional as F
from einops import rearrange,repeat
import random
import time
from tqdm import tqdm
from submodules.DynamiCrafter.scripts.evaluation.inference import image_guided_synthesis
from submodules.DynamiCrafter.scripts.evaluation.inference_adapt import add_lora_to_ldmvfi, load_model_checkpoint
import torchvision.transforms as transforms

from submodules.DynamiCrafter.main.adapter import LoRALinearLayer, freeze_original_parameters
from lvdm.modules.attention import CrossAttention
import lpips
from utils.loss_utils import psnr, ssim
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg
def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))



class DynamiCrafter(torch.nn.Module):
    def __init__(self, args, ckp_dir =None, adapted = False):
        super().__init__()


        self.device = torch.device("cuda:0")

        self.guidance_rescale = 0.7
        current_dir = os.getcwd()
        print(f"Current Directory: {current_dir}")
        sys.path.append(current_dir+'/submodules/DynamiCrafter')

        config = OmegaConf.load(args)


        
        config["model"]["params"]["cond_stage_config"]["params"]["device"] = "cuda:0"
        config["model"]["params"]["img_cond_stage_config"]["params"]["device"] = "cuda:0"

        self.config = config
        model_config = config.pop("model", OmegaConf.create())


        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)

        
        model = model.to(self.device)
        model.perframe_ae = model_config['params']['perframe_ae']
        if ckp_dir == None:
            ckp = os.path.join(current_dir+'/submodules/DynamiCrafter',model_config['pretrained_checkpoint'])
        else:
            ckp = ckp_dir
        assert os.path.exists(ckp), "Error: checkpoint Not Found!"
        

        if adapted==True:
            model.model = add_lora_to_ldmvfi(model.model, replace_layer = CrossAttention, adapte_layer = LoRALinearLayer)

        print("read ckp from "+ckp)
        model = load_model_checkpoint(model, ckp)

        model.eval()
        self.model = model

        

        self.num_interpolate = self.config["data"]["params"]["train"]["params"]["video_length"]




        self.alphas = self.model.alphas_cumprod.to(self.device) # for convenience

        self.unconditional_guidance_scale = config.pop("lightning", OmegaConf.create())["callbacks"]["batch_logger"]["params"]["log_images_kwargs"]["unconditional_guidance_scale"]
        print("use unconditional guidence scale:"+str(self.unconditional_guidance_scale))


        self.num_train_timesteps = model_config['params']['timesteps']
        t_range = [0.02, 0.98]
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])


        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to("cuda:0")  # closer to "traditional"


    def pseudo_loss(self, batch_input, batch_gt, down_rate = -1, loss_function = "l1", f_weight=None, lambda_f = 1):

            video_size = (320, 512)
            transform = transforms.Compose([
                transforms.Resize(min(video_size)),
                transforms.CenterCrop(video_size),])

            batch_gt = torch.clamp(batch_gt,-1,1)/2+0.5


            if down_rate>0:
                batch_input = F.interpolate(
                    batch_input[0].permute(1, 0, 2, 3),  # (T, C, H, W)
                    scale_factor=down_rate,
                    mode='bilinear',
                    align_corners=False
                ).permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)

            B, C, T, H, W = batch_input.shape

            flat_input = batch_input.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


            transformed = transform(flat_input)  # (B*T, C, H, W)

            batch_input = transformed.view(B, T, C, 320, 512).permute(0, 2, 1, 3, 4)


            B,C,T,H,W = batch_input.shape



            if_perform_pseudo_loss =True

            if if_perform_pseudo_loss == True:
                if loss_function=="lpips":
                    loss = self.loss_fn_vgg(batch_input.permute(0,2,1,3,4).reshape(-1,C,H,W), batch_gt.permute(0,2,1,3,4).reshape(-1,C,H,W))
                elif loss_function == "l1":
                    if f_weight is None:
                        loss = F.l1_loss(batch_input, batch_gt)
                    else:
                        loss = torch.mean(f_weight[:,None,...] * (batch_input - batch_gt)**2 - lambda_f*f_weight[:,None,...]**2)
            else:
                loss = torch.zeros(1)



            loss = torch.mean(loss)
            return loss

    def inference(self, batch_input):
        video_size = (320, 512)
        transform = transforms.Compose([
            transforms.Resize(min(video_size)),
            transforms.CenterCrop(video_size),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


        all_batch = []
        for i in range(batch_input.shape[0]):
            new_batch = []
            for j in range(batch_input.shape[2]):
                new_batch.append(transform(batch_input[i,:,j,...]))
            new_batch = torch.stack(new_batch)
            all_batch.append(new_batch)
            
        videos = torch.stack(all_batch).permute(0,2,1,3,4)

        model = self.model
        prompts = ["a driving scene"]*videos.shape[0]
        videos = videos
        noise_shape = [videos.shape[0], self.model.model.diffusion_model.out_channels, videos.shape[2], videos.shape[3]// 8, videos.shape[4]// 8]
        with torch.no_grad():

            batch_samples, batch_latent = image_guided_synthesis(model, prompts, videos, noise_shape, 1, 50, 1.0, \
                                7.5, None, 20, True, False, False, True, 'uniform_trailing', 0.7)
        return batch_samples, batch_latent


