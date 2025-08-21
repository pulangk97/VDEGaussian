## VDEGaussian

Official implementation of "VDEGaussian: Video Diffusion Enhanced 4D Gaussian Splatting for Dynamic Urban Scenes Modeling".

### [Project](https://pulangk97.github.io/VDEGaussian-Project/) | [Paper](https://www.arxiv.org/pdf/2508.02129)

### Pipeline

### Installation
#### Environments

```
# Make a conda environment.
conda create --name vdegaussian python=3.9
conda activate vdegaussian

# Install requirements.
pip install -r requirements.txt

# Install simple-knn
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./simple-knn

# a modified gaussian splatting (for feature rendering)
git clone --recursive https://github.com/SuLvXiangXin/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# Install nvdiffrast (for Envlight)
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast

## Install requirements for DynamiCrafter
pip install -r requirements_dc.txt


```
#### Download Checkpoints

1. Download the checkpoint of DynamiCrafter for [Interpolation](https://huggingface.co/Doubiiu/DynamiCrafter_512_Interp/blob/main/model.ckpt). 
2. Put `model.ckpt` into `./checkpoints/dynamicrafter_512_interp_v1/`
3. Download `open_clip_pytorch_model.bin` and put it into `./submodules/DynamiCrafter/pretrained/`
### Datasets
#### Waymo Open Datasets
1. Download raw scenes used in original paper (segment id: `102319, 103913, 106250, 109636, 121618, and 225932`) from [Waymo](https://waymo.com/open/)
2. Process the raw scenes following [PVG](https://github.com/fudan-zvg/PVG) 
3. Put the processed scenes into `./data/waymo_scenes`


### Getting Start

#### Stage 1 (Test Time Adaptation)
```
bash scene_train.sh
```

#### Stage 2 (4DGS Training)

```
bash pvg_train.sh
```