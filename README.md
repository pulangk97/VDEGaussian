## VDEGaussian

Official implementation of "VDEGaussian: Video Diffusion Enhanced 4D Gaussian Splatting for Dynamic Urban Scenes Modeling".

### [Project](https://pulangk97.github.io/VDEGaussian-Project/) [Paper](https://www.arxiv.org/pdf/2508.02129)

### Pipeline

### Installation

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

pip install -r requirements_dc.txt

```
### Getting Start


