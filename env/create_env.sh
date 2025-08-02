#!/bin/bash
set -e 

CONDA_BASE=$(conda info --base)

ENV_NAME=pytorch-env
PYTHON_VERSION=3.8.20
CUDA_VERSION=12.1
TORCH_VERSION=2.4.1
TORCHVISION_VERSION=0.19.1
TORCHAUDIO_VERSION=2.4.1

echo -e "\033[94mCreating conda environment:\033[0m $ENV_NAME"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -y -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

echo -e "\033[94mInstalling core deep learning stack with CUDA\033[0m $CUDA_VERSION"

pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION \
    torchaudio==$TORCHAUDIO_VERSION --index-url https://download.pytorch.org/whl/cu121

echo -e "\033[94mInstalling pip-based packages\033[0m"
pip install \
    albumentations==1.4.18 \
    huggingface-hub==0.30.2 \
    monai==1.3.2 \
    open3d==0.19.0 \
    timm==1.0.15 \
    transformers==4.46.3 \
    dash==2.18.2 \
    plotly==5.24.1 \
    beautifulsoup4==4.12.3 \
    colorama==0.4.6 \
    imageio==2.35.1 \
    json5==0.9.6 \
    matplotlib==3.7.5 \
    networkx==3.0 \
    opencv-python==4.10.0.84 \
    opencv-python-headless==4.12.0.88 \
    pandas==2.0.3 \
    pillow==10.2.0 \
    scikit-image==0.21.0 \
    scikit-learn==1.3.2 \
    scipy==1.10.1 \
    tqdm==4.67.1 \
    pandas==2.0.3 \
    jupyterlab==4.2.5 \
    ipykernel==6.29.5 \
    torchinfo \
    torchmetrics==1.5.2

echo -e "\033[94mInstalling PyG with CUDA wheels\033[0m"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install torch-geometric==2.6.1

echo -e "\033[92mEnvironment\033[0m $ENV_NAME \033[92mis ready. To activate: conda activate\033[0m $ENV_NAME"
