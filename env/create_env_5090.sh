#!/bin/bash
set -e 

CONDA_BASE=$(conda info --base)

ENV_NAME=pytorch-env
PYTHON_VERSION=3.10.18
CUDA_VERSION=12.8

TORCH_VERSION=2.7.1
TORCHVISION_VERSION=0.22.1
TORCHAUDIO_VERSION=2.7.1

echo -e "\033[94mCreating conda environment:\033[0m $ENV_NAME"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -y -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

echo -e "\033[94mInstalling core deep learning stack with CUDA\033[0m $CUDA_VERSION"

pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION \
    torchaudio==$TORCHAUDIO_VERSION --index-url https://download.pytorch.org/whl/cu128

echo -e "\033[94mInstalling pip-based packages\033[0m"
pip install \
    albumentations \
    huggingface-hub \
    open3d \
    timm \
    transformers \
    dash \
    plotly \
    beautifulsoup4 \
    colorama \
    imageio \
    json5 \
    matplotlib \
    networkx \
    opencv-python \
    opencv-python-headless \
    pandas \
    pillow \
    scikit-image \
    scikit-learn \
    scipy \
    tqdm \
    pandas \
    jupyterlab \
    ipykernel \
    torchinfo \
    torchmetrics

pip install monai --no-deps

echo -e "\033[94mInstalling PyG with CUDA wheels\033[0m"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

pip install torch-geometric==2.6.1

echo -e "\033[92mEnvironment\033[0m $ENV_NAME \033[92mis ready. To activate: conda activate\033[0m $ENV_NAME"
