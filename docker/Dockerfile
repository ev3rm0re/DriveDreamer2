FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 1. 基础环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 指定 CUDA 架构
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9+PTX"

# 2. 安装系统依赖
RUN apt-get update -q && \
    apt-get install -q -y \
    wget \
    git \
    ninja-build \
    ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxrender-dev libxext6 \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. 安装 Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 4. 安装 PyTorch
ENV PATH=${PATH}:/usr/local/cuda:/usr/local/cuda/bin
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu121


RUN pip install nuscenes-devkit
RUN pip install lmdb
RUN pip install decord
RUN pip install accelerate
RUN pip install transformers
RUN pip install pytorch-fid
RUN pip install lpips
RUN pip install terminaltables
RUN pip install opencv-python
RUN pip install ftfy
RUN pip install einops
RUN pip install compel
RUN pip install bs4
RUN pip install open_clip_torch
RUN pip install pynvml
RUN pip install paramiko
RUN pip install tensorboard
RUN pip install mediapipe
RUN pip install deepspeed
RUN pip install diffusers
RUN pip install imageio
RUN pip install imageio[ffmpeg]
RUN pip install scikit-image

