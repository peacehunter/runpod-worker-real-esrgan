FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Upgrade apt packages and install required dependencies
RUN apt update -o Acquire::ForceIPv4=true && \
    apt upgrade -y -o Acquire::ForceIPv4=true && \
    apt install -y -o Acquire::ForceIPv4=true \
      python3-dev \
      python3-pip \
      python3.10-venv \
      fonts-dejavu-core \
      rsync \
      git \
      jq \
      moreutils \
      aria2 \
      wget \
      curl \
      libglib2.0-0 \
      libsm6 \
      libgl1 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      libgoogle-perftools-dev \
      procps && \
    apt-get autoremove -y -o Acquire::ForceIPv4=true && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y -o Acquire::ForceIPv4=true

# Install PyTorch, NumPy, and other Python dependencies
RUN pip3 install --no-cache-dir torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir "numpy<2" basicsr

# Install the 'runpod' package
RUN pip3 install runpod

# Install the models
WORKDIR /workspace
RUN mkdir -p /workspace/models/ESRGAN && \
    cd /workspace/models/ESRGAN && \
    # Download the official Real-ESRGAN models
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth && \
    # Download the GFPGAN models
    mkdir -p /workspace/models/GFPGAN && \
    wget -O /workspace/models/GFPGAN/GFPGANv1.3.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

# Clone the worker repo, install dependencies, and run setup
RUN git clone https://github.com/ashleykleynhans/runpod-worker-real-esrgan.git && \
    cd runpod-worker-real-esrgan && \
    pip3 install -r requirements.txt && \
    python3 setup.py develop && \
    python3 create_test_json.py

# Docker container start script
ADD start_standalone.sh /start.sh
ADD rp_handler.py /workspace/runpod-worker-real-esrgan/rp_handler.py

# Start the container
RUN chmod +x /start.sh
ENTRYPOINT ["/start.sh"]