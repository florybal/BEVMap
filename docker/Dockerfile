FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

WORKDIR /workspace

RUN rm -f /etc/apt/sources.list.d/cuda* /etc/apt/sources.list.d/nvidia* || true \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC 2>/dev/null || true \
    && apt-get update -y || apt-get update -y --allow-releaseinfo-change \
    && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        build-essential \
        python3-dev \
        ninja-build \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN pip install -U pip wheel setuptools==79.0.1 \
 && pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 -f https://download.pytorch.org/whl/torch_stable.html \
 && pip install --no-build-isolation mmcv==2.0.0 \
 && pip install mmengine==0.10.7 \
 && pip install mmdet==3.0.0 mmsegmentation==1.0.0 \
 && pip install -r requirements/runtime.txt \
 && pip install -e . --no-build-isolation

CMD ["/bin/bash"]
