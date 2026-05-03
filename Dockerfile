FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# PyTorch with CUDA 12.8 support, installed from official index (no mirror)
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Remaining dependencies from requirements.txt (excluding torch/torchvision)
RUN pip install \
        "transformers>=4.45.0" \
        "PyYAML>=6.0.1" \
        "Pillow>=10.0.0" \
        "tqdm>=4.66.0" \
        "scikit-learn>=1.4.0" \
        "pytest>=8.0.0"

COPY . .

ENV PYTHONPATH=/workspace

CMD ["bash"]
