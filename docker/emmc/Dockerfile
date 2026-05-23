# syntax = docker/dockerfile:1.12
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS builder
RUN apt-get update  && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    cmake \
    python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126


RUN --mount=type=cache,target=/root/.cache pip uninstall  -y ninja && pip install  ninja==1.13.0 
RUN --mount=type=cache,target=/root/.cache \
    pip install --find-links /root/.cache/flash-attn \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl


RUN --mount=type=cache,target=/root/.cache MAX_JOBS=4 pip install   --no-warn-script-location --no-build-isolation  traker[fast]==0.3.2  
RUN --mount=type=cache,target=/root/.cache pip install   --no-warn-script-location clustpy[full]==0.0.2
RUN pip freeze



FROM nvidia/cuda:12.6.3-base-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    rsync \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-latex-recommended \
    libxrandr2\                 
   libxcomposite1\                              
  libxcursor1\                                 
         libxdamage1\                                 
         libgtk-3-0\                                  
         libpangocairo-1.0-0\                         
         libpango-1.0-0\                              
         libatk1.0-0\                                 
         libcairo-gobject2\                           
         libgdk-pixbuf-2.0-0\                         
         libasound2   \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages

COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install scipy==1.15.3
RUN pip freeze


RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -s /usr/bin/python3.10 /usr/bin/python


ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONNOUSERSITE=1

RUN playwright install-deps 
RUN playwright install

WORKDIR /app

CMD ["bash"]

