# syntax = docker/dockerfile:1.12
FROM nvidia/cuda:12.6.3-base-ubuntu22.04 AS builder
RUN apt-get update 
RUN apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    cmake \
    python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*\
    # python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache python3.10 -m pip install --user --no-cache-dir --upgrade pip setuptools wheel

# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# RUN git clone https://github.com/allenai/olmes.git && cd olmes  && pip install --user --no-cache-dir --no-build-isolation -e . && cd ..
RUN git clone https://github.com/allenai/olmes.git && cd olmes && git checkout 5a51f50

RUN cd olmes && \
# transformers
sed -i \
  -e 's/transformers>=4\.57\.0,<5/transformers>=5.0.0/' \
  -e 's/fsspec==2023\.5\.0/fsspec==2023.10.0/' \
  -e 's/"datasets>=3,<4"/"datasets==4.0.0",\n  "accelerate>=0.30.0"/' \
  pyproject.toml && cat   pyproject.toml

RUN --mount=type=cache,target=/root/.cache cd olmes && pip install --user  . && cd ..







RUN pip show datasets
RUN pip show transformers





ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages
ENV PYTHONNOUSERSITE=1


WORKDIR /app

RUN ln -s /usr/bin/python3.10 /usr/bin/python


CMD ["bash"]