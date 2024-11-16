ARG CUDA_VERSION
# CUDA_VERSION = 12.1.0 | 11.8

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu20.04

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

RUN apt-get update -y && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    ca-certificates \
    build-essential \
    curl \
    unzip \
    git \
    vim

RUN rm -rf /var/lib/apt/lists/* \
    && unlink /usr/bin/python3 \
    && ln -s /usr/bin/python3.10 /usr/bin/python3 \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && pip3 install -U pip

COPY pyproject.toml pyproject.toml
RUN mkdir src

RUN apt-get update && apt-get -y install python3.10-venv

ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

RUN /root/.local/bin/uv pip install --system --no-cache -e .[batch_inference,dev,vllm]

ENV PYTHONUNBUFFERED=1
COPY src src
COPY tests tests
COPY configs configs
