FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
COPY app.py ./
COPY setup.sh ./
COPY README.md ./
COPY diffusers_helper ./diffusers_helper

RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/outputs
RUN mkdir -p /app/hf_download

RUN chmod +x setup.sh

ENV HF_HOME=/app/hf_download

CMD ["python3", "app.py"] 