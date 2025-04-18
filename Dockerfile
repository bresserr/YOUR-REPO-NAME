FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 设置非交互式安装并避免不必要的包
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装基本工具和Python
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制需要的文件
COPY requirements.txt ./
COPY app.py ./
COPY setup.sh ./
COPY README.md ./
COPY diffusers_helper ./diffusers_helper

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 创建需要的目录
RUN mkdir -p /app/outputs
RUN mkdir -p /app/hf_download

# 设置权限
RUN chmod +x setup.sh

# 设置环境变量
ENV HF_HOME=/app/hf_download

# 运行应用
CMD ["python3", "app.py"] 