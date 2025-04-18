#!/bin/bash
# 创建必要的目录
mkdir -p hf_download
mkdir -p outputs

# 如果模型尚未下载，会在首次运行时自动下载
echo "环境准备完毕，运行 python app.py 启动应用" 