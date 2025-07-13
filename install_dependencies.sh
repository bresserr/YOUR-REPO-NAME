#!/bin/bash

# VR Video Analyzer Installation Script
echo "🚀 Installing VR Video Analyzer dependencies..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected"
    CUDA_AVAILABLE=true
else
    echo "⚠️  CUDA not detected. Some features may be limited."
    CUDA_AVAILABLE=false
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Check if Python version is 3.8 or higher
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv vr_analyzer_env
source vr_analyzer_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support if available
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🔥 Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install basic requirements
echo "📋 Installing basic requirements..."
pip install -r requirements.txt

# Install TensorRT if CUDA is available
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "⚡ Installing TensorRT..."
    pip install tensorrt
    pip install pycuda
    echo "✅ TensorRT installed"
else
    echo "⚠️  Skipping TensorRT installation (CUDA not available)"
fi

# Install MediaPipe
echo "🎥 Installing MediaPipe..."
pip install mediapipe

# Install additional video processing dependencies
echo "📹 Installing video processing dependencies..."
pip install opencv-contrib-python
pip install imageio[ffmpeg]

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install pytest pytest-cov black flake8 mypy

# Make scripts executable
chmod +x vr_video_analyzer.py
chmod +x install_dependencies.sh

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import torch
import cv2
import mediapipe as mp
import gradio as gr
import numpy as np
import scipy
print('✅ All core dependencies imported successfully')
"

if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🔍 Testing CUDA availability..."
    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  CUDA not available in PyTorch')
"
fi

# Test TensorRT if available
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🚀 Testing TensorRT..."
    python3 -c "
try:
    import tensorrt as trt
    print(f'✅ TensorRT available: {trt.__version__}')
except ImportError:
    print('⚠️  TensorRT not available')
" 2>/dev/null || echo "⚠️  TensorRT test failed"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "To run the application:"
echo "  source vr_analyzer_env/bin/activate"
echo "  python3 vr_video_analyzer.py"
echo ""
echo "Features available:"
echo "  ✅ Video processing"
echo "  ✅ Body part detection"
echo "  ✅ Movement analysis"
echo "  ✅ Funscript generation"
echo "  ✅ Web interface"
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "  ✅ CUDA acceleration"
    echo "  ✅ TensorRT optimization"
fi
echo ""
echo "📖 For more information, check the README.md file"