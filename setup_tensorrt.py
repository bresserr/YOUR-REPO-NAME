#!/usr/bin/env python3
"""
TensorRT Setup and Model Optimization Script for VR Body Analyzer
This script helps set up TensorRT and convert models for optimal performance.
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path

def check_cuda_availability():
    """Check if CUDA is available and which version"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA {cuda_version} detected")
            print(f"✓ GPU: {gpu_name}")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def install_tensorrt():
    """Install TensorRT based on system configuration"""
    system = platform.system().lower()
    
    if system == "linux":
        print("Installing TensorRT for Linux...")
        
        # For Ubuntu/Debian systems
        commands = [
            "pip install nvidia-tensorrt",
            "pip install pycuda"
        ]
        
        for cmd in commands:
            try:
                print(f"Running: {cmd}")
                subprocess.run(cmd.split(), check=True)
                print(f"✓ {cmd} completed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to run {cmd}: {e}")
                print("Please install TensorRT manually:")
                print("1. Download TensorRT from NVIDIA Developer site")
                print("2. Follow installation guide for your system")
                return False
                
    elif system == "windows":
        print("Installing TensorRT for Windows...")
        print("Please install TensorRT manually on Windows:")
        print("1. Download TensorRT from NVIDIA Developer site")
        print("2. Add TensorRT to PATH")
        print("3. Install: pip install pycuda")
        return False
        
    else:
        print(f"Unsupported system: {system}")
        return False
    
    return True

def download_pose_model():
    """Download a pre-trained pose detection model"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download MediaPipe pose model (this is handled automatically by MediaPipe)
    print("MediaPipe models will be downloaded automatically on first use")
    
    # For custom models, you would download them here
    # Example:
    # model_url = "https://example.com/pose_model.onnx"
    # model_path = models_dir / "pose_model.onnx"
    # 
    # if not model_path.exists():
    #     print(f"Downloading pose model to {model_path}")
    #     urllib.request.urlretrieve(model_url, model_path)
    #     print("✓ Model downloaded successfully")
    
    return True

def optimize_model_for_tensorrt():
    """Convert ONNX model to TensorRT engine"""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        print("TensorRT is available - you can optimize models")
        
        # Example optimization code (would need actual model)
        sample_code = '''
# Example TensorRT optimization code:
import tensorrt as trt

def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            return None
    
    engine = builder.build_engine(network, config)
    
    if engine:
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        return True
    return False
'''
        
        with open("tensorrt_optimization_example.py", "w") as f:
            f.write(sample_code)
        
        print("✓ TensorRT optimization example saved to tensorrt_optimization_example.py")
        return True
        
    except ImportError:
        print("✗ TensorRT not available for model optimization")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = ["uploads", "models", "outputs", "static/css", "static/js", "templates"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False

def main():
    """Main setup function"""
    print("VR Body Analyzer - TensorRT Setup")
    print("=" * 40)
    
    # Check system requirements
    print("\n1. Checking CUDA availability...")
    cuda_available = check_cuda_availability()
    
    # Setup directories
    print("\n2. Setting up directories...")
    setup_directories()
    
    # Install dependencies
    print("\n3. Installing dependencies...")
    if not install_dependencies():
        return False
    
    # Install TensorRT if CUDA is available
    if cuda_available:
        print("\n4. Installing TensorRT...")
        install_tensorrt()
        
        print("\n5. Setting up model optimization...")
        optimize_model_for_tensorrt()
    else:
        print("\n4. Skipping TensorRT installation (CUDA not available)")
        print("The application will use MediaPipe CPU mode")
    
    # Download models
    print("\n6. Setting up pose detection models...")
    download_pose_model()
    
    print("\n" + "=" * 40)
    print("Setup completed!")
    print("\nTo run the application:")
    print("  python vr_body_analyzer.py")
    print("\nTo access the web interface:")
    print("  http://localhost:5000")
    
    if not cuda_available:
        print("\nNote: Running in CPU mode. For GPU acceleration:")
        print("1. Install NVIDIA CUDA Toolkit")
        print("2. Install TensorRT")
        print("3. Re-run this setup script")
    
    return True

if __name__ == "__main__":
    main()