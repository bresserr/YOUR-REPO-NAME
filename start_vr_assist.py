#!/usr/bin/env python3
"""
VR Assistive Technology - Startup Script
Checks dependencies and launches the application
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'opencv-python',
        'mediapipe',
        'torch',
        'numpy',
        'scipy',
        'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is not installed")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages:")
        logger.info("pip install -r requirements_vr_assist.txt")
        return False
    
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA is available - {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            logger.warning("✗ CUDA is not available - using CPU")
            return False
    except ImportError:
        logger.warning("PyTorch not available - cannot check CUDA")
        return False

def check_directories():
    """Check and create necessary directories"""
    directories = ['uploads', 'output', 'static/css', 'static/js', 'templates']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory {directory} ready")

def check_files():
    """Check if required files exist"""
    required_files = [
        'vr_assist_app.py',
        'body_detector.py',
        'video_processor.py',
        'tracking_manager.py',
        'funscript_generator.py',
        'templates/index.html',
        'static/css/main.css',
        'static/js/main.js'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            logger.error(f"✗ Missing file: {file_path}")
        else:
            logger.info(f"✓ File exists: {file_path}")
    
    if missing_files:
        logger.error(f"Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_port(port=8000):
    """Check if port is available"""
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            logger.info(f"✓ Port {port} is available")
            return True
    except OSError:
        logger.error(f"✗ Port {port} is already in use")
        return False

def setup_environment():
    """Setup environment variables"""
    # Set default environment variables if not present
    env_vars = {
        'VR_ASSIST_CUDA_DEVICE': '0',
        'VR_ASSIST_BATCH_SIZE': '4',
        'VR_ASSIST_CONFIDENCE_THRESHOLD': '0.5',
        'VR_ASSIST_MAX_WORKERS': '4'
    }
    
    for var, default_value in env_vars.items():
        if var not in os.environ:
            os.environ[var] = default_value
            logger.info(f"Set {var} = {default_value}")

def run_application():
    """Run the VR assistive technology application"""
    logger.info("Starting VR Assistive Technology Application...")
    
    try:
        # Import and run the application
        from vr_assist_app import app
        import uvicorn
        
        logger.info("Application loaded successfully")
        logger.info("Starting server on http://localhost:8000")
        logger.info("Press Ctrl+C to stop the server")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
        
    except ImportError as e:
        logger.error(f"Failed to import application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error running application: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("=" * 60)
    print("VR Assistive Technology - Startup Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check CUDA
    logger.info("Checking CUDA availability...")
    cuda_available = check_cuda()
    
    # Check directories
    logger.info("Checking directories...")
    check_directories()
    
    # Check files
    logger.info("Checking required files...")
    if not check_files():
        sys.exit(1)
    
    # Check port
    logger.info("Checking port availability...")
    if not check_port():
        logger.info("You can specify a different port if needed")
        sys.exit(1)
    
    # Setup environment
    logger.info("Setting up environment...")
    setup_environment()
    
    # All checks passed
    logger.info("All checks passed! Starting application...")
    
    if not cuda_available:
        logger.warning("CUDA is not available. The application will run on CPU.")
        logger.warning("For better performance, consider setting up CUDA.")
    
    # Run the application
    run_application()

if __name__ == "__main__":
    main()