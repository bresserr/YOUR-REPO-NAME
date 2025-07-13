"""
Configuration management for VR Video Analysis Application
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    pose_model_path: str = "models/pose_estimation.trt"
    body_detection_model_path: str = "models/body_detection.trt"
    face_detection_model_path: str = "models/face_detection.trt"
    hand_detection_model_path: str = "models/hand_detection.trt"
    genital_detection_model_path: str = "models/genital_detection.trt"
    
    # Model parameters
    input_resolution: tuple = (640, 480)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 10


@dataclass
class VideoConfig:
    """Configuration for video processing"""
    supported_formats: List[str] = None
    max_resolution: tuple = (3840, 2160)  # 4K
    target_fps: int = 30
    batch_size: int = 4
    enable_3d_processing: bool = True
    vr_format_support: bool = True
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']


@dataclass
class CudaConfig:
    """Configuration for CUDA/TensorRT"""
    device_id: int = 0
    memory_fraction: float = 0.8
    enable_tensorrt: bool = True
    tensorrt_precision: str = "fp16"  # fp32, fp16, int8
    max_batch_size: int = 8
    max_workspace_size: int = 1 << 30  # 1GB


@dataclass
class TrackingConfig:
    """Configuration for body part tracking"""
    tracking_algorithm: str = "kalman"  # kalman, particle, optical_flow
    max_lost_frames: int = 10
    tracking_confidence: float = 0.6
    smoothing_factor: float = 0.3
    
    # Body part specific settings
    body_parts: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.body_parts is None:
            self.body_parts = {
                "head": {
                    "priority": 1,
                    "detection_size": (64, 64),
                    "tracking_accuracy": "high"
                },
                "mouth": {
                    "priority": 2,
                    "detection_size": (32, 32),
                    "tracking_accuracy": "high"
                },
                "hand_1": {
                    "priority": 3,
                    "detection_size": (48, 48),
                    "tracking_accuracy": "medium"
                },
                "hand_2": {
                    "priority": 3,
                    "detection_size": (48, 48),
                    "tracking_accuracy": "medium"
                },
                "breasts": {
                    "priority": 4,
                    "detection_size": (64, 64),
                    "tracking_accuracy": "medium"
                },
                "pelvis": {
                    "priority": 5,
                    "detection_size": (64, 64),
                    "tracking_accuracy": "medium"
                },
                "genitals": {
                    "priority": 6,
                    "detection_size": (48, 48),
                    "tracking_accuracy": "high"
                }
            }


@dataclass
class FunscriptConfig:
    """Configuration for funscript generation"""
    output_format: str = "funscript"  # funscript, csv, json
    sampling_rate: int = 100  # Hz
    smoothing_enabled: bool = True
    smoothing_window: int = 5
    
    # Movement analysis parameters
    velocity_threshold: float = 0.1
    acceleration_threshold: float = 0.05
    depth_sensitivity: float = 1.0
    speed_sensitivity: float = 1.0
    
    # Interaction detection
    interaction_distance_threshold: float = 0.05  # meters
    interaction_duration_threshold: float = 0.1  # seconds
    
    # Output ranges
    position_range: tuple = (0, 100)
    speed_range: tuple = (0, 100)


@dataclass
class WebConfig:
    """Configuration for web interface"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    max_upload_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    session_timeout: int = 3600  # 1 hour
    enable_cors: bool = True
    
    # UI settings
    theme: str = "dark"
    language: str = "en"
    auto_save: bool = True
    preview_quality: str = "medium"  # low, medium, high


class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: str = None):
        self.model = ModelConfig()
        self.video = VideoConfig()
        self.cuda = CudaConfig()
        self.tracking = TrackingConfig()
        self.funscript = FunscriptConfig()
        self.web = WebConfig()
        
        # Environment variables
        self.load_from_env()
        
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Create necessary directories
        self.create_directories()
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        # CUDA settings
        if "CUDA_DEVICE_ID" in os.environ:
            self.cuda.device_id = int(os.environ["CUDA_DEVICE_ID"])
        
        if "CUDA_MEMORY_FRACTION" in os.environ:
            self.cuda.memory_fraction = float(os.environ["CUDA_MEMORY_FRACTION"])
        
        # Web settings
        if "WEB_HOST" in os.environ:
            self.web.host = os.environ["WEB_HOST"]
        
        if "WEB_PORT" in os.environ:
            self.web.port = int(os.environ["WEB_PORT"])
        
        if "DEBUG" in os.environ:
            self.web.debug = os.environ["DEBUG"].lower() == "true"
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        import json
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Update configurations based on loaded data
        for section, values in config_data.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            "uploads",
            "outputs",
            "temp",
            "models",
            "logs",
            "static/css",
            "static/js",
            "static/images"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """Get full path for a model file"""
        model_mapping = {
            "pose": self.model.pose_model_path,
            "body": self.model.body_detection_model_path,
            "face": self.model.face_detection_model_path,
            "hand": self.model.hand_detection_model_path,
            "genital": self.model.genital_detection_model_path
        }
        return model_mapping.get(model_name, f"models/{model_name}.trt")
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        try:
            # Check CUDA availability
            import torch
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.cuda.enable_tensorrt = False
            
            # Check model paths
            for model_name in ["pose", "body", "face", "hand", "genital"]:
                model_path = self.get_model_path(model_name)
                if not os.path.exists(model_path):
                    print(f"Warning: Model not found: {model_path}")
            
            # Validate video settings
            if self.video.target_fps <= 0:
                print("Error: Invalid target FPS")
                return False
            
            # Validate tracking settings
            if self.tracking.tracking_confidence < 0 or self.tracking.tracking_confidence > 1:
                print("Error: Invalid tracking confidence")
                return False
            
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def save_to_file(self, config_file: str):
        """Save current configuration to file"""
        import json
        
        config_data = {
            "model": self.model.__dict__,
            "video": self.video.__dict__,
            "cuda": self.cuda.__dict__,
            "tracking": self.tracking.__dict__,
            "funscript": self.funscript.__dict__,
            "web": self.web.__dict__
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""
VR Video Analysis Configuration:
- Models: {self.model.__dict__}
- Video: {self.video.__dict__}
- CUDA: {self.cuda.__dict__}
- Tracking: {self.tracking.__dict__}
- Funscript: {self.funscript.__dict__}
- Web: {self.web.__dict__}
        """