"""
VR Video Processor for handling 3D VR video formats and frame extraction
"""

import os
import cv2
import numpy as np
from typing import Generator, Dict, List, Optional, Tuple, Any
import threading
import queue
import time
from pathlib import Path
from loguru import logger

import torch
import torchvision.transforms as transforms
from PIL import Image
import imageio
import ffmpeg

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    logger.warning("PyAV not available. Some video formats may not be supported.")

from src.utils.config import Config


class VRVideoProcessor:
    """
    Processes VR videos including stereo formats, 360-degree videos, and standard formats
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.supported_formats = config.video.supported_formats
        self.max_resolution = config.video.max_resolution
        self.target_fps = config.video.target_fps
        self.batch_size = config.video.batch_size
        self.enable_3d = config.video.enable_3d_processing
        
        # Video format detection patterns
        self.vr_formats = {
            'sbs': 'side_by_side',  # Side-by-side stereo
            'ou': 'over_under',     # Over-under stereo
            '360': '360_mono',      # 360-degree mono
            '360_3d': '360_stereo', # 360-degree stereo
            'flat': 'flat_mono'     # Standard flat video
        }
        
        # Initialize transforms
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Threading for performance
        self.frame_queue = queue.Queue(maxsize=30)
        self.processing_thread = None
        self.stop_processing = False
        
        logger.info("VR Video Processor initialized")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get comprehensive video information"""
        try:
            # Use OpenCV to get basic info
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Detect VR format
            vr_format = self.detect_vr_format(width, height, video_path)
            
            # Get file info
            file_size = os.path.getsize(video_path)
            file_ext = Path(video_path).suffix.lower()
            
            video_info = {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'vr_format': vr_format,
                'file_size': file_size,
                'file_extension': file_ext,
                'aspect_ratio': width / height if height > 0 else 0,
                'is_stereo': vr_format in ['side_by_side', 'over_under', '360_stereo'],
                'is_360': '360' in vr_format,
                'estimated_eye_width': width // 2 if vr_format == 'side_by_side' else width,
                'estimated_eye_height': height // 2 if vr_format == 'over_under' else height
            }
            
            logger.info(f"Video info: {video_info}")
            return video_info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}
    
    def detect_vr_format(self, width: int, height: int, video_path: str) -> str:
        """Detect VR format based on resolution and metadata"""
        aspect_ratio = width / height
        
        # Check for common VR aspect ratios
        if abs(aspect_ratio - 2.0) < 0.1:  # ~2:1 aspect ratio
            return 'side_by_side'
        elif abs(aspect_ratio - 1.0) < 0.1:  # ~1:1 aspect ratio
            return 'over_under'
        elif abs(aspect_ratio - 2.0) < 0.2 and width >= 3840:  # 4K+ with ~2:1
            return '360_stereo'
        elif abs(aspect_ratio - 2.0) < 0.2 and width >= 1920:  # HD+ with ~2:1
            return '360_mono'
        
        # Check filename for VR indicators
        filename = Path(video_path).name.lower()
        if any(indicator in filename for indicator in ['sbs', 'side_by_side', 'lr']):
            return 'side_by_side'
        elif any(indicator in filename for indicator in ['ou', 'over_under', 'tb']):
            return 'over_under'
        elif any(indicator in filename for indicator in ['360', 'vr']):
            return '360_mono'
        
        return 'flat_mono'
    
    def split_stereo_frame(self, frame: np.ndarray, vr_format: str) -> Tuple[np.ndarray, np.ndarray]:
        """Split stereo frame into left and right eye views"""
        if vr_format == 'side_by_side':
            height, width = frame.shape[:2]
            mid_width = width // 2
            left_eye = frame[:, :mid_width]
            right_eye = frame[:, mid_width:]
        elif vr_format == 'over_under':
            height, width = frame.shape[:2]
            mid_height = height // 2
            left_eye = frame[:mid_height, :]
            right_eye = frame[mid_height:, :]
        else:
            # For non-stereo formats, return the same frame for both eyes
            left_eye = frame
            right_eye = frame
        
        return left_eye, right_eye
    
    def preprocess_frame(self, frame: np.ndarray, vr_format: str) -> Dict[str, Any]:
        """Preprocess frame for analysis"""
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Handle stereo formats
        if vr_format in ['side_by_side', 'over_under', '360_stereo']:
            left_eye, right_eye = self.split_stereo_frame(frame_rgb, vr_format)
            
            # For now, we'll primarily analyze the left eye
            # In the future, we can combine information from both eyes
            primary_frame = left_eye
            stereo_data = {
                'left_eye': left_eye,
                'right_eye': right_eye,
                'has_stereo': True
            }
        else:
            primary_frame = frame_rgb
            stereo_data = {
                'left_eye': frame_rgb,
                'right_eye': frame_rgb,
                'has_stereo': False
            }
        
        # Resize if needed
        if primary_frame.shape[:2] != (480, 640):
            primary_frame = cv2.resize(primary_frame, (640, 480))
        
        # Convert to tensor for ML processing
        frame_tensor = self.transforms(primary_frame)
        
        return {
            'frame': primary_frame,
            'frame_tensor': frame_tensor,
            'stereo_data': stereo_data,
            'original_shape': frame.shape,
            'vr_format': vr_format
        }
    
    def process_video(self, video_path: str, start_frame: int = 0, 
                     end_frame: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """Process video frames with VR format handling"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Get video info
            video_info = self.get_video_info(video_path)
            vr_format = video_info['vr_format']
            total_frames = video_info['frame_count']
            
            # Set start frame
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_number = start_frame
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Check end frame
                if end_frame is not None and frame_number >= end_frame:
                    break
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame, vr_format)
                
                # Add frame metadata
                frame_data = {
                    'frame_number': frame_number,
                    'timestamp': frame_number / video_info['fps'],
                    'progress': frame_number / total_frames,
                    **processed_frame
                }
                
                yield frame_data
                
                frame_number += 1
                
                # Yield control for real-time processing
                if frame_number % 10 == 0:
                    time.sleep(0.001)
            
            cap.release()
            logger.info(f"Video processing completed: {frame_number} frames processed")
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
    
    def extract_frames_batch(self, video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
        """Extract specific frames by index"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            video_info = self.get_video_info(video_path)
            vr_format = video_info['vr_format']
            
            for frame_idx in sorted(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    processed_frame = self.preprocess_frame(frame, vr_format)
                    frames.append(processed_frame['frame'])
                else:
                    logger.warning(f"Could not extract frame {frame_idx}")
                    frames.append(None)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        
        return frames
    
    def create_preview_video(self, frames: List[np.ndarray], output_path: str, 
                           fps: int = 24) -> bool:
        """Create preview video from frames"""
        try:
            if not frames:
                return False
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                out.write(frame_bgr)
            
            out.release()
            logger.info(f"Preview video created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating preview video: {e}")
            return False
    
    def get_frame_at_time(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """Get frame at specific timestamp"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            # Set position by timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            
            cap.release()
            
            if ret:
                video_info = self.get_video_info(video_path)
                vr_format = video_info['vr_format']
                processed_frame = self.preprocess_frame(frame, vr_format)
                return processed_frame['frame']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting frame at time {timestamp}: {e}")
            return None
    
    def analyze_motion(self, frames: List[np.ndarray], method: str = 'optical_flow') -> Dict[str, Any]:
        """Analyze motion between frames"""
        if len(frames) < 2:
            return {}
        
        motion_data = {
            'method': method,
            'frame_count': len(frames),
            'motion_vectors': [],
            'motion_magnitude': [],
            'motion_direction': []
        }
        
        try:
            if method == 'optical_flow':
                # Lucas-Kanade optical flow
                for i in range(len(frames) - 1):
                    frame1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                    frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                    
                    # Calculate optical flow
                    flow = cv2.calcOpticalFlowPyrLK(frame1, frame2, None, None)
                    
                    if flow is not None:
                        # Calculate motion statistics
                        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                        direction = np.arctan2(flow[..., 1], flow[..., 0])
                        
                        motion_data['motion_vectors'].append(flow)
                        motion_data['motion_magnitude'].append(np.mean(magnitude))
                        motion_data['motion_direction'].append(np.mean(direction))
            
            elif method == 'frame_difference':
                # Simple frame difference
                for i in range(len(frames) - 1):
                    diff = cv2.absdiff(frames[i], frames[i + 1])
                    motion_magnitude = np.mean(diff)
                    motion_data['motion_magnitude'].append(motion_magnitude)
            
            logger.info(f"Motion analysis completed using {method}")
            
        except Exception as e:
            logger.error(f"Error in motion analysis: {e}")
        
        return motion_data
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_processing = True
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("VR Video Processor cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()