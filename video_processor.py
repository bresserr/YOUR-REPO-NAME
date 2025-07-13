import cv2
import numpy as np
import torch
import logging
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video loading, processing, and frame extraction for 3D VR videos"""
    
    def __init__(self):
        self.video_capture = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 30.0
        self.frame_width = 0
        self.frame_height = 0
        self.is_vr_360 = False
        self.frame_cache = {}
        self.cache_size = 100
        self.current_frame_id = 0
        self.lock = threading.Lock()
        
    def load_video(self, video_path: str) -> Tuple[bool, str]:
        """Load video file and extract metadata"""
        try:
            if not Path(video_path).exists():
                return False, f"Video file not found: {video_path}"
            
            # Release any existing video capture
            if self.video_capture is not None:
                self.video_capture.release()
            
            # Open video file
            self.video_capture = cv2.VideoCapture(video_path)
            
            if not self.video_capture.isOpened():
                return False, f"Could not open video file: {video_path}"
            
            # Get video properties
            self.video_path = video_path
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Detect if it's a VR 360 video (typically 2:1 aspect ratio)
            aspect_ratio = self.frame_width / self.frame_height
            self.is_vr_360 = abs(aspect_ratio - 2.0) < 0.1
            
            # Clear frame cache
            self.frame_cache.clear()
            
            logger.info(f"Video loaded: {video_path}")
            logger.info(f"Frames: {self.total_frames}, FPS: {self.fps}")
            logger.info(f"Resolution: {self.frame_width}x{self.frame_height}")
            logger.info(f"VR 360: {self.is_vr_360}")
            
            return True, "Video loaded successfully"
            
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return False, f"Error loading video: {str(e)}"
    
    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get a specific frame from the video"""
        try:
            with self.lock:
                if self.video_capture is None:
                    return None
                
                # Check cache first
                if frame_id in self.frame_cache:
                    return self.frame_cache[frame_id].copy()
                
                # Set video position
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                
                # Read frame
                ret, frame = self.video_capture.read()
                
                if not ret:
                    return None
                
                # Process frame for VR if needed
                if self.is_vr_360:
                    frame = self._process_vr_frame(frame)
                
                # Cache frame (with size limit)
                if len(self.frame_cache) >= self.cache_size:
                    # Remove oldest frame
                    oldest_key = min(self.frame_cache.keys())
                    del self.frame_cache[oldest_key]
                
                self.frame_cache[frame_id] = frame.copy()
                
                return frame
                
        except Exception as e:
            logger.error(f"Error getting frame {frame_id}: {e}")
            return None
    
    def _process_vr_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process VR 360 frame to extract the main view"""
        try:
            # For VR 360 videos, we typically want to extract the center portion
            # or convert from equirectangular to a more standard view
            
            height, width = frame.shape[:2]
            
            # Extract left eye view (typically left half of the frame)
            left_eye = frame[:, :width//2]
            
            # For now, we'll use the left eye view
            # In a more advanced implementation, we could:
            # 1. Combine both eyes for depth information
            # 2. Apply equirectangular to perspective conversion
            # 3. Allow user to select viewing angle
            
            return left_eye
            
        except Exception as e:
            logger.error(f"Error processing VR frame: {e}")
            return frame
    
    def get_frame_sequence(self, start_frame: int, end_frame: int) -> List[np.ndarray]:
        """Get a sequence of frames"""
        frames = []
        for frame_id in range(start_frame, min(end_frame + 1, self.total_frames)):
            frame = self.get_frame(frame_id)
            if frame is not None:
                frames.append(frame)
        return frames
    
    def get_total_frames(self) -> int:
        """Get total number of frames"""
        return self.total_frames
    
    def get_fps(self) -> float:
        """Get frames per second"""
        return self.fps
    
    def get_duration(self) -> float:
        """Get video duration in seconds"""
        if self.fps > 0:
            return self.total_frames / self.fps
        return 0.0
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get frame dimensions (width, height)"""
        return self.frame_width, self.frame_height
    
    def is_vr_video(self) -> bool:
        """Check if the video is detected as VR 360"""
        return self.is_vr_360
    
    def extract_frames_batch(self, frame_ids: List[int]) -> Dict[int, np.ndarray]:
        """Extract multiple frames in batch for efficiency"""
        frames = {}
        for frame_id in frame_ids:
            frame = self.get_frame(frame_id)
            if frame is not None:
                frames[frame_id] = frame
        return frames
    
    def preload_frames(self, start_frame: int, end_frame: int):
        """Preload frames into cache for faster access"""
        for frame_id in range(start_frame, min(end_frame + 1, self.total_frames)):
            if frame_id not in self.frame_cache:
                self.get_frame(frame_id)
    
    def clear_cache(self):
        """Clear frame cache"""
        with self.lock:
            self.frame_cache.clear()
    
    def release(self):
        """Release video resources"""
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.frame_cache.clear()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()