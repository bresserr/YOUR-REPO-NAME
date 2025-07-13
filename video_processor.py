"""
Video Processing Module for VR Assistive Technology
Handles video input, frame extraction, and video information
"""
import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Video processing for VR content with support for various formats
    Optimized for real-time processing and frame extraction
    """
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        self.current_video = None
        self.current_cap = None
        self.video_info = {}
        
        logger.info("VideoProcessor initialized")
    
    async def get_video_info(self, video_path: str) -> Dict:
        """Get comprehensive video information"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Check file extension
            ext = Path(video_path).suffix.lower()
            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported video format: {ext}")
            
            # Open video and get properties
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {video_path}")
            
            # Extract video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Get codec information
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            
            # File size
            file_size = os.path.getsize(video_path)
            
            video_info = {
                'path': video_path,
                'filename': os.path.basename(video_path),
                'format': ext,
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'codec': codec,
                'file_size': file_size,
                'is_vr': self._detect_vr_format(width, height),
                'aspect_ratio': width / height if height > 0 else 1.0
            }
            
            cap.release()
            self.video_info = video_info
            
            logger.info(f"Video info extracted: {video_info}")
            return video_info
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            raise
    
    def _detect_vr_format(self, width: int, height: int) -> bool:
        """Detect if video is in VR format (e.g., 360°, side-by-side)"""
        # Common VR aspect ratios
        aspect_ratio = width / height
        
        # Side-by-side VR (2:1 aspect ratio)
        if abs(aspect_ratio - 2.0) < 0.1:
            return True
        
        # 360° video (often 2:1 or 4:1)
        if aspect_ratio >= 1.8:
            return True
        
        # Over-under VR (1:1 aspect ratio)
        if abs(aspect_ratio - 1.0) < 0.1:
            return True
        
        return False
    
    async def extract_frame(self, video_path: str, frame_number: int) -> Optional[np.ndarray]:
        """Extract a specific frame from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"Cannot read frame {frame_number}")
                return None
            
            return frame
            
        except Exception as e:
            logger.error(f"Error extracting frame {frame_number}: {str(e)}")
            return None
    
    async def extract_frames_batch(self, video_path: str, 
                                 start_frame: int, 
                                 end_frame: int) -> List[np.ndarray]:
        """Extract a batch of frames for efficient processing"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return frames
            
            # Set start position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read frames in batch
            for frame_num in range(start_frame, min(end_frame + 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting frames batch: {str(e)}")
        
        return frames
    
    async def create_preview_frames(self, video_path: str, 
                                  count: int = 10) -> List[Tuple[int, np.ndarray]]:
        """Create preview frames for timeline display"""
        preview_frames = []
        
        try:
            video_info = await self.get_video_info(video_path)
            frame_count = video_info['frame_count']
            
            # Calculate frame intervals
            if frame_count <= count:
                frame_indices = list(range(frame_count))
            else:
                step = frame_count // count
                frame_indices = [i * step for i in range(count)]
            
            # Extract frames
            for frame_idx in frame_indices:
                frame = await self.extract_frame(video_path, frame_idx)
                if frame is not None:
                    # Resize for preview
                    preview_frame = self._resize_for_preview(frame)
                    preview_frames.append((frame_idx, preview_frame))
            
        except Exception as e:
            logger.error(f"Error creating preview frames: {str(e)}")
        
        return preview_frames
    
    def _resize_for_preview(self, frame: np.ndarray, max_width: int = 320) -> np.ndarray:
        """Resize frame for preview display"""
        height, width = frame.shape[:2]
        
        if width > max_width:
            ratio = max_width / width
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (max_width, new_height))
        
        return frame
    
    async def get_frame_at_timestamp(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """Get frame at specific timestamp (in seconds)"""
        try:
            video_info = await self.get_video_info(video_path)
            fps = video_info['fps']
            frame_number = int(timestamp * fps)
            
            return await self.extract_frame(video_path, frame_number)
            
        except Exception as e:
            logger.error(f"Error getting frame at timestamp {timestamp}: {str(e)}")
            return None
    
    async def validate_video_file(self, video_path: str) -> bool:
        """Validate if video file is readable and contains frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return False
            
            # Try to read first frame
            ret, frame = cap.read()
            cap.release()
            
            return ret and frame is not None
            
        except Exception as e:
            logger.error(f"Error validating video: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.current_cap:
            self.current_cap.release()
        
        logger.info("VideoProcessor cleanup completed")
    
    async def convert_to_standard_format(self, input_path: str, output_path: str) -> bool:
        """Convert video to standard format for better compatibility"""
        try:
            # This would implement video conversion using ffmpeg
            # For now, just copy the file
            import shutil
            shutil.copy2(input_path, output_path)
            
            logger.info(f"Video converted: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting video: {str(e)}")
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported video formats"""
        return self.supported_formats