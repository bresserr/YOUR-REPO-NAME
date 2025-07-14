import cv2
import numpy as np
import torch
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class BodyPartDetector:
    def __init__(self):
        # Initialize MediaPipe for initial pose detection
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Body part mappings
        self.body_parts = {
            'head': None,
            'mouth': None,
            'hand_left': None,
            'hand_right': None,
            'breasts': None,
            'pelvis': None,
            'genitals': None
        }
        
        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
    def detect_body_parts(self, frame: np.ndarray, is_vr: bool = True) -> Dict[str, Tuple[int, int]]:
        """Detect body parts in a frame"""
        if is_vr:
            # Split VR frame into left and right views
            height, width = frame.shape[:2]
            left_frame = frame[:, :width//2]
            right_frame = frame[:, width//2:]
            
            # Process left eye view (typically used for detection)
            return self._process_frame(left_frame)
        else:
            return self._process_frame(frame)
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """Process a single frame to detect body parts"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run holistic model
        results = self.holistic.process(rgb_frame)
        
        detected_parts = {}
        height, width = frame.shape[:2]
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Head (average of nose, left eye, right eye)
            if landmarks[0].visibility > 0.5:  # Nose
                head_x = int((landmarks[0].x + landmarks[2].x + landmarks[5].x) / 3 * width)
                head_y = int((landmarks[0].y + landmarks[2].y + landmarks[5].y) / 3 * height)
                detected_parts['head'] = (head_x, head_y)
            
            # Mouth
            if results.face_landmarks:
                mouth_landmarks = results.face_landmarks.landmark
                mouth_x = int(mouth_landmarks[13].x * width)  # Upper lip
                mouth_y = int(mouth_landmarks[13].y * height)
                detected_parts['mouth'] = (mouth_x, mouth_y)
            
            # Hands
            if landmarks[15].visibility > 0.5:  # Left wrist
                detected_parts['hand_left'] = (
                    int(landmarks[15].x * width),
                    int(landmarks[15].y * height)
                )
            
            if landmarks[16].visibility > 0.5:  # Right wrist
                detected_parts['hand_right'] = (
                    int(landmarks[16].x * width),
                    int(landmarks[16].y * height)
                )
            
            # Breasts (estimated from shoulders and chest)
            if landmarks[11].visibility > 0.5 and landmarks[12].visibility > 0.5:
                chest_x = int((landmarks[11].x + landmarks[12].x) / 2 * width)
                chest_y = int((landmarks[11].y + landmarks[12].y) / 2 * height)
                # Estimate breast position slightly below shoulders
                breast_y = chest_y + int(0.1 * height)
                detected_parts['breasts'] = (chest_x, breast_y)
            
            # Pelvis (hip center)
            if landmarks[23].visibility > 0.5 and landmarks[24].visibility > 0.5:
                pelvis_x = int((landmarks[23].x + landmarks[24].x) / 2 * width)
                pelvis_y = int((landmarks[23].y + landmarks[24].y) / 2 * height)
                detected_parts['pelvis'] = (pelvis_x, pelvis_y)
                
                # Estimate genital position
                genital_y = pelvis_y + int(0.05 * height)
                detected_parts['genitals'] = (pelvis_x, genital_y)
        
        return detected_parts
    
    def track_movement(self, positions_history: List[Dict[str, Tuple[int, int]]]) -> Dict[str, float]:
        """Calculate movement speed and direction for tracked parts"""
        if len(positions_history) < 2:
            return {}
        
        movements = {}
        
        for part in self.body_parts.keys():
            if part in positions_history[-1] and part in positions_history[-2]:
                current = positions_history[-1][part]
                previous = positions_history[-2][part]
                
                # Calculate displacement
                dx = current[0] - previous[0]
                dy = current[1] - previous[1]
                
                # Calculate speed (pixels per frame)
                speed = np.sqrt(dx**2 + dy**2)
                movements[part] = speed
        
        return movements
    
    def detect_interactions(self, positions: Dict[str, Tuple[int, int]], 
                          threshold: int = 50) -> List[Tuple[str, str, float]]:
        """Detect interactions between body parts"""
        interactions = []
        
        if 'genitals' not in positions:
            return interactions
        
        genital_pos = positions['genitals']
        
        for part, pos in positions.items():
            if part != 'genitals' and pos is not None:
                # Calculate distance
                distance = np.sqrt(
                    (pos[0] - genital_pos[0])**2 + 
                    (pos[1] - genital_pos[1])**2
                )
                
                if distance < threshold:
                    interactions.append((part, 'genitals', distance))
        
        return interactions