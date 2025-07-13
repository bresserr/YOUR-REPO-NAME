"""
Pose Detection Module using MediaPipe, TensorRT, and custom models
for advanced body part detection and pose estimation
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import mediapipe as mp
from loguru import logger
import json
from pathlib import Path

try:
    import tensorrt as trt
    from src.utils.cuda_utils import TensorRTEngine
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

from src.utils.config import Config


class PoseDetector:
    """
    Advanced pose detection with specialized body part detection
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose detection models
        self.pose_model = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=config.model.confidence_threshold,
            min_tracking_confidence=config.tracking.tracking_confidence
        )
        
        self.hand_model = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.model.confidence_threshold,
            min_tracking_confidence=config.tracking.tracking_confidence
        )
        
        self.face_model = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.model.confidence_threshold,
            min_tracking_confidence=config.tracking.tracking_confidence
        )
        
        # Body part landmark indices
        self.body_landmarks = self._init_body_landmarks()
        
        # TensorRT engines for specialized detection
        self.trt_engines = {}
        self._load_trt_engines()
        
        # Custom models for adult content detection
        self.adult_detector = None
        self._load_adult_detector()
        
        logger.info("Pose Detector initialized")
    
    def _init_body_landmarks(self) -> Dict[str, List[int]]:
        """Initialize body landmark mappings"""
        return {
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Head and face landmarks
            'mouth': [9, 10, 11, 12, 13, 14, 15, 16],  # Mouth region
            'left_hand': [15, 17, 19, 21],  # Left hand keypoints
            'right_hand': [16, 18, 20, 22],  # Right hand keypoints
            'torso': [11, 12, 23, 24],  # Torso region
            'pelvis': [23, 24, 25, 26, 27, 28],  # Pelvis region
            'left_shoulder': [11, 13, 15],
            'right_shoulder': [12, 14, 16],
            'left_arm': [13, 15, 17, 19, 21],
            'right_arm': [14, 16, 18, 20, 22],
            'left_leg': [23, 25, 27, 29, 31],
            'right_leg': [24, 26, 28, 30, 32]
        }
    
    def _load_trt_engines(self):
        """Load TensorRT engines for specialized detection"""
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available, using fallback methods")
            return
        
        engine_configs = {
            'body_detection': self.config.get_model_path('body'),
            'face_detection': self.config.get_model_path('face'),
            'hand_detection': self.config.get_model_path('hand'),
            'genital_detection': self.config.get_model_path('genital')
        }
        
        for name, path in engine_configs.items():
            if Path(path).exists():
                try:
                    self.trt_engines[name] = TensorRTEngine(path, self.config.cuda.max_batch_size)
                    logger.info(f"Loaded TensorRT engine: {name}")
                except Exception as e:
                    logger.error(f"Failed to load TensorRT engine {name}: {e}")
    
    def _load_adult_detector(self):
        """Load custom adult content detector"""
        try:
            # This would be a custom trained model for adult content detection
            # For now, we'll use a placeholder that can be replaced with actual model
            self.adult_detector = AdultContentDetector(self.config)
            logger.info("Adult content detector loaded")
        except Exception as e:
            logger.error(f"Failed to load adult content detector: {e}")
    
    def detect_poses(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect all poses and body parts in the frame
        """
        results = {
            'frame_shape': frame.shape,
            'timestamp': None,
            'pose_landmarks': None,
            'hand_landmarks': None,
            'face_landmarks': None,
            'body_parts': {},
            'confidence_scores': {},
            'adult_content': {}
        }
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe pose detection
            pose_results = self.pose_model.process(rgb_frame)
            hand_results = self.hand_model.process(rgb_frame)
            face_results = self.face_model.process(rgb_frame)
            
            # Process pose landmarks
            if pose_results.pose_landmarks:
                results['pose_landmarks'] = self._process_pose_landmarks(
                    pose_results.pose_landmarks, frame.shape
                )
                results['body_parts'].update(
                    self._extract_body_parts(results['pose_landmarks'])
                )
            
            # Process hand landmarks
            if hand_results.multi_hand_landmarks:
                results['hand_landmarks'] = self._process_hand_landmarks(
                    hand_results.multi_hand_landmarks, 
                    hand_results.multi_handedness, 
                    frame.shape
                )
                results['body_parts'].update(
                    self._extract_hand_parts(results['hand_landmarks'])
                )
            
            # Process face landmarks
            if face_results.multi_face_landmarks:
                results['face_landmarks'] = self._process_face_landmarks(
                    face_results.multi_face_landmarks, frame.shape
                )
                results['body_parts'].update(
                    self._extract_face_parts(results['face_landmarks'])
                )
            
            # TensorRT-based specialized detection
            if self.trt_engines:
                specialized_results = self._run_specialized_detection(frame)
                results['body_parts'].update(specialized_results)
            
            # Adult content detection
            if self.adult_detector:
                adult_results = self.adult_detector.detect(frame, results['body_parts'])
                results['adult_content'] = adult_results
            
            # Calculate confidence scores
            results['confidence_scores'] = self._calculate_confidence_scores(results)
            
        except Exception as e:
            logger.error(f"Error in pose detection: {e}")
        
        return results
    
    def _process_pose_landmarks(self, landmarks, frame_shape: Tuple[int, int, int]) -> List[Dict[str, float]]:
        """Process MediaPipe pose landmarks"""
        height, width = frame_shape[:2]
        processed_landmarks = []
        
        for landmark in landmarks.landmark:
            processed_landmarks.append({
                'x': landmark.x * width,
                'y': landmark.y * height,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        return processed_landmarks
    
    def _process_hand_landmarks(self, multi_hand_landmarks, multi_handedness, 
                              frame_shape: Tuple[int, int, int]) -> Dict[str, List[Dict[str, float]]]:
        """Process MediaPipe hand landmarks"""
        height, width = frame_shape[:2]
        hand_data = {}
        
        for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
            hand_label = handedness.classification[0].label.lower()
            
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x * width,
                    'y': landmark.y * height,
                    'z': landmark.z
                })
            
            hand_data[f'{hand_label}_hand'] = landmarks
        
        return hand_data
    
    def _process_face_landmarks(self, multi_face_landmarks, 
                               frame_shape: Tuple[int, int, int]) -> List[Dict[str, float]]:
        """Process MediaPipe face landmarks"""
        height, width = frame_shape[:2]
        face_landmarks = []
        
        for face_landmark in multi_face_landmarks:
            landmarks = []
            for landmark in face_landmark.landmark:
                landmarks.append({
                    'x': landmark.x * width,
                    'y': landmark.y * height,
                    'z': landmark.z
                })
            face_landmarks.append(landmarks)
        
        return face_landmarks
    
    def _extract_body_parts(self, pose_landmarks: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Extract specific body parts from pose landmarks"""
        body_parts = {}
        
        for part_name, landmark_indices in self.body_landmarks.items():
            if all(idx < len(pose_landmarks) for idx in landmark_indices):
                # Calculate centroid of the body part
                x_coords = [pose_landmarks[idx]['x'] for idx in landmark_indices]
                y_coords = [pose_landmarks[idx]['y'] for idx in landmark_indices]
                z_coords = [pose_landmarks[idx]['z'] for idx in landmark_indices]
                
                body_parts[part_name] = {
                    'x': np.mean(x_coords),
                    'y': np.mean(y_coords),
                    'z': np.mean(z_coords),
                    'confidence': np.mean([pose_landmarks[idx]['visibility'] for idx in landmark_indices])
                }
        
        return body_parts
    
    def _extract_hand_parts(self, hand_landmarks: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """Extract hand positions"""
        hand_parts = {}
        
        for hand_name, landmarks in hand_landmarks.items():
            if landmarks:
                # Use wrist position (landmark 0) as hand center
                wrist = landmarks[0]
                hand_parts[hand_name] = {
                    'x': wrist['x'],
                    'y': wrist['y'],
                    'z': wrist['z'],
                    'confidence': 1.0  # MediaPipe hands are generally reliable
                }
        
        return hand_parts
    
    def _extract_face_parts(self, face_landmarks: List[List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """Extract face parts"""
        face_parts = {}
        
        if face_landmarks:
            landmarks = face_landmarks[0]  # Use first face
            
            # Head center (approximate)
            if len(landmarks) > 10:
                head_landmarks = landmarks[1:11]  # Approximate head region
                face_parts['head'] = {
                    'x': np.mean([lm['x'] for lm in head_landmarks]),
                    'y': np.mean([lm['y'] for lm in head_landmarks]),
                    'z': np.mean([lm['z'] for lm in head_landmarks]),
                    'confidence': 0.8
                }
            
            # Mouth region
            if len(landmarks) > 17:
                mouth_landmarks = landmarks[13:17]  # Approximate mouth region
                face_parts['mouth'] = {
                    'x': np.mean([lm['x'] for lm in mouth_landmarks]),
                    'y': np.mean([lm['y'] for lm in mouth_landmarks]),
                    'z': np.mean([lm['z'] for lm in mouth_landmarks]),
                    'confidence': 0.8
                }
        
        return face_parts
    
    def _run_specialized_detection(self, frame: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Run TensorRT-based specialized detection"""
        specialized_parts = {}
        
        try:
            # Prepare input
            input_frame = cv2.resize(frame, (640, 480))
            input_frame = input_frame.astype(np.float32) / 255.0
            input_frame = np.transpose(input_frame, (2, 0, 1))
            input_frame = np.expand_dims(input_frame, axis=0)
            
            # Run specialized detectors
            for detector_name, engine in self.trt_engines.items():
                try:
                    outputs = engine.infer(input_frame)
                    
                    if detector_name == 'body_detection':
                        body_results = self._process_body_detection(outputs)
                        specialized_parts.update(body_results)
                    
                    elif detector_name == 'genital_detection':
                        genital_results = self._process_genital_detection(outputs)
                        specialized_parts.update(genital_results)
                    
                except Exception as e:
                    logger.error(f"Error in {detector_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error in specialized detection: {e}")
        
        return specialized_parts
    
    def _process_body_detection(self, outputs: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Process body detection results"""
        results = {}
        
        if outputs:
            # Assuming output format: [class_id, confidence, x, y, width, height]
            detections = outputs[0]
            
            for detection in detections:
                if len(detection) >= 6:
                    class_id, confidence, x, y, w, h = detection[:6]
                    
                    if confidence > self.config.model.confidence_threshold:
                        # Map class_id to body part name
                        part_name = self._map_class_to_body_part(int(class_id))
                        
                        if part_name:
                            results[part_name] = {
                                'x': x + w/2,
                                'y': y + h/2,
                                'z': 0.0,
                                'confidence': confidence,
                                'bbox': [x, y, w, h]
                            }
        
        return results
    
    def _process_genital_detection(self, outputs: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Process genital detection results"""
        results = {}
        
        if outputs:
            detections = outputs[0]
            
            for detection in detections:
                if len(detection) >= 6:
                    class_id, confidence, x, y, w, h = detection[:6]
                    
                    if confidence > self.config.model.confidence_threshold:
                        # Map to genital types
                        if class_id == 0:  # Penis
                            results['penis'] = {
                                'x': x + w/2,
                                'y': y + h/2,
                                'z': 0.0,
                                'confidence': confidence,
                                'bbox': [x, y, w, h]
                            }
                        elif class_id == 1:  # Vagina
                            results['vagina'] = {
                                'x': x + w/2,
                                'y': y + h/2,
                                'z': 0.0,
                                'confidence': confidence,
                                'bbox': [x, y, w, h]
                            }
        
        return results
    
    def _map_class_to_body_part(self, class_id: int) -> Optional[str]:
        """Map detection class ID to body part name"""
        class_mapping = {
            0: 'head',
            1: 'torso',
            2: 'left_arm',
            3: 'right_arm',
            4: 'left_leg',
            5: 'right_leg',
            6: 'breasts',
            7: 'pelvis'
        }
        return class_mapping.get(class_id)
    
    def _calculate_confidence_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall confidence scores"""
        confidence_scores = {}
        
        for part_name, part_data in results.get('body_parts', {}).items():
            if 'confidence' in part_data:
                confidence_scores[part_name] = part_data['confidence']
        
        # Overall detection confidence
        if confidence_scores:
            confidence_scores['overall'] = np.mean(list(confidence_scores.values()))
        else:
            confidence_scores['overall'] = 0.0
        
        return confidence_scores
    
    def visualize_detections(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Visualize detection results on frame"""
        vis_frame = frame.copy()
        
        try:
            # Draw body parts
            for part_name, part_data in results.get('body_parts', {}).items():
                x, y = int(part_data['x']), int(part_data['y'])
                confidence = part_data.get('confidence', 0.0)
                
                # Color based on confidence
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
                
                # Draw circle for body part
                cv2.circle(vis_frame, (x, y), 5, color, -1)
                
                # Draw label
                label = f"{part_name}: {confidence:.2f}"
                cv2.putText(vis_frame, label, (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw bounding box if available
                if 'bbox' in part_data:
                    bbox = part_data['bbox']
                    cv2.rectangle(vis_frame, (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color, 2)
        
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
        
        return vis_frame
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'pose_model'):
            self.pose_model.close()
        if hasattr(self, 'hand_model'):
            self.hand_model.close()
        if hasattr(self, 'face_model'):
            self.face_model.close()
        
        logger.info("Pose Detector cleaned up")


class AdultContentDetector:
    """
    Specialized detector for adult content body parts
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # This would be replaced with actual trained models
        self.breast_detector = None
        self.genital_detector = None
        
        logger.info("Adult Content Detector initialized")
    
    def detect(self, frame: np.ndarray, existing_parts: Dict[str, Any]) -> Dict[str, Any]:
        """Detect adult content body parts"""
        results = {
            'adult_parts_detected': [],
            'confidence_scores': {},
            'estimated_positions': {}
        }
        
        try:
            # Use existing body parts to estimate adult content locations
            if 'torso' in existing_parts:
                torso = existing_parts['torso']
                
                # Estimate breast location
                breast_y = torso['y'] - 50  # Approximate offset
                results['estimated_positions']['breasts'] = {
                    'x': torso['x'],
                    'y': breast_y,
                    'z': torso['z'],
                    'confidence': 0.6  # Lower confidence for estimation
                }
                results['adult_parts_detected'].append('breasts')
            
            if 'pelvis' in existing_parts:
                pelvis = existing_parts['pelvis']
                
                # Estimate genital location
                results['estimated_positions']['genitals'] = {
                    'x': pelvis['x'],
                    'y': pelvis['y'],
                    'z': pelvis['z'],
                    'confidence': 0.5  # Lower confidence for estimation
                }
                results['adult_parts_detected'].append('genitals')
        
        except Exception as e:
            logger.error(f"Error in adult content detection: {e}")
        
        return results