import cv2
import numpy as np
import torch
import mediapipe as mp
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class BodyPartPoint:
    """Represents a body part point with coordinates and confidence"""
    x: float
    y: float
    z: float = 0.0
    confidence: float = 0.0
    frame_id: int = 0
    timestamp: float = 0.0

class BodyPartDetector:
    """Detects and tracks body parts in video frames"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose estimation
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        
        # Initialize hand detection
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        
        # Initialize face detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=confidence_threshold
        )
        
        # Initialize face mesh for more detailed face analysis
        self.face_mesh_detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        
        # Body part mapping for MediaPipe landmarks
        self.pose_landmarks_map = {
            'head': [0, 9, 10],  # Nose, mouth landmarks
            'mouth': [9, 10],    # Mouth corners
            'left_shoulder': [11],
            'right_shoulder': [12],
            'left_elbow': [13],
            'right_elbow': [14],
            'left_wrist': [15],
            'right_wrist': [16],
            'left_hip': [23],
            'right_hip': [24],
            'pelvis': [23, 24],  # Hip midpoint
        }
        
        logger.info(f"Body part detector initialized with device: {self.device}")
    
    def detect_body_parts(self, frame: np.ndarray) -> Dict[str, Optional[BodyPartPoint]]:
        """Detect all body parts in a frame"""
        detections = {}
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect pose
            pose_results = self.pose_detector.process(rgb_frame)
            
            # Detect hands
            hand_results = self.hand_detector.process(rgb_frame)
            
            # Detect face
            face_results = self.face_detector.process(rgb_frame)
            face_mesh_results = self.face_mesh_detector.process(rgb_frame)
            
            # Process pose landmarks
            if pose_results.pose_landmarks:
                pose_detections = self._process_pose_landmarks(
                    pose_results.pose_landmarks, frame.shape
                )
                detections.update(pose_detections)
            
            # Process hand landmarks
            if hand_results.multi_hand_landmarks:
                hand_detections = self._process_hand_landmarks(
                    hand_results.multi_hand_landmarks, 
                    hand_results.multi_handedness,
                    frame.shape
                )
                detections.update(hand_detections)
            
            # Process face landmarks
            if face_results.detections:
                face_detections = self._process_face_detections(
                    face_results.detections, frame.shape
                )
                detections.update(face_detections)
            
            # Process face mesh for mouth detection
            if face_mesh_results.multi_face_landmarks:
                mouth_detection = self._process_face_mesh_for_mouth(
                    face_mesh_results.multi_face_landmarks[0], frame.shape
                )
                if mouth_detection:
                    detections['mouth'] = mouth_detection
            
            # Detect additional body parts using custom logic
            additional_detections = self._detect_additional_body_parts(
                frame, pose_results
            )
            detections.update(additional_detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting body parts: {e}")
            return {}
    
    def _process_pose_landmarks(self, landmarks, frame_shape) -> Dict[str, BodyPartPoint]:
        """Process MediaPipe pose landmarks"""
        detections = {}
        height, width = frame_shape[:2]
        
        try:
            # Convert landmarks to pixel coordinates
            landmark_points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                z = landmark.z
                confidence = landmark.visibility
                landmark_points.append((x, y, z, confidence))
            
            # Extract specific body parts
            for part_name, indices in self.pose_landmarks_map.items():
                if indices:
                    # Average multiple landmarks for more stable detection
                    x_coords = [landmark_points[i][0] for i in indices if i < len(landmark_points)]
                    y_coords = [landmark_points[i][1] for i in indices if i < len(landmark_points)]
                    z_coords = [landmark_points[i][2] for i in indices if i < len(landmark_points)]
                    confidences = [landmark_points[i][3] for i in indices if i < len(landmark_points)]
                    
                    if x_coords and y_coords:
                        avg_x = sum(x_coords) / len(x_coords)
                        avg_y = sum(y_coords) / len(y_coords)
                        avg_z = sum(z_coords) / len(z_coords)
                        avg_confidence = sum(confidences) / len(confidences)
                        
                        if avg_confidence >= self.confidence_threshold:
                            detections[part_name] = BodyPartPoint(
                                x=avg_x,
                                y=avg_y,
                                z=avg_z,
                                confidence=avg_confidence
                            )
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing pose landmarks: {e}")
            return {}
    
    def _process_hand_landmarks(self, hand_landmarks_list, handedness_list, frame_shape) -> Dict[str, BodyPartPoint]:
        """Process MediaPipe hand landmarks"""
        detections = {}
        height, width = frame_shape[:2]
        
        try:
            for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
                # Determine if left or right hand
                hand_label = handedness.classification[0].label.lower()
                
                # Get wrist landmark (landmark 0)
                wrist_landmark = hand_landmarks.landmark[0]
                x = int(wrist_landmark.x * width)
                y = int(wrist_landmark.y * height)
                z = wrist_landmark.z
                
                # Create detection
                hand_key = f"hand_{hand_label}"
                detections[hand_key] = BodyPartPoint(
                    x=x,
                    y=y,
                    z=z,
                    confidence=0.8  # MediaPipe doesn't provide confidence for hand landmarks
                )
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing hand landmarks: {e}")
            return {}
    
    def _process_face_detections(self, face_detections, frame_shape) -> Dict[str, BodyPartPoint]:
        """Process MediaPipe face detections"""
        detections = {}
        height, width = frame_shape[:2]
        
        try:
            for detection in face_detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int((bbox.xmin + bbox.width / 2) * width)
                y = int((bbox.ymin + bbox.height / 2) * height)
                confidence = detection.score[0]
                
                if confidence >= self.confidence_threshold:
                    detections['head'] = BodyPartPoint(
                        x=x,
                        y=y,
                        z=0.0,
                        confidence=confidence
                    )
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing face detections: {e}")
            return {}
    
    def _process_face_mesh_for_mouth(self, face_landmarks, frame_shape) -> Optional[BodyPartPoint]:
        """Extract mouth position from face mesh"""
        try:
            height, width = frame_shape[:2]
            
            # Mouth landmarks in MediaPipe face mesh
            mouth_landmarks = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
            
            # Calculate mouth center
            mouth_x = []
            mouth_y = []
            
            for landmark_id in mouth_landmarks:
                if landmark_id < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[landmark_id]
                    mouth_x.append(landmark.x * width)
                    mouth_y.append(landmark.y * height)
            
            if mouth_x and mouth_y:
                avg_x = sum(mouth_x) / len(mouth_x)
                avg_y = sum(mouth_y) / len(mouth_y)
                
                return BodyPartPoint(
                    x=avg_x,
                    y=avg_y,
                    z=0.0,
                    confidence=0.8
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing face mesh for mouth: {e}")
            return None
    
    def _detect_additional_body_parts(self, frame: np.ndarray, pose_results) -> Dict[str, BodyPartPoint]:
        """Detect additional body parts using custom logic"""
        detections = {}
        
        try:
            # This is where we would implement custom detection for:
            # - Breasts
            # - Genitals
            # - More precise pelvis detection
            
            # For now, we'll use basic approximations based on pose landmarks
            if pose_results.pose_landmarks:
                height, width = frame.shape[:2]
                landmarks = pose_results.pose_landmarks.landmark
                
                # Approximate breast position (between shoulders and below)
                if len(landmarks) > 24:  # Ensure we have enough landmarks
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    
                    # Calculate breast approximation
                    breast_x = int((left_shoulder.x + right_shoulder.x) / 2 * width)
                    breast_y = int((left_shoulder.y + right_shoulder.y) / 2 * height + 0.1 * height)
                    
                    detections['breasts'] = BodyPartPoint(
                        x=breast_x,
                        y=breast_y,
                        z=0.0,
                        confidence=0.6  # Lower confidence for approximation
                    )
                    
                    # Approximate genital position (below pelvis)
                    left_hip = landmarks[23]
                    right_hip = landmarks[24]
                    
                    genital_x = int((left_hip.x + right_hip.x) / 2 * width)
                    genital_y = int((left_hip.y + right_hip.y) / 2 * height + 0.05 * height)
                    
                    detections['genitals'] = BodyPartPoint(
                        x=genital_x,
                        y=genital_y,
                        z=0.0,
                        confidence=0.5  # Lower confidence for approximation
                    )
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting additional body parts: {e}")
            return {}
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold for all detectors"""
        self.confidence_threshold = threshold
        
        # Reinitialize detectors with new threshold
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=threshold,
            min_tracking_confidence=threshold
        )
        
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=threshold,
            min_tracking_confidence=threshold
        )
        
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=threshold
        )
        
        self.face_mesh_detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=threshold,
            min_tracking_confidence=threshold
        )
    
    def close(self):
        """Close all detectors"""
        if hasattr(self, 'pose_detector'):
            self.pose_detector.close()
        if hasattr(self, 'hand_detector'):
            self.hand_detector.close()
        if hasattr(self, 'face_detector'):
            self.face_detector.close()
        if hasattr(self, 'face_mesh_detector'):
            self.face_mesh_detector.close()