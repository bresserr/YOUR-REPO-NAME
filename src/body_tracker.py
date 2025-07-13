"""
Body Tracker Module for tracking body parts across frames
with Kalman filters and temporal consistency management
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import time
from dataclasses import dataclass
from loguru import logger
import math

from src.utils.config import Config


@dataclass
class BodyPartTrack:
    """Individual body part tracking data"""
    track_id: int
    part_name: str
    positions: deque  # Recent positions
    velocities: deque  # Recent velocities
    confidences: deque  # Recent confidence scores
    kalman_filter: Optional[cv2.KalmanFilter]
    last_update: float
    is_active: bool
    lost_frames: int
    tracking_quality: float


class BodyTracker:
    """
    Advanced body part tracking with Kalman filters and temporal consistency
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.tracking_config = config.tracking
        
        # Tracking state
        self.active_tracks: Dict[str, BodyPartTrack] = {}
        self.track_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.next_track_id = 0
        
        # Tracking parameters
        self.max_lost_frames = self.tracking_config.max_lost_frames
        self.tracking_confidence_threshold = self.tracking_config.tracking_confidence
        self.smoothing_factor = self.tracking_config.smoothing_factor
        
        # Distance thresholds for track matching
        self.distance_thresholds = {
            'head': 50.0,
            'mouth': 30.0,
            'hand_1': 40.0,
            'hand_2': 40.0,
            'left_hand': 40.0,
            'right_hand': 40.0,
            'breasts': 60.0,
            'pelvis': 50.0,
            'genitals': 40.0,
            'penis': 40.0,
            'vagina': 40.0
        }
        
        logger.info("Body Tracker initialized")
    
    def track_body_parts(self, pose_results: Dict[str, Any], frame_number: int, 
                        requested_parts: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Track body parts across frames with temporal consistency
        """
        current_time = time.time()
        tracked_parts = {}
        
        try:
            # Get detected body parts
            detected_parts = pose_results.get('body_parts', {})
            
            # Add adult content detections if available
            if 'adult_content' in pose_results:
                adult_parts = pose_results['adult_content'].get('estimated_positions', {})
                detected_parts.update(adult_parts)
            
            # Update existing tracks
            self._update_existing_tracks(detected_parts, current_time, frame_number)
            
            # Create new tracks for unmatched detections
            self._create_new_tracks(detected_parts, current_time, frame_number)
            
            # Clean up lost tracks
            self._cleanup_lost_tracks()
            
            # Generate tracking results
            for part_name in requested_parts:
                if part_name in self.active_tracks:
                    track = self.active_tracks[part_name]
                    if track.is_active:
                        tracked_parts[part_name] = self._get_track_output(track, frame_number)
                elif part_name in detected_parts:
                    # Use raw detection if no track exists
                    tracked_parts[part_name] = detected_parts[part_name]
            
            # Update tracking history
            self._update_tracking_history(tracked_parts, frame_number)
            
        except Exception as e:
            logger.error(f"Error in body tracking: {e}")
        
        return tracked_parts
    
    def _update_existing_tracks(self, detected_parts: Dict[str, Dict[str, Any]], 
                               current_time: float, frame_number: int):
        """Update existing tracks with new detections"""
        matched_detections = set()
        
        for part_name, track in self.active_tracks.items():
            if not track.is_active:
                continue
            
            # Find best matching detection
            best_match = None
            best_distance = float('inf')
            threshold = self.distance_thresholds.get(part_name, 50.0)
            
            # Get predicted position from Kalman filter
            predicted_pos = self._predict_position(track)
            
            # Check all detections for matches
            for detection_name, detection_data in detected_parts.items():
                if detection_name in matched_detections:
                    continue
                
                # Allow flexible matching (e.g., hand_1 can match left_hand)
                if self._can_match_parts(part_name, detection_name):
                    detection_pos = np.array([detection_data['x'], detection_data['y']])
                    distance = np.linalg.norm(predicted_pos - detection_pos)
                    
                    if distance < threshold and distance < best_distance:
                        best_match = detection_name
                        best_distance = distance
            
            if best_match:
                # Update track with matched detection
                detection_data = detected_parts[best_match]
                self._update_track(track, detection_data, current_time, frame_number)
                matched_detections.add(best_match)
                track.lost_frames = 0
                track.is_active = True
            else:
                # No match found, increment lost frames
                track.lost_frames += 1
                if track.lost_frames > self.max_lost_frames:
                    track.is_active = False
                
                # Use Kalman filter prediction
                self._predict_track_position(track, current_time, frame_number)
    
    def _create_new_tracks(self, detected_parts: Dict[str, Dict[str, Any]], 
                          current_time: float, frame_number: int):
        """Create new tracks for unmatched detections"""
        for part_name, detection_data in detected_parts.items():
            # Check if this part is already being tracked
            if part_name not in self.active_tracks or not self.active_tracks[part_name].is_active:
                # Create new track
                track = self._create_track(part_name, detection_data, current_time, frame_number)
                self.active_tracks[part_name] = track
    
    def _create_track(self, part_name: str, detection_data: Dict[str, Any], 
                     current_time: float, frame_number: int) -> BodyPartTrack:
        """Create a new body part track"""
        track = BodyPartTrack(
            track_id=self.next_track_id,
            part_name=part_name,
            positions=deque(maxlen=10),
            velocities=deque(maxlen=10),
            confidences=deque(maxlen=10),
            kalman_filter=self._create_kalman_filter(),
            last_update=current_time,
            is_active=True,
            lost_frames=0,
            tracking_quality=detection_data.get('confidence', 0.5)
        )
        
        self.next_track_id += 1
        
        # Initialize with first detection
        position = np.array([detection_data['x'], detection_data['y']])
        track.positions.append(position)
        track.confidences.append(detection_data.get('confidence', 0.5))
        
        # Initialize Kalman filter
        self._initialize_kalman_filter(track.kalman_filter, position)
        
        return track
    
    def _create_kalman_filter(self) -> cv2.KalmanFilter:
        """Create Kalman filter for position and velocity tracking"""
        # State: [x, y, vx, vy]
        kalman = cv2.KalmanFilter(4, 2)
        
        # Transition matrix (constant velocity model)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (observe position only)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance
        kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        
        return kalman
    
    def _initialize_kalman_filter(self, kalman: cv2.KalmanFilter, initial_position: np.ndarray):
        """Initialize Kalman filter state"""
        kalman.statePre = np.array([
            initial_position[0],
            initial_position[1],
            0.0,  # Initial velocity x
            0.0   # Initial velocity y
        ], dtype=np.float32)
        
        kalman.statePost = kalman.statePre.copy()
    
    def _predict_position(self, track: BodyPartTrack) -> np.ndarray:
        """Predict next position using Kalman filter"""
        if track.kalman_filter is None:
            # Fallback to last known position
            return track.positions[-1] if track.positions else np.array([0.0, 0.0])
        
        prediction = track.kalman_filter.predict()
        return prediction[:2]
    
    def _update_track(self, track: BodyPartTrack, detection_data: Dict[str, Any], 
                     current_time: float, frame_number: int):
        """Update track with new detection"""
        position = np.array([detection_data['x'], detection_data['y']])
        confidence = detection_data.get('confidence', 0.5)
        
        # Update Kalman filter
        if track.kalman_filter is not None:
            track.kalman_filter.correct(position.astype(np.float32))
            
            # Get smoothed position
            state = track.kalman_filter.statePost
            smoothed_position = state[:2]
        else:
            smoothed_position = position
        
        # Calculate velocity
        if len(track.positions) > 0:
            dt = current_time - track.last_update
            if dt > 0:
                velocity = (smoothed_position - track.positions[-1]) / dt
                track.velocities.append(velocity)
        
        # Update track data
        track.positions.append(smoothed_position)
        track.confidences.append(confidence)
        track.last_update = current_time
        
        # Update tracking quality
        track.tracking_quality = self._calculate_tracking_quality(track)
    
    def _predict_track_position(self, track: BodyPartTrack, current_time: float, frame_number: int):
        """Predict position for lost track"""
        if track.kalman_filter is None:
            return
        
        # Use Kalman filter prediction
        prediction = track.kalman_filter.predict()
        predicted_position = prediction[:2]
        
        # Add predicted position with low confidence
        track.positions.append(predicted_position)
        track.confidences.append(0.3)  # Low confidence for prediction
        track.last_update = current_time
    
    def _calculate_tracking_quality(self, track: BodyPartTrack) -> float:
        """Calculate tracking quality based on recent performance"""
        if not track.confidences:
            return 0.0
        
        # Average confidence
        avg_confidence = np.mean(list(track.confidences))
        
        # Stability (inverse of position variance)
        if len(track.positions) > 1:
            positions = np.array(track.positions)
            stability = 1.0 / (1.0 + np.var(positions))
        else:
            stability = 0.5
        
        # Continuity (inverse of lost frames)
        continuity = 1.0 / (1.0 + track.lost_frames)
        
        return 0.5 * avg_confidence + 0.3 * stability + 0.2 * continuity
    
    def _can_match_parts(self, track_name: str, detection_name: str) -> bool:
        """Check if track and detection can be matched"""
        # Direct match
        if track_name == detection_name:
            return True
        
        # Hand matching flexibility
        hand_mappings = {
            'hand_1': ['left_hand', 'right_hand'],
            'hand_2': ['left_hand', 'right_hand'],
            'left_hand': ['hand_1', 'hand_2'],
            'right_hand': ['hand_1', 'hand_2']
        }
        
        if track_name in hand_mappings and detection_name in hand_mappings[track_name]:
            return True
        
        # Genital matching
        genital_mappings = {
            'genitals': ['penis', 'vagina'],
            'penis': ['genitals'],
            'vagina': ['genitals']
        }
        
        if track_name in genital_mappings and detection_name in genital_mappings[track_name]:
            return True
        
        return False
    
    def _cleanup_lost_tracks(self):
        """Remove tracks that have been lost for too long"""
        tracks_to_remove = []
        
        for part_name, track in self.active_tracks.items():
            if not track.is_active and track.lost_frames > self.max_lost_frames * 2:
                tracks_to_remove.append(part_name)
        
        for part_name in tracks_to_remove:
            del self.active_tracks[part_name]
    
    def _get_track_output(self, track: BodyPartTrack, frame_number: int) -> Dict[str, Any]:
        """Get current track output"""
        if not track.positions:
            return {}
        
        current_pos = track.positions[-1]
        current_velocity = track.velocities[-1] if track.velocities else np.array([0.0, 0.0])
        
        return {
            'x': float(current_pos[0]),
            'y': float(current_pos[1]),
            'z': 0.0,  # 2D tracking for now
            'confidence': float(track.confidences[-1]) if track.confidences else 0.0,
            'tracking_quality': track.tracking_quality,
            'velocity': {
                'x': float(current_velocity[0]),
                'y': float(current_velocity[1])
            },
            'track_id': track.track_id,
            'frame_number': frame_number,
            'is_predicted': track.lost_frames > 0
        }
    
    def _update_tracking_history(self, tracked_parts: Dict[str, Dict[str, Any]], frame_number: int):
        """Update tracking history for analysis"""
        for part_name, part_data in tracked_parts.items():
            history_entry = {
                'frame_number': frame_number,
                'timestamp': time.time(),
                **part_data
            }
            self.track_history[part_name].append(history_entry)
            
            # Keep only recent history
            if len(self.track_history[part_name]) > 1000:
                self.track_history[part_name] = self.track_history[part_name][-500:]
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        stats = {
            'active_tracks': len([t for t in self.active_tracks.values() if t.is_active]),
            'total_tracks': len(self.active_tracks),
            'track_qualities': {},
            'average_quality': 0.0,
            'lost_tracks': len([t for t in self.active_tracks.values() if not t.is_active])
        }
        
        qualities = []
        for part_name, track in self.active_tracks.items():
            if track.is_active:
                quality = track.tracking_quality
                stats['track_qualities'][part_name] = quality
                qualities.append(quality)
        
        if qualities:
            stats['average_quality'] = np.mean(qualities)
        
        return stats
    
    def get_motion_analysis(self, part_name: str, frames_back: int = 30) -> Dict[str, Any]:
        """Analyze motion for a specific body part"""
        if part_name not in self.track_history:
            return {}
        
        history = self.track_history[part_name][-frames_back:]
        
        if len(history) < 2:
            return {}
        
        # Extract positions and velocities
        positions = np.array([[h['x'], h['y']] for h in history])
        velocities = []
        
        for i in range(1, len(history)):
            dt = history[i]['timestamp'] - history[i-1]['timestamp']
            if dt > 0:
                vel = (positions[i] - positions[i-1]) / dt
                velocities.append(vel)
        
        velocities = np.array(velocities)
        
        # Calculate motion statistics
        motion_stats = {
            'part_name': part_name,
            'frames_analyzed': len(history),
            'total_distance': 0.0,
            'average_speed': 0.0,
            'max_speed': 0.0,
            'acceleration': 0.0,
            'direction_changes': 0,
            'motion_smoothness': 0.0
        }
        
        if len(velocities) > 0:
            # Distance and speed
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            motion_stats['total_distance'] = np.sum(distances)
            
            speeds = np.linalg.norm(velocities, axis=1)
            motion_stats['average_speed'] = np.mean(speeds)
            motion_stats['max_speed'] = np.max(speeds)
            
            # Acceleration
            if len(velocities) > 1:
                accelerations = np.diff(velocities, axis=0)
                motion_stats['acceleration'] = np.mean(np.linalg.norm(accelerations, axis=1))
            
            # Direction changes
            if len(velocities) > 1:
                directions = velocities / (np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-8)
                direction_changes = np.sum(np.abs(np.diff(directions, axis=0)) > 0.5)
                motion_stats['direction_changes'] = direction_changes
            
            # Smoothness (inverse of jerk)
            if len(velocities) > 2:
                jerk = np.diff(velocities, n=2, axis=0)
                motion_stats['motion_smoothness'] = 1.0 / (1.0 + np.mean(np.linalg.norm(jerk, axis=1)))
        
        return motion_stats
    
    def get_interaction_analysis(self, part1: str, part2: str, frames_back: int = 30) -> Dict[str, Any]:
        """Analyze interaction between two body parts"""
        if part1 not in self.track_history or part2 not in self.track_history:
            return {}
        
        history1 = self.track_history[part1][-frames_back:]
        history2 = self.track_history[part2][-frames_back:]
        
        # Align histories by frame number
        aligned_data = []
        for h1 in history1:
            for h2 in history2:
                if h1['frame_number'] == h2['frame_number']:
                    aligned_data.append((h1, h2))
                    break
        
        if len(aligned_data) < 2:
            return {}
        
        # Calculate interaction metrics
        distances = []
        relative_velocities = []
        
        for h1, h2 in aligned_data:
            pos1 = np.array([h1['x'], h1['y']])
            pos2 = np.array([h2['x'], h2['y']])
            
            distance = np.linalg.norm(pos1 - pos2)
            distances.append(distance)
            
            vel1 = np.array([h1.get('velocity', {}).get('x', 0), h1.get('velocity', {}).get('y', 0)])
            vel2 = np.array([h2.get('velocity', {}).get('x', 0), h2.get('velocity', {}).get('y', 0)])
            
            rel_vel = np.linalg.norm(vel1 - vel2)
            relative_velocities.append(rel_vel)
        
        interaction_stats = {
            'part1': part1,
            'part2': part2,
            'frames_analyzed': len(aligned_data),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'avg_distance': np.mean(distances),
            'avg_relative_velocity': np.mean(relative_velocities),
            'interaction_intensity': 0.0,
            'contact_frames': 0
        }
        
        # Define interaction thresholds
        contact_threshold = self.config.funscript.interaction_distance_threshold * 1000  # Convert to pixels
        
        # Count contact frames
        contact_frames = np.sum(np.array(distances) < contact_threshold)
        interaction_stats['contact_frames'] = contact_frames
        
        # Calculate interaction intensity
        if distances:
            # Inverse of average distance, weighted by relative velocity
            avg_distance = np.mean(distances)
            avg_rel_velocity = np.mean(relative_velocities)
            interaction_stats['interaction_intensity'] = (avg_rel_velocity / (avg_distance + 1.0))
        
        return interaction_stats
    
    def cleanup(self):
        """Clean up tracking resources"""
        self.active_tracks.clear()
        self.track_history.clear()
        logger.info("Body Tracker cleaned up")