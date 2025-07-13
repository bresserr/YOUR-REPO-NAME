import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import torch

logger = logging.getLogger(__name__)

@dataclass
class MovementMetrics:
    """Metrics for body part movement analysis"""
    velocity: float = 0.0
    acceleration: float = 0.0
    distance: float = 0.0
    direction: float = 0.0  # Angle in radians
    smoothed_velocity: float = 0.0
    peak_velocity: float = 0.0
    avg_velocity: float = 0.0

@dataclass
class InteractionEvent:
    """Represents an interaction between body parts"""
    timestamp: float
    frame_id: int
    primary_part: str
    secondary_part: str
    interaction_type: str  # 'contact', 'proximity', 'movement_towards'
    intensity: float
    duration: float = 0.0
    distance: float = 0.0

class MovementAnalyzer:
    """Analyzes body part movements and interactions for funscript generation"""
    
    def __init__(self, smoothing_window: int = 5):
        self.smoothing_window = smoothing_window
        self.interaction_threshold = 50.0  # Pixels
        self.velocity_threshold = 5.0  # Pixels per frame
        self.contact_threshold = 30.0  # Pixels
        
        # Movement history for analysis
        self.movement_history = {}
        self.interaction_events = []
        
        # Analysis parameters
        self.min_interaction_duration = 0.5  # seconds
        self.max_interaction_gap = 0.3  # seconds
        
        logger.info("Movement analyzer initialized")
    
    def analyze_detections(self, detections_history: List) -> Dict[str, Any]:
        """Analyze movement patterns from detection history"""
        try:
            if not detections_history:
                return {"error": "No detection history provided"}
            
            # Calculate movement metrics for each body part
            movement_analysis = self._calculate_movement_metrics(detections_history)
            
            # Detect interactions between body parts
            interactions = self._detect_interactions(detections_history)
            
            # Analyze genital interactions specifically
            genital_analysis = self._analyze_genital_interactions(detections_history, interactions)
            
            # Generate movement patterns for funscript
            movement_patterns = self._generate_movement_patterns(genital_analysis)
            
            return {
                "movement_metrics": movement_analysis,
                "interactions": interactions,
                "genital_analysis": genital_analysis,
                "movement_patterns": movement_patterns,
                "total_frames": len(detections_history),
                "duration": detections_history[-1].timestamp if detections_history else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing detections: {e}")
            return {"error": f"Analysis error: {str(e)}"}
    
    def _calculate_movement_metrics(self, detections_history: List) -> Dict[str, Dict[str, Any]]:
        """Calculate movement metrics for each body part"""
        metrics = {}
        
        try:
            # Get all body part names
            body_parts = set()
            for detection in detections_history:
                for part in ['head', 'mouth', 'hand_left', 'hand_right', 'breasts', 'pelvis', 'genitals']:
                    if getattr(detection, part) is not None:
                        body_parts.add(part)
            
            # Calculate metrics for each body part
            for part_name in body_parts:
                part_metrics = self._calculate_part_metrics(detections_history, part_name)
                if part_metrics:
                    metrics[part_name] = part_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating movement metrics: {e}")
            return {}
    
    def _calculate_part_metrics(self, detections_history: List, part_name: str) -> Optional[Dict[str, Any]]:
        """Calculate metrics for a specific body part"""
        try:
            positions = []
            timestamps = []
            
            # Extract positions and timestamps
            for detection in detections_history:
                part = getattr(detection, part_name)
                if part is not None:
                    positions.append([part.x, part.y])
                    timestamps.append(detection.timestamp)
            
            if len(positions) < 2:
                return None
            
            positions = np.array(positions)
            timestamps = np.array(timestamps)
            
            # Calculate velocities
            velocities = []
            distances = []
            
            for i in range(1, len(positions)):
                dt = timestamps[i] - timestamps[i-1]
                if dt > 0:
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    velocity = distance / dt
                    
                    velocities.append(velocity)
                    distances.append(distance)
            
            if not velocities:
                return None
            
            velocities = np.array(velocities)
            distances = np.array(distances)
            
            # Smooth velocities
            if len(velocities) >= self.smoothing_window:
                smoothed_velocities = savgol_filter(velocities, self.smoothing_window, 2)
            else:
                smoothed_velocities = velocities
            
            # Calculate accelerations
            accelerations = []
            for i in range(1, len(smoothed_velocities)):
                dt = timestamps[i+1] - timestamps[i]
                if dt > 0:
                    acceleration = (smoothed_velocities[i] - smoothed_velocities[i-1]) / dt
                    accelerations.append(acceleration)
            
            # Calculate movement directions
            directions = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                direction = math.atan2(dy, dx)
                directions.append(direction)
            
            return {
                "positions": positions.tolist(),
                "timestamps": timestamps.tolist(),
                "velocities": velocities.tolist(),
                "smoothed_velocities": smoothed_velocities.tolist(),
                "accelerations": accelerations,
                "directions": directions,
                "total_distance": np.sum(distances),
                "avg_velocity": np.mean(velocities),
                "max_velocity": np.max(velocities),
                "min_velocity": np.min(velocities),
                "velocity_std": np.std(velocities),
                "movement_active": np.sum(velocities > self.velocity_threshold) / len(velocities)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {part_name}: {e}")
            return None
    
    def _detect_interactions(self, detections_history: List) -> List[InteractionEvent]:
        """Detect interactions between body parts"""
        interactions = []
        
        try:
            # Define interaction pairs of interest
            interaction_pairs = [
                ('hand_left', 'genitals'),
                ('hand_right', 'genitals'),
                ('mouth', 'genitals'),
                ('hand_left', 'breasts'),
                ('hand_right', 'breasts'),
                ('mouth', 'breasts'),
                ('genitals', 'pelvis'),
                ('hand_left', 'hand_right'),
                ('mouth', 'hand_left'),
                ('mouth', 'hand_right')
            ]
            
            for primary_part, secondary_part in interaction_pairs:
                part_interactions = self._detect_part_interactions(
                    detections_history, primary_part, secondary_part
                )
                interactions.extend(part_interactions)
            
            # Sort interactions by timestamp
            interactions.sort(key=lambda x: x.timestamp)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error detecting interactions: {e}")
            return []
    
    def _detect_part_interactions(self, detections_history: List, part1: str, part2: str) -> List[InteractionEvent]:
        """Detect interactions between two specific body parts"""
        interactions = []
        
        try:
            current_interaction = None
            
            for detection in detections_history:
                point1 = getattr(detection, part1)
                point2 = getattr(detection, part2)
                
                if point1 is None or point2 is None:
                    if current_interaction is not None:
                        # End current interaction
                        current_interaction.duration = detection.timestamp - current_interaction.timestamp
                        interactions.append(current_interaction)
                        current_interaction = None
                    continue
                
                # Calculate distance between parts
                distance = math.sqrt(
                    (point1.x - point2.x)**2 + (point1.y - point2.y)**2
                )
                
                # Determine interaction type and intensity
                if distance < self.contact_threshold:
                    interaction_type = 'contact'
                    intensity = 1.0 - (distance / self.contact_threshold)
                elif distance < self.interaction_threshold:
                    interaction_type = 'proximity'
                    intensity = 1.0 - (distance / self.interaction_threshold)
                else:
                    interaction_type = None
                    intensity = 0.0
                
                if interaction_type and intensity > 0.3:
                    if current_interaction is None:
                        # Start new interaction
                        current_interaction = InteractionEvent(
                            timestamp=detection.timestamp,
                            frame_id=detection.frame_id,
                            primary_part=part1,
                            secondary_part=part2,
                            interaction_type=interaction_type,
                            intensity=intensity,
                            distance=distance
                        )
                    else:
                        # Update existing interaction
                        current_interaction.intensity = max(current_interaction.intensity, intensity)
                        current_interaction.distance = min(current_interaction.distance, distance)
                else:
                    if current_interaction is not None:
                        # End current interaction
                        current_interaction.duration = detection.timestamp - current_interaction.timestamp
                        if current_interaction.duration >= self.min_interaction_duration:
                            interactions.append(current_interaction)
                        current_interaction = None
            
            # Handle any ongoing interaction at the end
            if current_interaction is not None:
                current_interaction.duration = detections_history[-1].timestamp - current_interaction.timestamp
                if current_interaction.duration >= self.min_interaction_duration:
                    interactions.append(current_interaction)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error detecting interactions between {part1} and {part2}: {e}")
            return []
    
    def _analyze_genital_interactions(self, detections_history: List, interactions: List[InteractionEvent]) -> Dict[str, Any]:
        """Analyze interactions specifically involving genitals"""
        try:
            genital_interactions = [
                interaction for interaction in interactions 
                if 'genitals' in [interaction.primary_part, interaction.secondary_part]
            ]
            
            if not genital_interactions:
                return {"error": "No genital interactions detected"}
            
            # Analyze interaction patterns
            interaction_analysis = {
                "total_interactions": len(genital_interactions),
                "interaction_types": {},
                "intensity_patterns": [],
                "duration_patterns": [],
                "speed_patterns": []
            }
            
            # Group by interaction type
            for interaction in genital_interactions:
                interaction_key = f"{interaction.primary_part}-{interaction.secondary_part}"
                if interaction_key not in interaction_analysis["interaction_types"]:
                    interaction_analysis["interaction_types"][interaction_key] = []
                interaction_analysis["interaction_types"][interaction_key].append(interaction)
            
            # Calculate patterns for each interaction type
            for interaction_type, type_interactions in interaction_analysis["interaction_types"].items():
                # Calculate intensity patterns
                intensities = [i.intensity for i in type_interactions]
                interaction_analysis["intensity_patterns"].append({
                    "type": interaction_type,
                    "avg_intensity": np.mean(intensities),
                    "max_intensity": np.max(intensities),
                    "intensity_variation": np.std(intensities)
                })
                
                # Calculate duration patterns
                durations = [i.duration for i in type_interactions]
                interaction_analysis["duration_patterns"].append({
                    "type": interaction_type,
                    "avg_duration": np.mean(durations),
                    "max_duration": np.max(durations),
                    "total_duration": np.sum(durations)
                })
            
            # Analyze movement speed during interactions
            speed_analysis = self._analyze_interaction_speeds(detections_history, genital_interactions)
            interaction_analysis["speed_patterns"] = speed_analysis
            
            return interaction_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing genital interactions: {e}")
            return {"error": f"Analysis error: {str(e)}"}
    
    def _analyze_interaction_speeds(self, detections_history: List, interactions: List[InteractionEvent]) -> List[Dict[str, Any]]:
        """Analyze movement speeds during interactions"""
        speed_patterns = []
        
        try:
            for interaction in interactions:
                # Get frames during interaction
                start_frame = interaction.frame_id
                end_frame = start_frame + int(interaction.duration * 30)  # Assuming 30fps
                
                # Extract movement data during interaction
                interaction_frames = [
                    detection for detection in detections_history
                    if start_frame <= detection.frame_id <= end_frame
                ]
                
                if len(interaction_frames) < 2:
                    continue
                
                # Calculate speeds for primary part
                primary_speeds = []
                for i in range(1, len(interaction_frames)):
                    curr_frame = interaction_frames[i]
                    prev_frame = interaction_frames[i-1]
                    
                    curr_point = getattr(curr_frame, interaction.primary_part)
                    prev_point = getattr(prev_frame, interaction.primary_part)
                    
                    if curr_point and prev_point:
                        dx = curr_point.x - prev_point.x
                        dy = curr_point.y - prev_point.y
                        dt = curr_frame.timestamp - prev_frame.timestamp
                        
                        if dt > 0:
                            speed = math.sqrt(dx*dx + dy*dy) / dt
                            primary_speeds.append(speed)
                
                if primary_speeds:
                    speed_patterns.append({
                        "interaction_type": f"{interaction.primary_part}-{interaction.secondary_part}",
                        "timestamp": interaction.timestamp,
                        "duration": interaction.duration,
                        "avg_speed": np.mean(primary_speeds),
                        "max_speed": np.max(primary_speeds),
                        "speed_variation": np.std(primary_speeds),
                        "speeds": primary_speeds
                    })
            
            return speed_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing interaction speeds: {e}")
            return []
    
    def _generate_movement_patterns(self, genital_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate movement patterns for funscript generation"""
        try:
            if "error" in genital_analysis:
                return {"error": genital_analysis["error"]}
            
            patterns = {
                "stroke_patterns": [],
                "intensity_curve": [],
                "rhythm_analysis": {},
                "peak_events": []
            }
            
            # Generate stroke patterns from speed analysis
            if "speed_patterns" in genital_analysis:
                for speed_pattern in genital_analysis["speed_patterns"]:
                    if "hand" in speed_pattern["interaction_type"]:
                        stroke_pattern = self._generate_stroke_pattern(speed_pattern)
                        patterns["stroke_patterns"].append(stroke_pattern)
            
            # Generate intensity curve
            if "intensity_patterns" in genital_analysis:
                intensity_curve = self._generate_intensity_curve(genital_analysis["intensity_patterns"])
                patterns["intensity_curve"] = intensity_curve
            
            # Analyze rhythm
            rhythm_analysis = self._analyze_rhythm(genital_analysis)
            patterns["rhythm_analysis"] = rhythm_analysis
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error generating movement patterns: {e}")
            return {"error": f"Pattern generation error: {str(e)}"}
    
    def _generate_stroke_pattern(self, speed_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Generate stroke pattern from speed analysis"""
        try:
            speeds = speed_pattern["speeds"]
            
            # Normalize speeds to 0-100 range for funscript
            if speeds:
                min_speed = min(speeds)
                max_speed = max(speeds)
                
                if max_speed > min_speed:
                    normalized_speeds = [
                        int(((speed - min_speed) / (max_speed - min_speed)) * 100)
                        for speed in speeds
                    ]
                else:
                    normalized_speeds = [50] * len(speeds)  # Default to middle value
                
                return {
                    "timestamp": speed_pattern["timestamp"],
                    "duration": speed_pattern["duration"],
                    "positions": normalized_speeds,
                    "avg_intensity": speed_pattern["avg_speed"],
                    "max_intensity": speed_pattern["max_speed"],
                    "type": speed_pattern["interaction_type"]
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error generating stroke pattern: {e}")
            return {}
    
    def _generate_intensity_curve(self, intensity_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate overall intensity curve"""
        try:
            curve = []
            
            for pattern in intensity_patterns:
                curve.append({
                    "type": pattern["type"],
                    "avg_intensity": pattern["avg_intensity"],
                    "max_intensity": pattern["max_intensity"],
                    "variation": pattern["intensity_variation"]
                })
            
            return curve
            
        except Exception as e:
            logger.error(f"Error generating intensity curve: {e}")
            return []
    
    def _analyze_rhythm(self, genital_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze rhythm patterns"""
        try:
            rhythm = {
                "average_duration": 0.0,
                "rhythm_consistency": 0.0,
                "peak_frequency": 0.0,
                "pattern_type": "irregular"
            }
            
            if "duration_patterns" in genital_analysis:
                durations = [p["avg_duration"] for p in genital_analysis["duration_patterns"]]
                if durations:
                    rhythm["average_duration"] = np.mean(durations)
                    rhythm["rhythm_consistency"] = 1.0 - (np.std(durations) / np.mean(durations))
                    
                    # Determine pattern type
                    if rhythm["rhythm_consistency"] > 0.7:
                        rhythm["pattern_type"] = "regular"
                    elif rhythm["rhythm_consistency"] > 0.4:
                        rhythm["pattern_type"] = "semi-regular"
                    else:
                        rhythm["pattern_type"] = "irregular"
            
            return rhythm
            
        except Exception as e:
            logger.error(f"Error analyzing rhythm: {e}")
            return {}