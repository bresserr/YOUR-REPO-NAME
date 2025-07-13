"""
Tracking Manager for VR Assistive Technology
Handles tracking data, manual corrections, and interaction analysis
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import math

logger = logging.getLogger(__name__)

@dataclass
class TrackingPoint:
    """Individual tracking point with position and metadata"""
    x: float
    y: float
    confidence: float
    timestamp: float
    is_corrected: bool = False
    correction_id: Optional[str] = None

@dataclass
class BodyPartTrack:
    """Tracking data for a specific body part"""
    name: str
    points: List[TrackingPoint]
    is_active: bool = True
    last_seen: float = 0.0
    color: Tuple[int, int, int] = (255, 255, 255)

@dataclass
class InteractionEvent:
    """Represents an interaction between body parts"""
    primary_part: str
    secondary_part: str
    interaction_type: str  # 'contact', 'proximity', 'movement'
    start_time: float
    end_time: float
    intensity: float
    speed: float
    depth: float
    metadata: Dict[str, Any]

class TrackingManager:
    """
    Manages tracking data, corrections, and interaction analysis
    for assistive technology applications
    """
    
    def __init__(self):
        self.tracking_data = {}  # frame_number -> List[BodyPartTrack]
        self.body_part_tracks = {}  # body_part_name -> List[TrackingPoint]
        self.interactions = []  # List[InteractionEvent]
        self.corrections = {}  # correction_id -> correction_data
        self.current_frame = 0
        self.fps = 30.0
        
        # Body part colors for visualization
        self.body_part_colors = {
            'head': (255, 0, 0),      # Red
            'mouth': (255, 128, 0),   # Orange
            'hand_1': (0, 255, 0),    # Green
            'hand_2': (0, 255, 128),  # Light Green
            'breasts': (255, 0, 255), # Magenta
            'pelvis': (255, 255, 0),  # Yellow
            'genitals': (128, 0, 255) # Purple
        }
        
        # Interaction parameters
        self.interaction_distance_threshold = 50.0  # pixels
        self.speed_smoothing_window = 5
        self.depth_analysis_window = 10
        
        logger.info("TrackingManager initialized")
    
    def update_tracking(self, frame_number: int, detections: List[Dict]):
        """Update tracking data with new detections"""
        self.current_frame = frame_number
        timestamp = frame_number / self.fps
        
        # Initialize frame data if not exists
        if frame_number not in self.tracking_data:
            self.tracking_data[frame_number] = []
        
        # Process detections
        for detection in detections:
            body_part_name = detection['name']
            points = detection['points']
            
            # Create tracking points
            tracking_points = []
            for point in points:
                tracking_point = TrackingPoint(
                    x=point['x'],
                    y=point['y'],
                    confidence=point['confidence'],
                    timestamp=timestamp,
                    is_corrected=False
                )
                tracking_points.append(tracking_point)
            
            # Create body part track
            body_part_track = BodyPartTrack(
                name=body_part_name,
                points=tracking_points,
                is_active=True,
                last_seen=timestamp,
                color=self.body_part_colors.get(body_part_name, (255, 255, 255))
            )
            
            # Add to frame tracking data
            self.tracking_data[frame_number].append(body_part_track)
            
            # Update body part tracks history
            if body_part_name not in self.body_part_tracks:
                self.body_part_tracks[body_part_name] = []
            
            self.body_part_tracks[body_part_name].extend(tracking_points)
        
        # Analyze interactions for current frame
        self._analyze_interactions(frame_number)
    
    def correct_tracking(self, frame_number: int, body_part: str, new_position: Dict):
        """Manually correct tracking point"""
        try:
            if frame_number not in self.tracking_data:
                logger.error(f"No tracking data for frame {frame_number}")
                return False
            
            # Find body part in frame
            for body_part_track in self.tracking_data[frame_number]:
                if body_part_track.name == body_part:
                    # Update first point (assuming single point per body part)
                    if body_part_track.points:
                        correction_id = f"{frame_number}_{body_part}_{time.time()}"
                        
                        # Store original position for undo
                        original_point = body_part_track.points[0]
                        self.corrections[correction_id] = {
                            'frame_number': frame_number,
                            'body_part': body_part,
                            'original_position': asdict(original_point),
                            'new_position': new_position,
                            'timestamp': time.time()
                        }
                        
                        # Apply correction
                        body_part_track.points[0].x = new_position['x']
                        body_part_track.points[0].y = new_position['y']
                        body_part_track.points[0].is_corrected = True
                        body_part_track.points[0].correction_id = correction_id
                        
                        # Update body part tracks history
                        for point in self.body_part_tracks[body_part]:
                            if abs(point.timestamp - original_point.timestamp) < 0.01:
                                point.x = new_position['x']
                                point.y = new_position['y']
                                point.is_corrected = True
                                point.correction_id = correction_id
                                break
                        
                        logger.info(f"Tracking corrected: {body_part} at frame {frame_number}")
                        return True
            
            logger.error(f"Body part {body_part} not found in frame {frame_number}")
            return False
            
        except Exception as e:
            logger.error(f"Error correcting tracking: {str(e)}")
            return False
    
    def _analyze_interactions(self, frame_number: int):
        """Analyze interactions between body parts in current frame"""
        if frame_number not in self.tracking_data:
            return
        
        frame_data = self.tracking_data[frame_number]
        timestamp = frame_number / self.fps
        
        # Get positions of all body parts
        positions = {}
        for body_part_track in frame_data:
            if body_part_track.points:
                point = body_part_track.points[0]
                positions[body_part_track.name] = (point.x, point.y)
        
        # Analyze specific interactions relevant to the application
        if 'genitals' in positions:
            genital_pos = positions['genitals']
            
            # Check interactions with hands
            for hand_name in ['hand_1', 'hand_2']:
                if hand_name in positions:
                    hand_pos = positions[hand_name]
                    distance = self._calculate_distance(genital_pos, hand_pos)
                    
                    if distance < self.interaction_distance_threshold:
                        # Calculate interaction metrics
                        speed = self._calculate_speed(hand_name, frame_number)
                        depth = self._calculate_depth_metric(hand_name, frame_number)
                        
                        # Create interaction event
                        interaction = InteractionEvent(
                            primary_part='genitals',
                            secondary_part=hand_name,
                            interaction_type='contact',
                            start_time=timestamp,
                            end_time=timestamp,
                            intensity=1.0 - (distance / self.interaction_distance_threshold),
                            speed=speed,
                            depth=depth,
                            metadata={
                                'distance': distance,
                                'frame_number': frame_number,
                                'positions': {'genitals': genital_pos, hand_name: hand_pos}
                            }
                        )
                        
                        # Add to interactions
                        self.interactions.append(interaction)
            
            # Check interactions with mouth
            if 'mouth' in positions:
                mouth_pos = positions['mouth']
                distance = self._calculate_distance(genital_pos, mouth_pos)
                
                if distance < self.interaction_distance_threshold:
                    speed = self._calculate_speed('mouth', frame_number)
                    depth = self._calculate_depth_metric('mouth', frame_number)
                    
                    interaction = InteractionEvent(
                        primary_part='genitals',
                        secondary_part='mouth',
                        interaction_type='contact',
                        start_time=timestamp,
                        end_time=timestamp,
                        intensity=1.0 - (distance / self.interaction_distance_threshold),
                        speed=speed,
                        depth=depth,
                        metadata={
                            'distance': distance,
                            'frame_number': frame_number,
                            'positions': {'genitals': genital_pos, 'mouth': mouth_pos}
                        }
                    )
                    
                    self.interactions.append(interaction)
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_speed(self, body_part: str, current_frame: int) -> float:
        """Calculate movement speed for a body part"""
        if body_part not in self.body_part_tracks:
            return 0.0
        
        points = self.body_part_tracks[body_part]
        if len(points) < 2:
            return 0.0
        
        # Get recent points for speed calculation
        recent_points = []
        current_time = current_frame / self.fps
        
        for point in reversed(points):
            if current_time - point.timestamp <= self.speed_smoothing_window / self.fps:
                recent_points.append(point)
            else:
                break
        
        if len(recent_points) < 2:
            return 0.0
        
        # Calculate average speed
        total_distance = 0.0
        total_time = 0.0
        
        for i in range(len(recent_points) - 1):
            p1 = recent_points[i]
            p2 = recent_points[i + 1]
            
            distance = self._calculate_distance((p1.x, p1.y), (p2.x, p2.y))
            time_diff = abs(p1.timestamp - p2.timestamp)
            
            if time_diff > 0:
                total_distance += distance
                total_time += time_diff
        
        return total_distance / total_time if total_time > 0 else 0.0
    
    def _calculate_depth_metric(self, body_part: str, current_frame: int) -> float:
        """Calculate depth/penetration metric based on movement patterns"""
        if body_part not in self.body_part_tracks:
            return 0.0
        
        points = self.body_part_tracks[body_part]
        if len(points) < self.depth_analysis_window:
            return 0.0
        
        # Get recent points for depth analysis
        recent_points = []
        current_time = current_frame / self.fps
        
        for point in reversed(points):
            if len(recent_points) >= self.depth_analysis_window:
                break
            if current_time - point.timestamp <= self.depth_analysis_window / self.fps:
                recent_points.append(point)
        
        if len(recent_points) < self.depth_analysis_window:
            return 0.0
        
        # Analyze movement patterns for depth estimation
        # This is a simplified implementation
        movement_variance = 0.0
        center_x = sum(p.x for p in recent_points) / len(recent_points)
        center_y = sum(p.y for p in recent_points) / len(recent_points)
        
        for point in recent_points:
            variance = (point.x - center_x)**2 + (point.y - center_y)**2
            movement_variance += variance
        
        movement_variance /= len(recent_points)
        
        # Normalize to 0-1 range
        max_variance = 1000.0  # Adjust based on expected movement
        depth = min(movement_variance / max_variance, 1.0)
        
        return depth
    
    def get_current_interactions(self) -> Dict[str, float]:
        """Get current interaction metrics"""
        if not self.interactions:
            return {}
        
        # Get recent interactions
        current_time = self.current_frame / self.fps
        recent_interactions = [
            interaction for interaction in self.interactions
            if current_time - interaction.start_time <= 1.0  # Last 1 second
        ]
        
        if not recent_interactions:
            return {}
        
        # Calculate average metrics
        avg_speed = sum(interaction.speed for interaction in recent_interactions) / len(recent_interactions)
        avg_depth = sum(interaction.depth for interaction in recent_interactions) / len(recent_interactions)
        avg_intensity = sum(interaction.intensity for interaction in recent_interactions) / len(recent_interactions)
        
        return {
            'speed': avg_speed,
            'depth': avg_depth,
            'intensity': avg_intensity,
            'interaction_count': len(recent_interactions)
        }
    
    def get_tracking_data(self) -> Dict:
        """Get all tracking data for funscript generation"""
        return {
            'tracking_data': self.tracking_data,
            'body_part_tracks': self.body_part_tracks,
            'interactions': [asdict(interaction) for interaction in self.interactions],
            'corrections': self.corrections,
            'fps': self.fps
        }
    
    def has_tracking_data(self) -> bool:
        """Check if tracking data is available"""
        return bool(self.tracking_data)
    
    def get_frame_data(self, frame_number: int) -> Optional[List[Dict]]:
        """Get tracking data for specific frame"""
        if frame_number not in self.tracking_data:
            return None
        
        return [asdict(track) for track in self.tracking_data[frame_number]]
    
    def export_tracking_data(self, output_path: str):
        """Export tracking data to JSON file"""
        try:
            export_data = {
                'tracking_data': {
                    str(frame): [asdict(track) for track in tracks]
                    for frame, tracks in self.tracking_data.items()
                },
                'interactions': [asdict(interaction) for interaction in self.interactions],
                'corrections': self.corrections,
                'fps': self.fps,
                'export_timestamp': time.time()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Tracking data exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting tracking data: {str(e)}")
    
    def import_tracking_data(self, input_path: str):
        """Import tracking data from JSON file"""
        try:
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            # Convert back to proper data structures
            self.tracking_data = {}
            for frame_str, tracks_data in import_data['tracking_data'].items():
                frame_num = int(frame_str)
                self.tracking_data[frame_num] = []
                
                for track_data in tracks_data:
                    points = [TrackingPoint(**point) for point in track_data['points']]
                    track = BodyPartTrack(
                        name=track_data['name'],
                        points=points,
                        is_active=track_data['is_active'],
                        last_seen=track_data['last_seen'],
                        color=tuple(track_data['color'])
                    )
                    self.tracking_data[frame_num].append(track)
            
            # Import interactions
            self.interactions = [InteractionEvent(**interaction) for interaction in import_data['interactions']]
            
            # Import corrections and other data
            self.corrections = import_data['corrections']
            self.fps = import_data['fps']
            
            logger.info(f"Tracking data imported from {input_path}")
            
        except Exception as e:
            logger.error(f"Error importing tracking data: {str(e)}")
    
    def clear_tracking_data(self):
        """Clear all tracking data"""
        self.tracking_data.clear()
        self.body_part_tracks.clear()
        self.interactions.clear()
        self.corrections.clear()
        self.current_frame = 0
        
        logger.info("Tracking data cleared")
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        stats = {
            'total_frames': len(self.tracking_data),
            'total_interactions': len(self.interactions),
            'total_corrections': len(self.corrections),
            'body_parts_tracked': list(self.body_part_tracks.keys()),
            'tracking_duration': max(self.tracking_data.keys()) / self.fps if self.tracking_data else 0
        }
        
        return stats