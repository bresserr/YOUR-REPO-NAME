"""
Funscript Generator for VR Assistive Technology
Converts tracking data and interactions into funscript format for the Handy device
"""
import asyncio
import json
import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import interpolate, signal

logger = logging.getLogger(__name__)

@dataclass
class FunscriptAction:
    """Individual funscript action with position and timestamp"""
    at: int  # Timestamp in milliseconds
    pos: int  # Position 0-100

@dataclass
class FunscriptMetadata:
    """Metadata for funscript file"""
    creator: str = "VR Assistive Technology"
    description: str = "Generated from VR video analysis"
    duration: float = 0.0
    fps: float = 30.0
    notes: str = ""
    script_url: str = ""
    tags: List[str] = None
    type: str = "basic"
    version: str = "1.0"

class FunscriptGenerator:
    """
    Generates funscript files from tracking data and interactions
    Optimized for assistive technology applications
    """
    
    def __init__(self):
        self.min_position = 0
        self.max_position = 100
        self.min_interval = 100  # Minimum time between actions in ms
        self.max_speed = 500  # Maximum speed in units/second
        self.smoothing_window = 5  # Frames for smoothing
        
        # Interaction to funscript mapping parameters
        self.speed_multiplier = 1.0
        self.depth_multiplier = 1.0
        self.intensity_multiplier = 1.0
        
        # Pattern generation parameters
        self.stroke_length_base = 50  # Base stroke length
        self.stroke_frequency_base = 1.0  # Base frequency in Hz
        
        logger.info("FunscriptGenerator initialized")
    
    async def generate_from_tracking(self, tracking_data: Dict) -> Dict:
        """Generate funscript from tracking data"""
        try:
            interactions = tracking_data.get('interactions', [])
            fps = tracking_data.get('fps', 30.0)
            
            if not interactions:
                logger.warning("No interactions found in tracking data")
                return self._create_empty_funscript()
            
            # Convert interactions to funscript actions
            actions = await self._convert_interactions_to_actions(interactions, fps)
            
            # Apply smoothing and optimization
            actions = self._smooth_actions(actions)
            actions = self._optimize_actions(actions)
            
            # Create funscript metadata
            metadata = self._create_metadata(tracking_data)
            
            # Create final funscript
            funscript = {
                "version": "1.0",
                "inverted": False,
                "range": 100,
                "actions": [{"at": action.at, "pos": action.pos} for action in actions],
                "metadata": metadata.__dict__
            }
            
            logger.info(f"Generated funscript with {len(actions)} actions")
            return funscript
            
        except Exception as e:
            logger.error(f"Error generating funscript: {str(e)}")
            return self._create_empty_funscript()
    
    async def _convert_interactions_to_actions(self, interactions: List[Dict], fps: float) -> List[FunscriptAction]:
        """Convert interaction events to funscript actions"""
        actions = []
        
        # Sort interactions by timestamp
        sorted_interactions = sorted(interactions, key=lambda x: x['start_time'])
        
        for interaction in sorted_interactions:
            # Skip non-relevant interactions
            if interaction['primary_part'] != 'genitals':
                continue
            
            # Calculate action parameters
            timestamp_ms = int(interaction['start_time'] * 1000)
            speed = interaction['speed']
            depth = interaction['depth']
            intensity = interaction['intensity']
            
            # Generate position based on interaction type and metrics
            position = self._calculate_position_from_interaction(
                interaction['secondary_part'],
                speed,
                depth,
                intensity
            )
            
            # Create action
            action = FunscriptAction(at=timestamp_ms, pos=position)
            actions.append(action)
            
            # Generate additional actions for continuous movements
            if speed > 0.1:  # Only for significant movement
                additional_actions = self._generate_movement_sequence(
                    interaction, fps
                )
                actions.extend(additional_actions)
        
        return actions
    
    def _calculate_position_from_interaction(self, secondary_part: str, speed: float, 
                                           depth: float, intensity: float) -> int:
        """Calculate funscript position based on interaction parameters"""
        # Base position based on body part
        base_positions = {
            'hand_1': 50,
            'hand_2': 50,
            'mouth': 75,
            'breasts': 25,
            'pelvis': 60
        }
        
        base_pos = base_positions.get(secondary_part, 50)
        
        # Adjust based on speed (faster = more extreme positions)
        speed_adjustment = (speed * self.speed_multiplier) * 30
        
        # Adjust based on depth (deeper = lower position)
        depth_adjustment = (depth * self.depth_multiplier) * 40
        
        # Adjust based on intensity
        intensity_adjustment = (intensity * self.intensity_multiplier) * 20
        
        # Combine adjustments
        final_position = base_pos + speed_adjustment - depth_adjustment + intensity_adjustment
        
        # Clamp to valid range
        final_position = max(self.min_position, min(self.max_position, int(final_position)))
        
        return final_position
    
    def _generate_movement_sequence(self, interaction: Dict, fps: float) -> List[FunscriptAction]:
        """Generate a sequence of actions for continuous movement"""
        actions = []
        
        speed = interaction['speed']
        depth = interaction['depth']
        duration = interaction['end_time'] - interaction['start_time']
        
        if duration < 0.1:  # Too short to generate sequence
            return actions
        
        # Calculate stroke parameters
        stroke_frequency = self.stroke_frequency_base * (1 + speed * 0.5)
        stroke_length = self.stroke_length_base * (1 + depth * 0.3)
        
        # Generate oscillating pattern
        num_strokes = int(duration * stroke_frequency)
        
        for i in range(num_strokes):
            # Calculate stroke timing
            stroke_progress = i / max(num_strokes - 1, 1)
            timestamp = interaction['start_time'] + duration * stroke_progress
            timestamp_ms = int(timestamp * 1000)
            
            # Generate oscillating position
            phase = stroke_progress * 2 * math.pi * stroke_frequency
            oscillation = math.sin(phase) * stroke_length / 2
            
            base_position = self._calculate_position_from_interaction(
                interaction['secondary_part'],
                speed,
                depth,
                interaction['intensity']
            )
            
            position = base_position + oscillation
            position = max(self.min_position, min(self.max_position, int(position)))
            
            action = FunscriptAction(at=timestamp_ms, pos=position)
            actions.append(action)
        
        return actions
    
    def _smooth_actions(self, actions: List[FunscriptAction]) -> List[FunscriptAction]:
        """Apply smoothing to actions to reduce jitter"""
        if len(actions) < 3:
            return actions
        
        # Extract positions and timestamps
        positions = [action.pos for action in actions]
        timestamps = [action.at for action in actions]
        
        # Apply moving average smoothing
        smoothed_positions = []
        for i in range(len(positions)):
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(positions), i + self.smoothing_window // 2 + 1)
            
            avg_position = sum(positions[start_idx:end_idx]) / (end_idx - start_idx)
            smoothed_positions.append(int(avg_position))
        
        # Create smoothed actions
        smoothed_actions = []
        for i, (timestamp, position) in enumerate(zip(timestamps, smoothed_positions)):
            action = FunscriptAction(at=timestamp, pos=position)
            smoothed_actions.append(action)
        
        return smoothed_actions
    
    def _optimize_actions(self, actions: List[FunscriptAction]) -> List[FunscriptAction]:
        """Optimize actions by removing redundant points and enforcing constraints"""
        if len(actions) < 2:
            return actions
        
        optimized_actions = []
        
        for i, action in enumerate(actions):
            # Skip if too close to previous action
            if optimized_actions and action.at - optimized_actions[-1].at < self.min_interval:
                continue
            
            # Check speed constraint
            if optimized_actions:
                prev_action = optimized_actions[-1]
                time_diff = (action.at - prev_action.at) / 1000.0  # Convert to seconds
                pos_diff = abs(action.pos - prev_action.pos)
                
                if time_diff > 0:
                    speed = pos_diff / time_diff
                    if speed > self.max_speed:
                        # Adjust position to respect speed limit
                        max_pos_change = self.max_speed * time_diff
                        if action.pos > prev_action.pos:
                            action.pos = min(action.pos, prev_action.pos + int(max_pos_change))
                        else:
                            action.pos = max(action.pos, prev_action.pos - int(max_pos_change))
            
            optimized_actions.append(action)
        
        return optimized_actions
    
    def _create_metadata(self, tracking_data: Dict) -> FunscriptMetadata:
        """Create metadata for funscript"""
        interactions = tracking_data.get('interactions', [])
        fps = tracking_data.get('fps', 30.0)
        
        duration = 0.0
        if interactions:
            duration = max(interaction['end_time'] for interaction in interactions)
        
        metadata = FunscriptMetadata(
            creator="VR Assistive Technology",
            description=f"Generated from VR video analysis with {len(interactions)} interactions",
            duration=duration,
            fps=fps,
            notes="Automatically generated for assistive technology use",
            type="basic",
            version="1.0"
        )
        
        return metadata
    
    def _create_empty_funscript(self) -> Dict:
        """Create empty funscript when no data is available"""
        return {
            "version": "1.0",
            "inverted": False,
            "range": 100,
            "actions": [],
            "metadata": {
                "creator": "VR Assistive Technology",
                "description": "Empty funscript - no interactions detected",
                "duration": 0.0,
                "fps": 30.0,
                "notes": "No interactions found in tracking data",
                "type": "basic",
                "version": "1.0"
            }
        }
    
    def validate_funscript(self, funscript: Dict) -> Tuple[bool, List[str]]:
        """Validate funscript format and content"""
        errors = []
        
        # Check required fields
        required_fields = ['version', 'inverted', 'range', 'actions']
        for field in required_fields:
            if field not in funscript:
                errors.append(f"Missing required field: {field}")
        
        # Validate actions
        if 'actions' in funscript:
            actions = funscript['actions']
            
            if not isinstance(actions, list):
                errors.append("Actions must be a list")
            else:
                for i, action in enumerate(actions):
                    if not isinstance(action, dict):
                        errors.append(f"Action {i} must be a dictionary")
                        continue
                    
                    if 'at' not in action or 'pos' not in action:
                        errors.append(f"Action {i} missing 'at' or 'pos' field")
                    
                    if not isinstance(action.get('at'), int):
                        errors.append(f"Action {i} 'at' must be an integer")
                    
                    if not isinstance(action.get('pos'), int):
                        errors.append(f"Action {i} 'pos' must be an integer")
                    
                    if not (0 <= action.get('pos', -1) <= 100):
                        errors.append(f"Action {i} 'pos' must be between 0 and 100")
        
        # Check temporal ordering
        if 'actions' in funscript and len(funscript['actions']) > 1:
            timestamps = [action['at'] for action in funscript['actions']]
            if timestamps != sorted(timestamps):
                errors.append("Actions must be in chronological order")
        
        return len(errors) == 0, errors
    
    def export_funscript(self, funscript: Dict, output_path: str) -> bool:
        """Export funscript to file"""
        try:
            # Validate before export
            is_valid, errors = self.validate_funscript(funscript)
            if not is_valid:
                logger.error(f"Invalid funscript: {errors}")
                return False
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(funscript, f, indent=2)
            
            logger.info(f"Funscript exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting funscript: {str(e)}")
            return False
    
    def import_funscript(self, input_path: str) -> Optional[Dict]:
        """Import funscript from file"""
        try:
            with open(input_path, 'r') as f:
                funscript = json.load(f)
            
            # Validate imported funscript
            is_valid, errors = self.validate_funscript(funscript)
            if not is_valid:
                logger.error(f"Invalid imported funscript: {errors}")
                return None
            
            logger.info(f"Funscript imported from {input_path}")
            return funscript
            
        except Exception as e:
            logger.error(f"Error importing funscript: {str(e)}")
            return None
    
    def set_generation_parameters(self, **kwargs):
        """Set funscript generation parameters"""
        if 'speed_multiplier' in kwargs:
            self.speed_multiplier = kwargs['speed_multiplier']
        if 'depth_multiplier' in kwargs:
            self.depth_multiplier = kwargs['depth_multiplier']
        if 'intensity_multiplier' in kwargs:
            self.intensity_multiplier = kwargs['intensity_multiplier']
        if 'stroke_length_base' in kwargs:
            self.stroke_length_base = kwargs['stroke_length_base']
        if 'stroke_frequency_base' in kwargs:
            self.stroke_frequency_base = kwargs['stroke_frequency_base']
        
        logger.info(f"Generation parameters updated: {kwargs}")
    
    def get_generation_parameters(self) -> Dict:
        """Get current generation parameters"""
        return {
            'speed_multiplier': self.speed_multiplier,
            'depth_multiplier': self.depth_multiplier,
            'intensity_multiplier': self.intensity_multiplier,
            'stroke_length_base': self.stroke_length_base,
            'stroke_frequency_base': self.stroke_frequency_base,
            'min_position': self.min_position,
            'max_position': self.max_position,
            'min_interval': self.min_interval,
            'max_speed': self.max_speed
        }
    
    def analyze_funscript(self, funscript: Dict) -> Dict:
        """Analyze funscript and return statistics"""
        try:
            actions = funscript.get('actions', [])
            
            if not actions:
                return {'total_actions': 0, 'duration': 0, 'avg_speed': 0}
            
            # Basic statistics
            total_actions = len(actions)
            duration = (actions[-1]['at'] - actions[0]['at']) / 1000.0 if total_actions > 1 else 0
            
            # Calculate speeds
            speeds = []
            for i in range(1, len(actions)):
                time_diff = (actions[i]['at'] - actions[i-1]['at']) / 1000.0
                pos_diff = abs(actions[i]['pos'] - actions[i-1]['pos'])
                if time_diff > 0:
                    speeds.append(pos_diff / time_diff)
            
            avg_speed = sum(speeds) / len(speeds) if speeds else 0
            max_speed = max(speeds) if speeds else 0
            
            # Position statistics
            positions = [action['pos'] for action in actions]
            avg_position = sum(positions) / len(positions)
            min_position = min(positions)
            max_position = max(positions)
            
            return {
                'total_actions': total_actions,
                'duration': duration,
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'avg_position': avg_position,
                'min_position': min_position,
                'max_position': max_position,
                'position_range': max_position - min_position
            }
            
        except Exception as e:
            logger.error(f"Error analyzing funscript: {str(e)}")
            return {}