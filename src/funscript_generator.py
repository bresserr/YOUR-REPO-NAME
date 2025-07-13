"""
Funscript Generator for creating haptic device control scripts
based on body part interactions and movements
"""

import numpy as np
import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from scipy import signal
from scipy.interpolate import interp1d
import time
from loguru import logger

from src.utils.config import Config


@dataclass
class FunscriptAction:
    """Single funscript action point"""
    at: int  # timestamp in milliseconds
    pos: int  # position 0-100


@dataclass
class FunscriptMetadata:
    """Funscript metadata"""
    creator: str = "VR Video Analyzer"
    description: str = "Generated from VR video analysis"
    duration: Optional[int] = None
    fps: Optional[float] = None
    notes: str = ""
    tags: List[str] = None
    type: str = "basic"
    url: str = ""
    version: str = "1.0"


class FunscriptGenerator:
    """
    Generates funscript files from body part tracking data
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.funscript_config = config.funscript
        
        # Motion analysis parameters
        self.velocity_threshold = self.funscript_config.velocity_threshold
        self.acceleration_threshold = self.funscript_config.acceleration_threshold
        self.depth_sensitivity = self.funscript_config.depth_sensitivity
        self.speed_sensitivity = self.funscript_config.speed_sensitivity
        
        # Interaction detection parameters
        self.interaction_distance_threshold = self.funscript_config.interaction_distance_threshold
        self.interaction_duration_threshold = self.funscript_config.interaction_duration_threshold
        
        # Output parameters
        self.position_range = self.funscript_config.position_range
        self.speed_range = self.funscript_config.speed_range
        self.sampling_rate = self.funscript_config.sampling_rate
        
        # Smoothing parameters
        self.smoothing_enabled = self.funscript_config.smoothing_enabled
        self.smoothing_window = self.funscript_config.smoothing_window
        
        logger.info("Funscript Generator initialized")
    
    def generate_funscript(self, body_parts_data: Dict[int, Dict[str, Any]], 
                          target_part: str = "penis", 
                          interaction_parts: List[str] = None,
                          sensitivity: float = 1.0) -> Dict[str, Any]:
        """
        Generate funscript from body part tracking data
        """
        if interaction_parts is None:
            interaction_parts = ["hand_1", "hand_2", "mouth"]
        
        logger.info(f"Generating funscript for {target_part} with interactions: {interaction_parts}")
        
        try:
            # Extract time series data
            time_series = self._extract_time_series(body_parts_data)
            
            if not time_series:
                logger.warning("No time series data available")
                return self._create_empty_funscript()
            
            # Analyze target part motion
            target_motion = self._analyze_target_motion(time_series, target_part)
            
            # Analyze interactions
            interaction_data = self._analyze_interactions(time_series, target_part, interaction_parts)
            
            # Generate base motion curve
            base_motion = self._generate_base_motion(target_motion, sensitivity)
            
            # Apply interaction modulations
            modulated_motion = self._apply_interaction_modulations(base_motion, interaction_data, sensitivity)
            
            # Smooth the motion if enabled
            if self.smoothing_enabled:
                smoothed_motion = self._smooth_motion(modulated_motion)
            else:
                smoothed_motion = modulated_motion
            
            # Convert to funscript actions
            actions = self._convert_to_funscript_actions(smoothed_motion)
            
            # Generate metadata
            metadata = self._generate_metadata(time_series, target_part, interaction_parts)
            
            # Create final funscript
            funscript = {
                "version": "1.0",
                "inverted": False,
                "range": 100,
                "actions": actions,
                "metadata": asdict(metadata)
            }
            
            logger.info(f"Generated funscript with {len(actions)} actions")
            return funscript
            
        except Exception as e:
            logger.error(f"Error generating funscript: {e}")
            return self._create_empty_funscript()
    
    def _extract_time_series(self, body_parts_data: Dict[int, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract time series data from body parts tracking"""
        time_series = {}
        
        for frame_number, frame_data in sorted(body_parts_data.items()):
            for part_name, part_data in frame_data.items():
                if part_name not in time_series:
                    time_series[part_name] = []
                
                # Create time series entry
                entry = {
                    'frame': frame_number,
                    'timestamp': frame_number * (1000.0 / 30.0),  # Assume 30 FPS
                    'x': part_data.get('x', 0.0),
                    'y': part_data.get('y', 0.0),
                    'z': part_data.get('z', 0.0),
                    'confidence': part_data.get('confidence', 0.0),
                    'velocity': part_data.get('velocity', {'x': 0.0, 'y': 0.0}),
                    'tracking_quality': part_data.get('tracking_quality', 0.0)
                }
                
                time_series[part_name].append(entry)
        
        return time_series
    
    def _analyze_target_motion(self, time_series: Dict[str, List[Dict[str, Any]]], 
                              target_part: str) -> Dict[str, Any]:
        """Analyze motion of the target body part"""
        motion_analysis = {
            'part_name': target_part,
            'positions': [],
            'velocities': [],
            'accelerations': [],
            'timestamps': [],
            'motion_intensity': [],
            'direction_changes': [],
            'rhythmic_patterns': []
        }
        
        if target_part not in time_series:
            logger.warning(f"Target part {target_part} not found in time series")
            return motion_analysis
        
        data = time_series[target_part]
        
        # Extract basic motion data
        for entry in data:
            motion_analysis['positions'].append([entry['x'], entry['y'], entry['z']])
            motion_analysis['velocities'].append([entry['velocity']['x'], entry['velocity']['y']])
            motion_analysis['timestamps'].append(entry['timestamp'])
        
        positions = np.array(motion_analysis['positions'])
        velocities = np.array(motion_analysis['velocities'])
        timestamps = np.array(motion_analysis['timestamps'])
        
        if len(positions) < 2:
            return motion_analysis
        
        # Calculate accelerations
        accelerations = np.diff(velocities, axis=0)
        motion_analysis['accelerations'] = accelerations.tolist()
        
        # Calculate motion intensity (combined speed and acceleration)
        speeds = np.linalg.norm(velocities, axis=1)
        accel_magnitudes = np.linalg.norm(accelerations, axis=1) if len(accelerations) > 0 else np.zeros(len(speeds)-1)
        
        # Pad acceleration to match speed length
        if len(accel_magnitudes) < len(speeds):
            accel_magnitudes = np.pad(accel_magnitudes, (0, len(speeds) - len(accel_magnitudes)), mode='constant')
        
        intensity = speeds * self.speed_sensitivity + accel_magnitudes[:len(speeds)] * self.acceleration_threshold
        motion_analysis['motion_intensity'] = intensity.tolist()
        
        # Detect direction changes
        direction_changes = self._detect_direction_changes(velocities)
        motion_analysis['direction_changes'] = direction_changes
        
        # Analyze rhythmic patterns
        rhythmic_patterns = self._analyze_rhythmic_patterns(positions, timestamps)
        motion_analysis['rhythmic_patterns'] = rhythmic_patterns
        
        return motion_analysis
    
    def _analyze_interactions(self, time_series: Dict[str, List[Dict[str, Any]]], 
                             target_part: str, interaction_parts: List[str]) -> Dict[str, Any]:
        """Analyze interactions between target part and other body parts"""
        interaction_analysis = {
            'target_part': target_part,
            'interaction_parts': interaction_parts,
            'interactions': {},
            'combined_intensity': []
        }
        
        if target_part not in time_series:
            return interaction_analysis
        
        target_data = time_series[target_part]
        
        for interaction_part in interaction_parts:
            if interaction_part not in time_series:
                continue
            
            interaction_data = time_series[interaction_part]
            
            # Analyze interaction between target and this part
            interaction_info = self._analyze_pairwise_interaction(target_data, interaction_data, 
                                                                 target_part, interaction_part)
            
            interaction_analysis['interactions'][interaction_part] = interaction_info
        
        # Calculate combined interaction intensity
        combined_intensity = self._calculate_combined_interaction_intensity(interaction_analysis['interactions'])
        interaction_analysis['combined_intensity'] = combined_intensity
        
        return interaction_analysis
    
    def _analyze_pairwise_interaction(self, target_data: List[Dict[str, Any]], 
                                     interaction_data: List[Dict[str, Any]], 
                                     target_part: str, interaction_part: str) -> Dict[str, Any]:
        """Analyze interaction between two body parts"""
        interaction_info = {
            'target_part': target_part,
            'interaction_part': interaction_part,
            'distances': [],
            'contact_events': [],
            'interaction_intensity': [],
            'synchronized_motion': [],
            'timestamps': []
        }
        
        # Align data by timestamp
        aligned_data = []
        for target_entry in target_data:
            for interaction_entry in interaction_data:
                if abs(target_entry['timestamp'] - interaction_entry['timestamp']) < 50:  # 50ms tolerance
                    aligned_data.append((target_entry, interaction_entry))
                    break
        
        if len(aligned_data) < 2:
            return interaction_info
        
        # Calculate distances and interactions
        for target_entry, interaction_entry in aligned_data:
            target_pos = np.array([target_entry['x'], target_entry['y'], target_entry['z']])
            interaction_pos = np.array([interaction_entry['x'], interaction_entry['y'], interaction_entry['z']])
            
            distance = np.linalg.norm(target_pos - interaction_pos)
            interaction_info['distances'].append(distance)
            interaction_info['timestamps'].append(target_entry['timestamp'])
            
            # Detect contact events
            is_contact = distance < (self.interaction_distance_threshold * 1000)  # Convert to pixels
            interaction_info['contact_events'].append(is_contact)
            
            # Calculate interaction intensity
            target_vel = np.array([target_entry['velocity']['x'], target_entry['velocity']['y']])
            interaction_vel = np.array([interaction_entry['velocity']['x'], interaction_entry['velocity']['y']])
            
            # High intensity when close and moving
            proximity_factor = max(0, 1.0 - distance / (self.interaction_distance_threshold * 2000))
            motion_factor = (np.linalg.norm(target_vel) + np.linalg.norm(interaction_vel)) / 2.0
            
            intensity = proximity_factor * motion_factor
            interaction_info['interaction_intensity'].append(intensity)
            
            # Synchronized motion (correlation of velocity directions)
            if np.linalg.norm(target_vel) > 0 and np.linalg.norm(interaction_vel) > 0:
                target_dir = target_vel / np.linalg.norm(target_vel)
                interaction_dir = interaction_vel / np.linalg.norm(interaction_vel)
                sync = np.dot(target_dir, interaction_dir)
                interaction_info['synchronized_motion'].append(sync)
            else:
                interaction_info['synchronized_motion'].append(0.0)
        
        return interaction_info
    
    def _detect_direction_changes(self, velocities: np.ndarray) -> List[int]:
        """Detect significant direction changes in motion"""
        direction_changes = []
        
        if len(velocities) < 3:
            return direction_changes
        
        for i in range(1, len(velocities) - 1):
            prev_vel = velocities[i-1]
            curr_vel = velocities[i]
            next_vel = velocities[i+1]
            
            # Calculate angle changes
            if np.linalg.norm(prev_vel) > 0 and np.linalg.norm(next_vel) > 0:
                prev_dir = prev_vel / np.linalg.norm(prev_vel)
                next_dir = next_vel / np.linalg.norm(next_vel)
                
                angle_change = np.arccos(np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0))
                
                if angle_change > np.pi / 3:  # 60 degrees
                    direction_changes.append(i)
        
        return direction_changes
    
    def _analyze_rhythmic_patterns(self, positions: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze rhythmic patterns in motion"""
        rhythmic_analysis = {
            'dominant_frequency': 0.0,
            'rhythm_strength': 0.0,
            'periodic_components': [],
            'tempo_changes': []
        }
        
        if len(positions) < 10:
            return rhythmic_analysis
        
        # Calculate motion signal (distance from center)
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        
        # Resample to uniform time intervals
        t_uniform = np.linspace(timestamps[0], timestamps[-1], len(distances))
        
        # Perform FFT to find dominant frequencies
        fft = np.fft.fft(distances)
        freqs = np.fft.fftfreq(len(distances), d=(timestamps[-1] - timestamps[0]) / len(distances) / 1000.0)
        
        # Find dominant frequency
        positive_freqs = freqs[freqs > 0]
        positive_fft = np.abs(fft[freqs > 0])
        
        if len(positive_freqs) > 0:
            dominant_idx = np.argmax(positive_fft)
            rhythmic_analysis['dominant_frequency'] = positive_freqs[dominant_idx]
            rhythmic_analysis['rhythm_strength'] = positive_fft[dominant_idx] / np.sum(positive_fft)
        
        return rhythmic_analysis
    
    def _calculate_combined_interaction_intensity(self, interactions: Dict[str, Dict[str, Any]]) -> List[float]:
        """Calculate combined intensity from all interactions"""
        if not interactions:
            return []
        
        # Get the longest interaction sequence
        max_length = max(len(interaction['interaction_intensity']) 
                        for interaction in interactions.values() 
                        if interaction['interaction_intensity'])
        
        if max_length == 0:
            return []
        
        combined_intensity = []
        
        for i in range(max_length):
            intensity_sum = 0.0
            count = 0
            
            for interaction in interactions.values():
                if i < len(interaction['interaction_intensity']):
                    intensity_sum += interaction['interaction_intensity'][i]
                    count += 1
            
            if count > 0:
                combined_intensity.append(intensity_sum / count)
            else:
                combined_intensity.append(0.0)
        
        return combined_intensity
    
    def _generate_base_motion(self, target_motion: Dict[str, Any], sensitivity: float) -> List[Dict[str, Any]]:
        """Generate base motion curve from target part analysis"""
        base_motion = []
        
        motion_intensity = target_motion.get('motion_intensity', [])
        timestamps = target_motion.get('timestamps', [])
        
        if not motion_intensity or not timestamps:
            return base_motion
        
        # Normalize intensity to position range
        if motion_intensity:
            max_intensity = max(motion_intensity)
            if max_intensity > 0:
                normalized_intensity = np.array(motion_intensity) / max_intensity
            else:
                normalized_intensity = np.zeros(len(motion_intensity))
        else:
            normalized_intensity = np.array([])
        
        # Apply sensitivity
        adjusted_intensity = normalized_intensity * sensitivity
        
        # Convert to position range
        for i, (timestamp, intensity) in enumerate(zip(timestamps, adjusted_intensity)):
            position = int(np.clip(intensity * 100, self.position_range[0], self.position_range[1]))
            
            base_motion.append({
                'timestamp': timestamp,
                'position': position,
                'intensity': intensity,
                'source': 'base_motion'
            })
        
        return base_motion
    
    def _apply_interaction_modulations(self, base_motion: List[Dict[str, Any]], 
                                     interaction_data: Dict[str, Any], 
                                     sensitivity: float) -> List[Dict[str, Any]]:
        """Apply interaction-based modulations to base motion"""
        if not base_motion:
            return base_motion
        
        modulated_motion = base_motion.copy()
        combined_intensity = interaction_data.get('combined_intensity', [])
        
        if not combined_intensity:
            return modulated_motion
        
        # Apply interaction modulations
        for i, motion_point in enumerate(modulated_motion):
            if i < len(combined_intensity):
                interaction_intensity = combined_intensity[i]
                
                # Modulate position based on interaction intensity
                base_position = motion_point['position']
                interaction_boost = interaction_intensity * sensitivity * 30  # Scale factor
                
                new_position = base_position + interaction_boost
                new_position = int(np.clip(new_position, self.position_range[0], self.position_range[1]))
                
                motion_point['position'] = new_position
                motion_point['interaction_intensity'] = interaction_intensity
        
        return modulated_motion
    
    def _smooth_motion(self, motion: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply smoothing to motion curve"""
        if len(motion) < self.smoothing_window:
            return motion
        
        smoothed_motion = motion.copy()
        positions = [point['position'] for point in motion]
        
        # Apply moving average smoothing
        smoothed_positions = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(positions)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(positions), i + half_window + 1)
            
            window_positions = positions[start_idx:end_idx]
            smoothed_pos = np.mean(window_positions)
            smoothed_positions.append(int(smoothed_pos))
        
        # Update motion points with smoothed positions
        for i, point in enumerate(smoothed_motion):
            point['position'] = smoothed_positions[i]
        
        return smoothed_motion
    
    def _convert_to_funscript_actions(self, motion: List[Dict[str, Any]]) -> List[Dict[str, int]]:
        """Convert motion curve to funscript actions"""
        if not motion:
            return []
        
        actions = []
        
        # Resample to target sampling rate
        timestamps = [point['timestamp'] for point in motion]
        positions = [point['position'] for point in motion]
        
        if len(timestamps) < 2:
            return actions
        
        # Create interpolation function
        interp_func = interp1d(timestamps, positions, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        
        # Generate uniform time samples
        start_time = timestamps[0]
        end_time = timestamps[-1]
        sample_interval = 1000.0 / self.sampling_rate  # Convert Hz to ms
        
        sample_times = np.arange(start_time, end_time, sample_interval)
        sample_positions = interp_func(sample_times)
        
        # Convert to funscript actions
        for timestamp, position in zip(sample_times, sample_positions):
            actions.append({
                'at': int(timestamp),
                'pos': int(np.clip(position, 0, 100))
            })
        
        # Remove consecutive duplicate positions
        filtered_actions = []
        for action in actions:
            if not filtered_actions or action['pos'] != filtered_actions[-1]['pos']:
                filtered_actions.append(action)
        
        return filtered_actions
    
    def _generate_metadata(self, time_series: Dict[str, List[Dict[str, Any]]], 
                          target_part: str, interaction_parts: List[str]) -> FunscriptMetadata:
        """Generate metadata for the funscript"""
        if time_series:
            # Calculate duration from time series
            all_timestamps = []
            for part_data in time_series.values():
                if part_data:
                    all_timestamps.extend([entry['timestamp'] for entry in part_data])
            
            if all_timestamps:
                duration = int(max(all_timestamps) - min(all_timestamps))
            else:
                duration = 0
        else:
            duration = 0
        
        metadata = FunscriptMetadata(
            creator="VR Video Analyzer",
            description=f"Generated from {target_part} interactions with {', '.join(interaction_parts)}",
            duration=duration,
            fps=30.0,  # Assumed FPS
            notes=f"Target: {target_part}, Interactions: {', '.join(interaction_parts)}",
            tags=[target_part] + interaction_parts,
            type="generated"
        )
        
        return metadata
    
    def _create_empty_funscript(self) -> Dict[str, Any]:
        """Create empty funscript as fallback"""
        return {
            "version": "1.0",
            "inverted": False,
            "range": 100,
            "actions": [],
            "metadata": asdict(FunscriptMetadata())
        }
    
    def validate_funscript(self, funscript: Dict[str, Any]) -> bool:
        """Validate funscript format"""
        required_fields = ['version', 'inverted', 'range', 'actions']
        
        for field in required_fields:
            if field not in funscript:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate actions
        for action in funscript['actions']:
            if 'at' not in action or 'pos' not in action:
                logger.error("Action missing required fields")
                return False
            
            if not isinstance(action['at'], int) or not isinstance(action['pos'], int):
                logger.error("Action fields must be integers")
                return False
            
            if action['pos'] < 0 or action['pos'] > 100:
                logger.error("Action position out of range")
                return False
        
        return True
    
    def optimize_funscript(self, funscript: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize funscript for better performance"""
        actions = funscript.get('actions', [])
        
        if not actions:
            return funscript
        
        # Remove redundant actions
        optimized_actions = []
        
        for i, action in enumerate(actions):
            if i == 0 or i == len(actions) - 1:
                # Keep first and last actions
                optimized_actions.append(action)
            else:
                # Check if this action is necessary
                prev_action = actions[i-1]
                next_action = actions[i+1]
                
                # Linear interpolation check
                time_ratio = (action['at'] - prev_action['at']) / (next_action['at'] - prev_action['at'])
                expected_pos = prev_action['pos'] + time_ratio * (next_action['pos'] - prev_action['pos'])
                
                # Keep action if it deviates significantly from interpolation
                if abs(action['pos'] - expected_pos) > 2:  # 2% threshold
                    optimized_actions.append(action)
        
        funscript['actions'] = optimized_actions
        
        logger.info(f"Optimized funscript: {len(actions)} -> {len(optimized_actions)} actions")
        
        return funscript