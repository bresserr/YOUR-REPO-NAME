import json
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FunscriptAction:
    """Represents a single action in a funscript"""
    at: int  # Timestamp in milliseconds
    pos: int  # Position from 0-100

@dataclass
class FunscriptMetadata:
    """Metadata for the funscript"""
    type: str = "basic"
    version: str = "1.0"
    inverted: bool = False
    range: int = 100
    creator: str = "VR Video Analyzer"
    description: str = "Generated from VR video analysis"

class FunscriptGenerator:
    """Generates funscript files from movement analysis data"""
    
    def __init__(self):
        self.min_action_interval = 100  # Minimum ms between actions
        self.max_position = 100
        self.min_position = 0
        self.smoothing_factor = 0.3
        self.intensity_multiplier = 1.0
        
        logger.info("Funscript generator initialized")
    
    def generate_script(self, movement_data: Dict[str, Any], output_path: str, fps: float = 30.0) -> Tuple[bool, str]:
        """Generate funscript from movement analysis data"""
        try:
            if "error" in movement_data:
                return False, f"Movement data error: {movement_data['error']}"
            
            # Extract movement patterns
            movement_patterns = movement_data.get("movement_patterns", {})
            
            if "error" in movement_patterns:
                return False, f"Movement patterns error: {movement_patterns['error']}"
            
            # Generate actions from stroke patterns
            actions = self._generate_actions_from_patterns(movement_patterns, fps)
            
            if not actions:
                return False, "No actions generated from movement patterns"
            
            # Apply smoothing and filtering
            smoothed_actions = self._smooth_actions(actions)
            
            # Filter out actions that are too close together
            filtered_actions = self._filter_actions(smoothed_actions)
            
            # Create funscript structure
            funscript = self._create_funscript_structure(filtered_actions, movement_data)
            
            # Write to file
            success = self._write_funscript(funscript, output_path)
            
            if success:
                return True, f"Funscript generated successfully: {len(filtered_actions)} actions"
            else:
                return False, "Failed to write funscript file"
                
        except Exception as e:
            logger.error(f"Error generating funscript: {e}")
            return False, f"Generation error: {str(e)}"
    
    def _generate_actions_from_patterns(self, movement_patterns: Dict[str, Any], fps: float) -> List[FunscriptAction]:
        """Generate actions from movement patterns"""
        actions = []
        
        try:
            stroke_patterns = movement_patterns.get("stroke_patterns", [])
            
            for pattern in stroke_patterns:
                pattern_actions = self._process_stroke_pattern(pattern, fps)
                actions.extend(pattern_actions)
            
            # If no stroke patterns, generate from intensity curve
            if not actions:
                intensity_curve = movement_patterns.get("intensity_curve", [])
                if intensity_curve:
                    actions = self._generate_from_intensity_curve(intensity_curve, fps)
            
            # Sort actions by timestamp
            actions.sort(key=lambda x: x.at)
            
            return actions
            
        except Exception as e:
            logger.error(f"Error generating actions from patterns: {e}")
            return []
    
    def _process_stroke_pattern(self, pattern: Dict[str, Any], fps: float) -> List[FunscriptAction]:
        """Process a single stroke pattern into actions"""
        actions = []
        
        try:
            timestamp = pattern.get("timestamp", 0.0)
            duration = pattern.get("duration", 0.0)
            positions = pattern.get("positions", [])
            
            if not positions:
                return []
            
            # Calculate time intervals
            start_time_ms = int(timestamp * 1000)
            duration_ms = int(duration * 1000)
            
            if len(positions) > 1:
                interval_ms = duration_ms / (len(positions) - 1)
            else:
                interval_ms = duration_ms
            
            # Generate actions for each position
            for i, position in enumerate(positions):
                action_time = start_time_ms + int(i * interval_ms)
                
                # Clamp position to valid range
                clamped_pos = max(self.min_position, min(self.max_position, position))
                
                actions.append(FunscriptAction(
                    at=action_time,
                    pos=clamped_pos
                ))
            
            return actions
            
        except Exception as e:
            logger.error(f"Error processing stroke pattern: {e}")
            return []
    
    def _generate_from_intensity_curve(self, intensity_curve: List[Dict[str, Any]], fps: float) -> List[FunscriptAction]:
        """Generate actions from intensity curve when no stroke patterns available"""
        actions = []
        
        try:
            # Create basic up/down pattern based on intensity
            time_offset = 0
            
            for curve_point in intensity_curve:
                avg_intensity = curve_point.get("avg_intensity", 0.5)
                max_intensity = curve_point.get("max_intensity", 1.0)
                
                # Convert intensity to position (0-100)
                base_position = int(avg_intensity * 100)
                max_position = int(max_intensity * 100)
                
                # Generate simple up/down pattern
                pattern_duration = 2000  # 2 seconds per pattern
                
                # Down stroke
                actions.append(FunscriptAction(
                    at=time_offset,
                    pos=max_position
                ))
                
                # Up stroke
                actions.append(FunscriptAction(
                    at=time_offset + pattern_duration // 2,
                    pos=base_position
                ))
                
                time_offset += pattern_duration
            
            return actions
            
        except Exception as e:
            logger.error(f"Error generating from intensity curve: {e}")
            return []
    
    def _smooth_actions(self, actions: List[FunscriptAction]) -> List[FunscriptAction]:
        """Apply smoothing to actions"""
        if len(actions) < 3:
            return actions
        
        try:
            smoothed_actions = []
            
            for i in range(len(actions)):
                if i == 0 or i == len(actions) - 1:
                    # Keep first and last actions unchanged
                    smoothed_actions.append(actions[i])
                else:
                    # Apply smoothing
                    prev_pos = actions[i-1].pos
                    curr_pos = actions[i].pos
                    next_pos = actions[i+1].pos
                    
                    # Simple moving average
                    smoothed_pos = int((prev_pos + curr_pos + next_pos) / 3)
                    
                    # Apply smoothing factor
                    final_pos = int(curr_pos * (1 - self.smoothing_factor) + 
                                   smoothed_pos * self.smoothing_factor)
                    
                    smoothed_actions.append(FunscriptAction(
                        at=actions[i].at,
                        pos=final_pos
                    ))
            
            return smoothed_actions
            
        except Exception as e:
            logger.error(f"Error smoothing actions: {e}")
            return actions
    
    def _filter_actions(self, actions: List[FunscriptAction]) -> List[FunscriptAction]:
        """Filter out actions that are too close together"""
        if not actions:
            return actions
        
        try:
            filtered_actions = [actions[0]]  # Always keep first action
            
            for action in actions[1:]:
                last_action = filtered_actions[-1]
                
                # Check time interval
                if action.at - last_action.at >= self.min_action_interval:
                    filtered_actions.append(action)
                else:
                    # Update position of last action if this one is more extreme
                    if abs(action.pos - 50) > abs(last_action.pos - 50):
                        filtered_actions[-1] = FunscriptAction(
                            at=last_action.at,
                            pos=action.pos
                        )
            
            return filtered_actions
            
        except Exception as e:
            logger.error(f"Error filtering actions: {e}")
            return actions
    
    def _create_funscript_structure(self, actions: List[FunscriptAction], movement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create the complete funscript structure"""
        try:
            # Convert actions to the funscript format
            action_list = []
            for action in actions:
                action_list.append({
                    "at": action.at,
                    "pos": action.pos
                })
            
            # Create metadata
            metadata = FunscriptMetadata(
                description=f"Generated from VR video analysis - {movement_data.get('total_frames', 0)} frames"
            )
            
            # Create complete funscript
            funscript = {
                "version": metadata.version,
                "inverted": metadata.inverted,
                "range": metadata.range,
                "actions": action_list,
                "metadata": {
                    "type": metadata.type,
                    "creator": metadata.creator,
                    "description": metadata.description,
                    "duration": movement_data.get("duration", 0.0),
                    "total_actions": len(action_list),
                    "generation_info": {
                        "total_frames": movement_data.get("total_frames", 0),
                        "patterns_used": len(movement_data.get("movement_patterns", {}).get("stroke_patterns", [])),
                        "intensity_levels": len(movement_data.get("movement_patterns", {}).get("intensity_curve", [])),
                        "rhythm_type": movement_data.get("movement_patterns", {}).get("rhythm_analysis", {}).get("pattern_type", "unknown")
                    }
                }
            }
            
            return funscript
            
        except Exception as e:
            logger.error(f"Error creating funscript structure: {e}")
            return {}
    
    def _write_funscript(self, funscript: Dict[str, Any], output_path: str) -> bool:
        """Write funscript to file"""
        try:
            output_file = Path(output_path)
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(funscript, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Funscript written to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing funscript to {output_path}: {e}")
            return False
    
    def validate_funscript(self, funscript_path: str) -> Tuple[bool, str]:
        """Validate a funscript file"""
        try:
            with open(funscript_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required fields
            required_fields = ['version', 'actions']
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field: {field}"
            
            # Validate actions
            actions = data.get('actions', [])
            if not isinstance(actions, list):
                return False, "Actions must be a list"
            
            for i, action in enumerate(actions):
                if not isinstance(action, dict):
                    return False, f"Action {i} must be a dictionary"
                
                if 'at' not in action or 'pos' not in action:
                    return False, f"Action {i} missing 'at' or 'pos'"
                
                if not isinstance(action['at'], int) or not isinstance(action['pos'], int):
                    return False, f"Action {i} 'at' and 'pos' must be integers"
                
                if action['pos'] < 0 or action['pos'] > 100:
                    return False, f"Action {i} position out of range (0-100)"
            
            # Check if actions are sorted by time
            for i in range(1, len(actions)):
                if actions[i]['at'] < actions[i-1]['at']:
                    return False, f"Actions not sorted by time at index {i}"
            
            return True, f"Valid funscript with {len(actions)} actions"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update generator settings"""
        try:
            if 'min_action_interval' in settings:
                self.min_action_interval = settings['min_action_interval']
            
            if 'smoothing_factor' in settings:
                self.smoothing_factor = settings['smoothing_factor']
            
            if 'intensity_multiplier' in settings:
                self.intensity_multiplier = settings['intensity_multiplier']
            
            logger.info("Funscript generator settings updated")
            
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
    
    def analyze_funscript(self, funscript_path: str) -> Dict[str, Any]:
        """Analyze an existing funscript file"""
        try:
            with open(funscript_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            actions = data.get('actions', [])
            
            if not actions:
                return {"error": "No actions found in funscript"}
            
            # Calculate statistics
            positions = [action['pos'] for action in actions]
            timestamps = [action['at'] for action in actions]
            
            # Calculate intervals
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            
            # Calculate position changes
            position_changes = [abs(positions[i+1] - positions[i]) for i in range(len(positions)-1)]
            
            analysis = {
                "total_actions": len(actions),
                "duration_ms": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
                "avg_interval_ms": np.mean(intervals) if intervals else 0,
                "min_interval_ms": np.min(intervals) if intervals else 0,
                "max_interval_ms": np.max(intervals) if intervals else 0,
                "avg_position": np.mean(positions),
                "min_position": np.min(positions),
                "max_position": np.max(positions),
                "position_range": np.max(positions) - np.min(positions),
                "avg_position_change": np.mean(position_changes) if position_changes else 0,
                "max_position_change": np.max(position_changes) if position_changes else 0,
                "actions_per_second": len(actions) / (timestamps[-1] / 1000) if timestamps[-1] > 0 else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing funscript: {e}")
            return {"error": f"Analysis error: {str(e)}"}
    
    def merge_funscripts(self, funscript_paths: List[str], output_path: str) -> Tuple[bool, str]:
        """Merge multiple funscripts into one"""
        try:
            all_actions = []
            
            for path in funscript_paths:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    actions = data.get('actions', [])
                    all_actions.extend(actions)
            
            # Sort by timestamp
            all_actions.sort(key=lambda x: x['at'])
            
            # Remove duplicates
            unique_actions = []
            for action in all_actions:
                if not unique_actions or action['at'] != unique_actions[-1]['at']:
                    unique_actions.append(action)
            
            # Create merged funscript
            merged = {
                "version": "1.0",
                "inverted": False,
                "range": 100,
                "actions": unique_actions,
                "metadata": {
                    "type": "merged",
                    "creator": "VR Video Analyzer",
                    "description": f"Merged from {len(funscript_paths)} funscripts",
                    "source_files": funscript_paths
                }
            }
            
            # Write merged funscript
            success = self._write_funscript(merged, output_path)
            
            if success:
                return True, f"Merged {len(funscript_paths)} funscripts into {len(unique_actions)} actions"
            else:
                return False, "Failed to write merged funscript"
                
        except Exception as e:
            logger.error(f"Error merging funscripts: {e}")
            return False, f"Merge error: {str(e)}"