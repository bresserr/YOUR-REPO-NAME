import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import timedelta

@dataclass
class FunscriptAction:
    """Represents a single action in the funscript"""
    at: int  # Time in milliseconds
    pos: int  # Position (0-100)

class FunscriptGenerator:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.frame_duration_ms = 1000.0 / fps
        self.actions: List[FunscriptAction] = []
        
    def analyze_interactions(self, 
                           interaction_history: List[Dict],
                           movement_history: List[Dict[str, float]]) -> List[FunscriptAction]:
        """Analyze interaction history and generate funscript actions"""
        actions = []
        
        for i, (interactions, movements) in enumerate(zip(interaction_history, movement_history)):
            timestamp_ms = int(i * self.frame_duration_ms)
            
            # Check for genital interactions
            genital_interactions = self._get_genital_interactions(interactions)
            
            if genital_interactions:
                # Calculate position based on interaction type and distance
                position = self._calculate_position(genital_interactions, movements)
                
                # Add action if significantly different from previous
                if not actions or abs(actions[-1].pos - position) > 5:
                    actions.append(FunscriptAction(at=timestamp_ms, pos=position))
        
        # Smooth the actions
        self.actions = self._smooth_actions(actions)
        return self.actions
    
    def _get_genital_interactions(self, interactions: Dict) -> List[Tuple[str, str, float]]:
        """Extract interactions involving genitals"""
        if not interactions:
            return []
        
        genital_interactions = []
        for interaction in interactions:
            if len(interaction) >= 3 and 'genitals' in interaction[:2]:
                genital_interactions.append(interaction)
        
        return genital_interactions
    
    def _calculate_position(self, 
                          interactions: List[Tuple[str, str, float]], 
                          movements: Dict[str, float]) -> int:
        """Calculate funscript position based on interactions and movements"""
        if not interactions:
            return 50  # Default middle position
        
        # Find the closest interaction
        closest_interaction = min(interactions, key=lambda x: x[2])
        interacting_part = closest_interaction[0] if closest_interaction[0] != 'genitals' else closest_interaction[1]
        distance = closest_interaction[2]
        
        # Base position on interaction type
        base_positions = {
            'hand_left': 70,
            'hand_right': 70,
            'mouth': 90,
            'pelvis': 80,
            'breasts': 60
        }
        
        base_pos = base_positions.get(interacting_part, 50)
        
        # Adjust based on distance (closer = deeper stroke)
        if distance < 20:
            position = min(100, base_pos + 20)
        elif distance < 40:
            position = base_pos + 10
        else:
            position = max(0, base_pos - 10)
        
        # Adjust based on movement speed
        if interacting_part in movements:
            speed = movements[interacting_part]
            if speed > 10:  # Fast movement
                position = min(100, position + 10)
        
        return int(position)
    
    def _smooth_actions(self, actions: List[FunscriptAction], 
                       window_size: int = 3) -> List[FunscriptAction]:
        """Smooth the action positions to avoid jerky movements"""
        if len(actions) < window_size:
            return actions
        
        smoothed = []
        positions = [a.pos for a in actions]
        
        # Apply moving average
        for i in range(len(actions)):
            start = max(0, i - window_size // 2)
            end = min(len(actions), i + window_size // 2 + 1)
            avg_pos = int(np.mean(positions[start:end]))
            
            smoothed.append(FunscriptAction(at=actions[i].at, pos=avg_pos))
        
        return smoothed
    
    def add_manual_action(self, timestamp_ms: int, position: int):
        """Add a manually corrected action"""
        # Find the correct position to insert
        insert_idx = 0
        for i, action in enumerate(self.actions):
            if action.at > timestamp_ms:
                insert_idx = i
                break
            elif action.at == timestamp_ms:
                # Replace existing action
                self.actions[i] = FunscriptAction(at=timestamp_ms, pos=position)
                return
        else:
            insert_idx = len(self.actions)
        
        self.actions.insert(insert_idx, FunscriptAction(at=timestamp_ms, pos=position))
    
    def generate_pattern(self, start_ms: int, duration_ms: int, 
                        pattern_type: str = 'stroke', 
                        speed: float = 1.0) -> List[FunscriptAction]:
        """Generate predefined patterns"""
        actions = []
        
        if pattern_type == 'stroke':
            # Basic up-down stroke pattern
            stroke_duration = int(1000 / speed)  # Duration of one stroke cycle
            num_strokes = duration_ms // stroke_duration
            
            for i in range(num_strokes):
                base_time = start_ms + i * stroke_duration
                # Down stroke
                actions.append(FunscriptAction(at=base_time, pos=90))
                # Up stroke
                actions.append(FunscriptAction(at=base_time + stroke_duration // 2, pos=10))
        
        elif pattern_type == 'vibrate':
            # Rapid small movements
            vibrate_interval = int(100 / speed)
            num_vibes = duration_ms // vibrate_interval
            
            for i in range(num_vibes):
                time = start_ms + i * vibrate_interval
                pos = 50 + (10 if i % 2 == 0 else -10)
                actions.append(FunscriptAction(at=time, pos=pos))
        
        elif pattern_type == 'edge':
            # Slow teasing movements
            edge_duration = int(2000 / speed)
            num_edges = duration_ms // edge_duration
            
            for i in range(num_edges):
                base_time = start_ms + i * edge_duration
                actions.append(FunscriptAction(at=base_time, pos=70))
                actions.append(FunscriptAction(at=base_time + edge_duration // 3, pos=75))
                actions.append(FunscriptAction(at=base_time + 2 * edge_duration // 3, pos=65))
        
        return actions
    
    def export_funscript(self, filename: str, metadata: Dict = None):
        """Export the funscript to a JSON file"""
        funscript = {
            "version": "1.0",
            "inverted": False,
            "range": 100,
            "actions": [asdict(action) for action in self.actions]
        }
        
        if metadata:
            funscript["metadata"] = metadata
        
        with open(filename, 'w') as f:
            json.dump(funscript, f, indent=2)
    
    def import_funscript(self, filename: str):
        """Import an existing funscript file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.actions = [
            FunscriptAction(at=a['at'], pos=a['pos']) 
            for a in data.get('actions', [])
        ]
        
        return data.get('metadata', {})