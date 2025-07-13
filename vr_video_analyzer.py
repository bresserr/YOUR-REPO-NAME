import os
import sys
import json
import asyncio
import threading
import numpy as np
import cv2
import torch
import gradio as gr
import mediapipe as mp
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom modules
from body_part_detector import BodyPartDetector
from tensorrt_optimizer import TensorRTOptimizer
from video_processor import VideoProcessor
from movement_analyzer import MovementAnalyzer
from funscript_generator import FunscriptGenerator
from gui_components import GUIComponents

@dataclass
class BodyPartPoint:
    """Represents a body part point with coordinates and confidence"""
    x: float
    y: float
    z: float = 0.0
    confidence: float = 0.0
    frame_id: int = 0
    timestamp: float = 0.0

@dataclass
class BodyPartDetection:
    """Container for all body part detections in a frame"""
    frame_id: int
    timestamp: float
    head: Optional[BodyPartPoint] = None
    mouth: Optional[BodyPartPoint] = None
    hand_left: Optional[BodyPartPoint] = None
    hand_right: Optional[BodyPartPoint] = None
    breasts: Optional[BodyPartPoint] = None
    pelvis: Optional[BodyPartPoint] = None
    genitals: Optional[BodyPartPoint] = None

class VRVideoAnalyzer:
    """Main application class for VR video analysis"""
    
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.body_detector = BodyPartDetector()
        self.tensorrt_optimizer = TensorRTOptimizer()
        self.movement_analyzer = MovementAnalyzer()
        self.funscript_generator = FunscriptGenerator()
        self.gui = GUIComponents()
        
        # Application state
        self.current_video_path = None
        self.current_frame = None
        self.current_frame_id = 0
        self.total_frames = 0
        self.fps = 30.0
        self.is_playing = False
        self.is_paused = False
        self.detections_history = []
        self.corrected_points = {}
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # CUDA setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_video(self, video_path: str) -> Tuple[bool, str]:
        """Load video file and initialize video processor"""
        try:
            success, message = self.video_processor.load_video(video_path)
            if success:
                self.current_video_path = video_path
                self.total_frames = self.video_processor.get_total_frames()
                self.fps = self.video_processor.get_fps()
                self.current_frame_id = 0
                self.detections_history = []
                logger.info(f"Video loaded: {video_path}, frames: {self.total_frames}, fps: {self.fps}")
                return True, f"Video loaded successfully: {self.total_frames} frames at {self.fps} fps"
            return False, message
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return False, f"Error loading video: {str(e)}"
    
    def process_frame(self, frame_id: int) -> Optional[BodyPartDetection]:
        """Process a single frame and detect body parts"""
        try:
            frame = self.video_processor.get_frame(frame_id)
            if frame is None:
                return None
            
            # Run body part detection
            detections = self.body_detector.detect_body_parts(frame)
            
            # Create detection object
            detection = BodyPartDetection(
                frame_id=frame_id,
                timestamp=frame_id / self.fps,
                head=detections.get('head'),
                mouth=detections.get('mouth'),
                hand_left=detections.get('hand_left'),
                hand_right=detections.get('hand_right'),
                breasts=detections.get('breasts'),
                pelvis=detections.get('pelvis'),
                genitals=detections.get('genitals')
            )
            
            # Store in history
            self.detections_history.append(detection)
            
            return detection
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            return None
    
    def get_annotated_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get frame with body part annotations"""
        try:
            frame = self.video_processor.get_frame(frame_id)
            if frame is None:
                return None
            
            # Get or create detection for this frame
            if frame_id < len(self.detections_history):
                detection = self.detections_history[frame_id]
            else:
                detection = self.process_frame(frame_id)
            
            if detection is None:
                return frame
            
            # Annotate frame with body parts
            annotated_frame = self._annotate_frame(frame, detection)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error getting annotated frame {frame_id}: {e}")
            return None
    
    def _annotate_frame(self, frame: np.ndarray, detection: BodyPartDetection) -> np.ndarray:
        """Annotate frame with body part detections"""
        annotated = frame.copy()
        
        # Define colors for different body parts
        colors = {
            'head': (0, 255, 0),      # Green
            'mouth': (255, 0, 0),     # Blue
            'hand_left': (0, 0, 255), # Red
            'hand_right': (0, 0, 255), # Red
            'breasts': (255, 255, 0), # Cyan
            'pelvis': (255, 0, 255),  # Magenta
            'genitals': (0, 255, 255) # Yellow
        }
        
        # Draw body part points
        for part_name, color in colors.items():
            point = getattr(detection, part_name)
            if point is not None:
                x, y = int(point.x), int(point.y)
                # Draw circle for the point
                cv2.circle(annotated, (x, y), 8, color, -1)
                # Draw confidence text
                cv2.putText(annotated, f"{part_name}: {point.confidence:.2f}", 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
    
    def correct_body_part(self, frame_id: int, part_name: str, x: int, y: int) -> bool:
        """Manually correct a body part position"""
        try:
            if frame_id >= len(self.detections_history):
                return False
            
            detection = self.detections_history[frame_id]
            
            # Create corrected point
            corrected_point = BodyPartPoint(
                x=float(x),
                y=float(y),
                z=0.0,
                confidence=1.0,  # High confidence for manual correction
                frame_id=frame_id,
                timestamp=frame_id / self.fps
            )
            
            # Update detection
            setattr(detection, part_name, corrected_point)
            
            # Store correction
            if frame_id not in self.corrected_points:
                self.corrected_points[frame_id] = {}
            self.corrected_points[frame_id][part_name] = corrected_point
            
            logger.info(f"Corrected {part_name} in frame {frame_id} to ({x}, {y})")
            return True
            
        except Exception as e:
            logger.error(f"Error correcting body part: {e}")
            return False
    
    def analyze_movements(self) -> Dict[str, Any]:
        """Analyze movements and generate movement data"""
        try:
            if not self.detections_history:
                return {"error": "No detections available for analysis"}
            
            # Use movement analyzer to process detections
            analysis_results = self.movement_analyzer.analyze_detections(self.detections_history)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing movements: {e}")
            return {"error": f"Analysis error: {str(e)}"}
    
    def generate_funscript(self, output_path: str) -> Tuple[bool, str]:
        """Generate funscript file based on movement analysis"""
        try:
            # Analyze movements first
            movement_data = self.analyze_movements()
            
            if "error" in movement_data:
                return False, movement_data["error"]
            
            # Generate funscript
            success, message = self.funscript_generator.generate_script(
                movement_data, output_path, self.fps
            )
            
            return success, message
            
        except Exception as e:
            logger.error(f"Error generating funscript: {e}")
            return False, f"Error generating funscript: {str(e)}"
    
    def create_gradio_interface(self) -> gr.Blocks:
        """Create the main Gradio interface"""
        with gr.Blocks(title="VR Video Analyzer", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# VR Video Analyzer")
            gr.Markdown("Upload a VR video to analyze body parts and generate funscripts")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Video upload and controls
                    video_upload = gr.File(
                        label="Upload VR Video",
                        file_types=[".mp4", ".avi", ".mov", ".mkv"]
                    )
                    
                    with gr.Row():
                        load_btn = gr.Button("Load Video", variant="primary")
                        play_btn = gr.Button("Play")
                        pause_btn = gr.Button("Pause")
                        stop_btn = gr.Button("Stop")
                    
                    # Frame controls
                    with gr.Row():
                        frame_slider = gr.Slider(
                            label="Frame",
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=1
                        )
                        frame_input = gr.Number(
                            label="Go to Frame",
                            value=0,
                            precision=0
                        )
                    
                    # Video display
                    video_display = gr.Image(
                        label="Video Preview",
                        interactive=True,
                        height=400
                    )
                    
                    # Body part correction
                    with gr.Row():
                        part_selector = gr.Dropdown(
                            label="Select Body Part",
                            choices=["head", "mouth", "hand_left", "hand_right", 
                                   "breasts", "pelvis", "genitals"],
                            value="head"
                        )
                        correct_btn = gr.Button("Correct Position")
                
                with gr.Column(scale=1):
                    # Status and info
                    status_display = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False
                    )
                    
                    # Analysis results
                    analysis_display = gr.JSON(
                        label="Movement Analysis",
                        value={}
                    )
                    
                    # Funscript generation
                    with gr.Group():
                        gr.Markdown("### Funscript Generation")
                        output_path = gr.Textbox(
                            label="Output Path",
                            value="output.funscript",
                            placeholder="Enter output file path"
                        )
                        generate_btn = gr.Button("Generate Funscript", variant="secondary")
                        download_link = gr.File(label="Download Funscript")
                    
                    # Settings
                    with gr.Group():
                        gr.Markdown("### Settings")
                        use_tensorrt = gr.Checkbox(
                            label="Use TensorRT Optimization",
                            value=True
                        )
                        confidence_threshold = gr.Slider(
                            label="Confidence Threshold",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5
                        )
            
            # Event handlers
            def load_video_handler(video_file):
                if video_file is None:
                    return "Please upload a video file", None, gr.update(maximum=100)
                
                success, message = self.load_video(video_file.name)
                if success:
                    max_frames = self.total_frames - 1
                    return message, self.get_annotated_frame(0), gr.update(maximum=max_frames)
                return message, None, gr.update(maximum=100)
            
            def frame_change_handler(frame_id):
                if self.current_video_path is None:
                    return None
                return self.get_annotated_frame(int(frame_id))
            
            def click_handler(evt: gr.SelectData):
                if evt.index is not None:
                    x, y = evt.index
                    current_frame = int(frame_slider.value)
                    selected_part = part_selector.value
                    
                    if self.correct_body_part(current_frame, selected_part, x, y):
                        return self.get_annotated_frame(current_frame)
                return None
            
            def analyze_handler():
                if not self.detections_history:
                    return {"error": "No detections available"}
                return self.analyze_movements()
            
            def generate_funscript_handler(output_path):
                if not output_path:
                    return "Please specify output path", None
                
                success, message = self.generate_funscript(output_path)
                if success:
                    return message, output_path
                return message, None
            
            # Connect events
            load_btn.click(
                load_video_handler,
                inputs=[video_upload],
                outputs=[status_display, video_display, frame_slider]
            )
            
            frame_slider.change(
                frame_change_handler,
                inputs=[frame_slider],
                outputs=[video_display]
            )
            
            frame_input.submit(
                frame_change_handler,
                inputs=[frame_input],
                outputs=[video_display]
            )
            
            video_display.select(
                click_handler,
                outputs=[video_display]
            )
            
            correct_btn.click(
                analyze_handler,
                outputs=[analysis_display]
            )
            
            generate_btn.click(
                generate_funscript_handler,
                inputs=[output_path],
                outputs=[status_display, download_link]
            )
        
        return interface
    
    def run(self, share=False, server_name="0.0.0.0", server_port=7860):
        """Run the Gradio interface"""
        interface = self.create_gradio_interface()
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            debug=True
        )

def main():
    """Main entry point with command line argument support"""
    parser = argparse.ArgumentParser(
        description="VR Video Analyzer - Analyze VR videos and generate funscripts"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    
    parser.add_argument(
        "--server-name",
        default="0.0.0.0",
        help="Server hostname (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )
    
    parser.add_argument(
        "--video",
        help="Path to video file to analyze"
    )
    
    parser.add_argument(
        "--output",
        help="Output path for funscript file"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode (no GUI)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Initialize analyzer
    analyzer = VRVideoAnalyzer()
    
    # Batch mode
    if args.batch:
        if not args.video:
            print("Error: --video is required in batch mode")
            sys.exit(1)
        
        if not args.output:
            print("Error: --output is required in batch mode")
            sys.exit(1)
        
        print(f"Processing video: {args.video}")
        
        # Load video
        success, message = analyzer.load_video(args.video)
        if not success:
            print(f"Error loading video: {message}")
            sys.exit(1)
        
        print(f"Video loaded: {message}")
        
        # Process all frames
        print("Processing frames...")
        for frame_id in range(analyzer.total_frames):
            if frame_id % 100 == 0:
                print(f"Processed {frame_id}/{analyzer.total_frames} frames")
            analyzer.process_frame(frame_id)
        
        print("Analyzing movements...")
        analysis = analyzer.analyze_movements()
        
        if "error" in analysis:
            print(f"Error in analysis: {analysis['error']}")
            sys.exit(1)
        
        print("Generating funscript...")
        success, message = analyzer.generate_funscript(args.output)
        
        if success:
            print(f"Funscript generated: {message}")
            print(f"Output saved to: {args.output}")
        else:
            print(f"Error generating funscript: {message}")
            sys.exit(1)
    
    else:
        # GUI mode
        print("Starting VR Video Analyzer...")
        print(f"Web interface will be available at: http://{args.server_name}:{args.server_port}")
        
        if args.video:
            print(f"Pre-loading video: {args.video}")
            analyzer.load_video(args.video)
        
        analyzer.run(
            share=args.share,
            server_name=args.server_name,
            server_port=args.server_port
        )

if __name__ == "__main__":
    main()