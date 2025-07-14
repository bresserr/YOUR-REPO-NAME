import gradio as gr
import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class GUIComponents:
    """Helper class for creating and managing GUI components"""
    
    def __init__(self):
        self.theme = gr.themes.Soft()
        self.colors = {
            'primary': '#3B82F6',
            'secondary': '#10B981',
            'danger': '#EF4444',
            'warning': '#F59E0B',
            'info': '#06B6D4',
            'success': '#10B981'
        }
        
    def create_video_player(self, height: int = 400) -> gr.Image:
        """Create a video player component"""
        return gr.Image(
            label="Video Preview",
            interactive=True,
            height=height,
            show_download_button=True,
            show_share_button=False
        )
    
    def create_body_part_selector(self) -> gr.Dropdown:
        """Create body part selector dropdown"""
        return gr.Dropdown(
            label="Select Body Part to Correct",
            choices=[
                "head",
                "mouth", 
                "hand_left",
                "hand_right",
                "breasts",
                "pelvis",
                "genitals"
            ],
            value="head",
            info="Select a body part to manually correct its position"
        )
    
    def create_analysis_display(self) -> gr.JSON:
        """Create analysis results display"""
        return gr.JSON(
            label="Movement Analysis Results",
            value={},
            show_label=True
        )
    
    def create_settings_panel(self) -> Dict[str, gr.Component]:
        """Create settings panel components"""
        components = {}
        
        with gr.Group():
            gr.Markdown("### Detection Settings")
            
            components['confidence_threshold'] = gr.Slider(
                label="Confidence Threshold",
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.05,
                info="Minimum confidence for body part detection"
            )
            
            components['use_tensorrt'] = gr.Checkbox(
                label="Use TensorRT Optimization",
                value=True,
                info="Use TensorRT for faster inference (requires CUDA)"
            )
            
            components['smooth_tracking'] = gr.Checkbox(
                label="Smooth Tracking",
                value=True,
                info="Apply smoothing to tracked body parts"
            )
        
        with gr.Group():
            gr.Markdown("### Funscript Settings")
            
            components['min_action_interval'] = gr.Slider(
                label="Minimum Action Interval (ms)",
                minimum=50,
                maximum=500,
                value=100,
                step=10,
                info="Minimum time between funscript actions"
            )
            
            components['intensity_multiplier'] = gr.Slider(
                label="Intensity Multiplier",
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                info="Multiply detected intensity by this factor"
            )
            
            components['smoothing_factor'] = gr.Slider(
                label="Smoothing Factor",
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.05,
                info="Amount of smoothing to apply to funscript"
            )
        
        return components
    
    def create_progress_bar(self) -> gr.HTML:
        """Create a progress bar component"""
        return gr.HTML(
            value="<div style='width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px;'><div style='width: 0%; height: 100%; background-color: #3B82F6; border-radius: 10px; transition: width 0.3s;'></div></div>",
            visible=False
        )
    
    def update_progress_bar(self, progress: float) -> str:
        """Update progress bar HTML"""
        percentage = max(0, min(100, progress * 100))
        return f"""
        <div style='width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; margin: 10px 0;'>
            <div style='width: {percentage}%; height: 100%; background-color: #3B82F6; border-radius: 10px; transition: width 0.3s;'></div>
        </div>
        <div style='text-align: center; margin-top: 5px;'>Processing: {percentage:.1f}%</div>
        """
    
    def create_status_indicator(self) -> gr.HTML:
        """Create status indicator"""
        return gr.HTML(
            value=self._generate_status_html("ready", "Ready"),
            visible=True
        )
    
    def _generate_status_html(self, status: str, message: str) -> str:
        """Generate HTML for status indicator"""
        color_map = {
            'ready': '#10B981',
            'processing': '#F59E0B', 
            'error': '#EF4444',
            'success': '#10B981'
        }
        
        color = color_map.get(status, '#6B7280')
        
        return f"""
        <div style='display: flex; align-items: center; padding: 10px; border-radius: 5px; background-color: {color}20;'>
            <div style='width: 12px; height: 12px; border-radius: 50%; background-color: {color}; margin-right: 10px;'></div>
            <span style='color: {color}; font-weight: 500;'>{message}</span>
        </div>
        """
    
    def create_movement_plot(self, movement_data: Dict[str, Any]) -> go.Figure:
        """Create movement visualization plot"""
        try:
            fig = go.Figure()
            
            if not movement_data or "movement_metrics" not in movement_data:
                fig.add_annotation(
                    text="No movement data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                return fig
            
            metrics = movement_data["movement_metrics"]
            
            # Plot velocities for each body part
            for part_name, part_data in metrics.items():
                if "velocities" in part_data and "timestamps" in part_data:
                    timestamps = part_data["timestamps"]
                    velocities = part_data["velocities"]
                    
                    # Skip if not enough data
                    if len(timestamps) < 2 or len(velocities) < 1:
                        continue
                    
                    # Use timestamps[1:] since velocities has one fewer element
                    plot_timestamps = timestamps[1:len(velocities)+1]
                    
                    fig.add_trace(go.Scatter(
                        x=plot_timestamps,
                        y=velocities,
                        mode='lines',
                        name=f"{part_name} velocity",
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title="Body Part Movement Velocities",
                xaxis_title="Time (seconds)",
                yaxis_title="Velocity (pixels/second)",
                template="plotly_white",
                hovermode="x unified"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating movement plot: {e}")
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig
    
    def create_interaction_timeline(self, interaction_data: List[Dict[str, Any]]) -> go.Figure:
        """Create interaction timeline visualization"""
        try:
            fig = go.Figure()
            
            if not interaction_data:
                fig.add_annotation(
                    text="No interaction data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                return fig
            
            # Group interactions by type
            interaction_types = {}
            for interaction in interaction_data:
                interaction_type = f"{interaction['primary_part']}-{interaction['secondary_part']}"
                if interaction_type not in interaction_types:
                    interaction_types[interaction_type] = []
                interaction_types[interaction_type].append(interaction)
            
            # Create timeline bars
            y_position = 0
            for interaction_type, interactions in interaction_types.items():
                for interaction in interactions:
                    fig.add_trace(go.Scatter(
                        x=[interaction['timestamp'], interaction['timestamp'] + interaction['duration']],
                        y=[y_position, y_position],
                        mode='lines',
                        line=dict(width=10, color=px.colors.qualitative.Set3[y_position % len(px.colors.qualitative.Set3)]),
                        name=interaction_type,
                        showlegend=False,
                        hovertemplate=f"<b>{interaction_type}</b><br>Start: {interaction['timestamp']:.2f}s<br>Duration: {interaction['duration']:.2f}s<br>Intensity: {interaction['intensity']:.2f}<extra></extra>"
                    ))
                
                y_position += 1
            
            fig.update_layout(
                title="Interaction Timeline",
                xaxis_title="Time (seconds)",
                yaxis_title="Interaction Type",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(interaction_types))),
                    ticktext=list(interaction_types.keys())
                ),
                template="plotly_white",
                height=max(400, len(interaction_types) * 50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interaction timeline: {e}")
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating timeline: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig
    
    def create_funscript_preview(self, funscript_path: str) -> go.Figure:
        """Create funscript preview visualization"""
        try:
            fig = go.Figure()
            
            # Load funscript
            with open(funscript_path, 'r') as f:
                funscript_data = json.load(f)
            
            actions = funscript_data.get('actions', [])
            
            if not actions:
                fig.add_annotation(
                    text="No actions in funscript",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                return fig
            
            # Extract timestamps and positions
            timestamps = [action['at'] / 1000 for action in actions]  # Convert to seconds
            positions = [action['pos'] for action in actions]
            
            # Create line plot
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=positions,
                mode='lines+markers',
                name='Position',
                line=dict(color='#3B82F6', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title="Funscript Preview",
                xaxis_title="Time (seconds)",
                yaxis_title="Position (0-100)",
                template="plotly_white",
                yaxis=dict(range=[0, 100])
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating funscript preview: {e}")
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading funscript: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig
    
    def create_stats_display(self, stats: Dict[str, Any]) -> str:
        """Create formatted stats display"""
        try:
            if not stats:
                return "No statistics available"
            
            html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;'>"
            
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    html += f"""
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;'>
                        <div style='font-size: 24px; font-weight: bold; color: #3B82F6;'>{formatted_value}</div>
                        <div style='font-size: 14px; color: #6B7280; margin-top: 5px;'>{key.replace('_', ' ').title()}</div>
                    </div>
                    """
                elif isinstance(value, str):
                    html += f"""
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;'>
                        <div style='font-size: 16px; font-weight: bold; color: #3B82F6;'>{value}</div>
                        <div style='font-size: 14px; color: #6B7280; margin-top: 5px;'>{key.replace('_', ' ').title()}</div>
                    </div>
                    """
            
            html += "</div>"
            return html
            
        except Exception as e:
            logger.error(f"Error creating stats display: {e}")
            return f"Error displaying stats: {str(e)}"
    
    def create_help_text(self) -> str:
        """Create help text for the application"""
        return """
        ## How to use VR Video Analyzer:
        
        1. **Upload Video**: Click "Upload VR Video" and select your video file
        2. **Load Video**: Click "Load Video" to process the video
        3. **Navigate**: Use the frame slider or input to navigate through the video
        4. **Correct Body Parts**: 
           - Select a body part from the dropdown
           - Click on the video where the body part should be positioned
           - The system will learn from your corrections
        5. **Generate Funscript**: 
           - Set your desired output path
           - Click "Generate Funscript" to create the control file
        
        ## Features:
        - **Real-time body part detection** using MediaPipe
        - **Manual correction** of detection errors
        - **Movement analysis** and interaction detection
        - **Funscript generation** for compatible devices
        - **TensorRT optimization** for faster processing
        
        ## Tips:
        - Use higher confidence thresholds for more accurate detection
        - Enable TensorRT optimization if you have a compatible GPU
        - Manually correct key frames for better results
        - Adjust intensity multiplier based on your preferences
        """
    
    def create_keyboard_shortcuts(self) -> gr.HTML:
        """Create keyboard shortcuts help"""
        shortcuts_html = """
        <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <h4>Keyboard Shortcuts:</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li><strong>Space</strong>: Play/Pause</li>
                <li><strong>←/→</strong>: Previous/Next frame</li>
                <li><strong>Ctrl+S</strong>: Save current corrections</li>
                <li><strong>Ctrl+G</strong>: Generate funscript</li>
                <li><strong>Ctrl+R</strong>: Reset corrections</li>
            </ul>
        </div>
        """
        return gr.HTML(shortcuts_html)
    
    def format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def create_export_options(self) -> Dict[str, gr.Component]:
        """Create export options panel"""
        components = {}
        
        with gr.Group():
            gr.Markdown("### Export Options")
            
            components['export_format'] = gr.Dropdown(
                label="Export Format",
                choices=["funscript", "csv", "json"],
                value="funscript",
                info="Choose export format"
            )
            
            components['include_metadata'] = gr.Checkbox(
                label="Include Metadata",
                value=True,
                info="Include analysis metadata in export"
            )
            
            components['compress_output'] = gr.Checkbox(
                label="Compress Output",
                value=False,
                info="Compress output file (zip)"
            )
        
        return components