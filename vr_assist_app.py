"""
VR Assistive Technology Application
Body part detection and tracking for 3D VR video with funscript generation
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

from body_detector import BodyPartDetector
from video_processor import VideoProcessor
from funscript_generator import FunscriptGenerator
from tracking_manager import TrackingManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(title="VR Assistive Technology", description="Body part detection and tracking for accessibility")

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global state
class AppState:
    def __init__(self):
        self.detector = None
        self.processor = None
        self.tracking_manager = None
        self.funscript_generator = None
        self.current_video_path = None
        self.is_processing = False
        self.connected_clients = set()
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing VR Assistive Technology components...")
        
        # Initialize body part detector with CUDA/TensorRT
        self.detector = BodyPartDetector(use_tensorrt=True)
        await self.detector.initialize()
        
        # Initialize video processor
        self.processor = VideoProcessor()
        
        # Initialize tracking manager
        self.tracking_manager = TrackingManager()
        
        # Initialize funscript generator
        self.funscript_generator = FunscriptGenerator()
        
        logger.info("All components initialized successfully")

# Global app state
app_state = AppState()

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    await app_state.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if app_state.detector:
        app_state.detector.cleanup()

# Data models
class TrackingPoint(BaseModel):
    x: float
    y: float
    confidence: float
    timestamp: float

class BodyPartData(BaseModel):
    name: str
    points: List[TrackingPoint]
    is_corrected: bool = False

class AnalysisResult(BaseModel):
    frame_number: int
    timestamp: float
    body_parts: List[BodyPartData]
    interactions: Dict[str, float]  # e.g., {"speed": 0.5, "depth": 0.3}

# Web routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Main application page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    """Upload VR video for processing"""
    try:
        # Save uploaded video
        video_path = f"uploads/{video.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        app_state.current_video_path = video_path
        
        # Get video info
        video_info = await app_state.processor.get_video_info(video_path)
        
        return {
            "success": True,
            "video_path": video_path,
            "video_info": video_info
        }
    
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/start-analysis")
async def start_analysis():
    """Start video analysis and tracking"""
    if not app_state.current_video_path:
        return {"success": False, "error": "No video uploaded"}
    
    try:
        app_state.is_processing = True
        
        # Start processing in background
        asyncio.create_task(process_video_background())
        
        return {"success": True, "message": "Analysis started"}
    
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/pause-analysis")
async def pause_analysis():
    """Pause video analysis"""
    app_state.is_processing = False
    return {"success": True, "message": "Analysis paused"}

@app.post("/correct-tracking")
async def correct_tracking(correction_data: Dict):
    """Correct tracking points manually"""
    try:
        frame_number = correction_data["frame_number"]
        body_part = correction_data["body_part"]
        new_position = correction_data["position"]
        
        # Update tracking data
        app_state.tracking_manager.correct_tracking(
            frame_number, body_part, new_position
        )
        
        return {"success": True, "message": "Tracking corrected"}
    
    except Exception as e:
        logger.error(f"Error correcting tracking: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/generate-funscript")
async def generate_funscript():
    """Generate funscript file from tracking data"""
    try:
        if not app_state.tracking_manager.has_tracking_data():
            return {"success": False, "error": "No tracking data available"}
        
        # Generate funscript
        funscript_data = await app_state.funscript_generator.generate_from_tracking(
            app_state.tracking_manager.get_tracking_data()
        )
        
        # Save funscript file
        output_path = "output/generated.funscript"
        os.makedirs("output", exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(funscript_data, f, indent=2)
        
        return {
            "success": True,
            "funscript_path": output_path,
            "actions_count": len(funscript_data.get("actions", []))
        }
    
    except Exception as e:
        logger.error(f"Error generating funscript: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/download-funscript")
async def download_funscript():
    """Download generated funscript file"""
    funscript_path = "output/generated.funscript"
    if os.path.exists(funscript_path):
        return FileResponse(
            funscript_path,
            media_type="application/json",
            filename="generated.funscript"
        )
    return {"error": "No funscript file available"}

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    app_state.connected_clients.add(websocket)
    
    try:
        while True:
            # Keep connection alive and handle messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message["type"] == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message["type"] == "get_status":
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "is_processing": app_state.is_processing,
                    "has_video": app_state.current_video_path is not None
                }))
    
    except WebSocketDisconnect:
        app_state.connected_clients.discard(websocket)

async def broadcast_to_clients(message: Dict):
    """Broadcast message to all connected WebSocket clients"""
    if app_state.connected_clients:
        disconnected = set()
        for client in app_state.connected_clients:
            try:
                await client.send_text(json.dumps(message))
            except:
                disconnected.add(client)
        
        # Remove disconnected clients
        app_state.connected_clients -= disconnected

async def process_video_background():
    """Background task to process video and detect body parts"""
    try:
        logger.info("Starting video processing...")
        
        # Open video
        cap = cv2.VideoCapture(app_state.current_video_path)
        frame_number = 0
        
        while cap.isOpened() and app_state.is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect body parts
            detections = await app_state.detector.detect_body_parts(frame)
            
            # Update tracking
            app_state.tracking_manager.update_tracking(frame_number, detections)
            
            # Create analysis result
            analysis_result = AnalysisResult(
                frame_number=frame_number,
                timestamp=frame_number / cap.get(cv2.CAP_PROP_FPS),
                body_parts=detections,
                interactions=app_state.tracking_manager.get_current_interactions()
            )
            
            # Broadcast to clients
            await broadcast_to_clients({
                "type": "analysis_update",
                "data": analysis_result.dict()
            })
            
            frame_number += 1
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)
        
        cap.release()
        app_state.is_processing = False
        
        # Notify completion
        await broadcast_to_clients({
            "type": "analysis_complete",
            "message": "Video analysis completed"
        })
        
        logger.info("Video processing completed")
    
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        app_state.is_processing = False
        await broadcast_to_clients({
            "type": "error",
            "message": str(e)
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)