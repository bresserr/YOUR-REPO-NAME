"""
3D VR Video Analysis Application
Main application file with FastAPI backend for body part detection and tracking
"""

import os
import sys
import asyncio
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import torch
import tensorrt as trt
from loguru import logger

# Import our custom modules
from src.video_processor import VRVideoProcessor
from src.pose_detector import PoseDetector
from src.body_tracker import BodyTracker
from src.funscript_generator import FunscriptGenerator
from src.ui.web_interface import WebInterface
from src.utils.config import Config
from src.utils.cuda_utils import setup_cuda_environment

# Initialize configuration
config = Config()

# Setup CUDA environment
setup_cuda_environment()

# Initialize FastAPI app
app = FastAPI(
    title="3D VR Video Analysis",
    description="AI-powered body part detection and tracking for VR videos",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_sessions: Dict[str, Dict] = {}
video_processor = None
pose_detector = None
body_tracker = None
funscript_generator = None

# Data models
class VideoAnalysisRequest(BaseModel):
    video_path: str
    analysis_type: str = "full"
    body_parts: List[str] = ["head", "mouth", "hand_1", "hand_2", "breasts", "pelvis", "genitals"]

class BodyPartCorrection(BaseModel):
    session_id: str
    frame_number: int
    body_part: str
    position: Dict[str, float]  # x, y, z coordinates

class FunscriptRequest(BaseModel):
    session_id: str
    target_body_part: str = "penis"
    interaction_parts: List[str] = ["hand_1", "hand_2", "mouth"]
    sensitivity: float = 1.0

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global video_processor, pose_detector, body_tracker, funscript_generator
    
    logger.info("Starting VR Video Analysis Application...")
    
    # Initialize components
    video_processor = VRVideoProcessor(config)
    pose_detector = PoseDetector(config)
    body_tracker = BodyTracker(config)
    funscript_generator = FunscriptGenerator(config)
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    logger.info("Application initialized successfully")

@app.get("/")
async def read_root():
    """Serve the main web interface"""
    return HTMLResponse(content=WebInterface().get_main_page())

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a VR video file for analysis"""
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = f"uploads/{session_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize video processor for this session
        session_data = {
            "session_id": session_id,
            "video_path": file_path,
            "status": "uploaded",
            "created_at": datetime.now().isoformat(),
            "current_frame": 0,
            "total_frames": 0,
            "body_parts": {},
            "corrections": []
        }
        
        active_sessions[session_id] = session_data
        
        # Get basic video info
        video_info = video_processor.get_video_info(file_path)
        session_data.update(video_info)
        
        return JSONResponse({
            "session_id": session_id,
            "status": "success",
            "message": "Video uploaded successfully",
            "video_info": video_info
        })
        
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Error uploading video: {str(e)}"
        }, status_code=500)

@app.post("/analyze-video")
async def analyze_video(request: VideoAnalysisRequest):
    """Start video analysis for body part detection"""
    try:
        session_id = request.video_path.split('/')[-1].split('_')[0]
        
        if session_id not in active_sessions:
            return JSONResponse({
                "status": "error",
                "message": "Session not found"
            }, status_code=404)
        
        session_data = active_sessions[session_id]
        session_data["status"] = "analyzing"
        
        # Start analysis in background
        asyncio.create_task(
            run_video_analysis(session_id, request.video_path, request.body_parts)
        )
        
        return JSONResponse({
            "status": "success",
            "message": "Analysis started",
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Error starting analysis: {str(e)}"
        }, status_code=500)

async def run_video_analysis(session_id: str, video_path: str, body_parts: List[str]):
    """Run video analysis in background"""
    try:
        session_data = active_sessions[session_id]
        
        # Process video frame by frame
        for frame_data in video_processor.process_video(video_path):
            # Detect poses in current frame
            poses = pose_detector.detect_poses(frame_data["frame"])
            
            # Track body parts
            tracked_parts = body_tracker.track_body_parts(
                poses, frame_data["frame_number"], body_parts
            )
            
            # Update session data
            session_data["current_frame"] = frame_data["frame_number"]
            session_data["body_parts"][frame_data["frame_number"]] = tracked_parts
            
            # Yield control to allow other operations
            await asyncio.sleep(0.001)
        
        session_data["status"] = "completed"
        logger.info(f"Analysis completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error in video analysis: {str(e)}")
        active_sessions[session_id]["status"] = "error"
        active_sessions[session_id]["error"] = str(e)

@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get current analysis status for a session"""
    if session_id not in active_sessions:
        return JSONResponse({
            "status": "error",
            "message": "Session not found"
        }, status_code=404)
    
    session_data = active_sessions[session_id]
    return JSONResponse({
        "session_id": session_id,
        "status": session_data["status"],
        "current_frame": session_data.get("current_frame", 0),
        "total_frames": session_data.get("total_frames", 0),
        "progress": session_data.get("current_frame", 0) / max(session_data.get("total_frames", 1), 1)
    })

@app.get("/session/{session_id}/frame/{frame_number}")
async def get_frame_data(session_id: str, frame_number: int):
    """Get body part data for a specific frame"""
    if session_id not in active_sessions:
        return JSONResponse({
            "status": "error",
            "message": "Session not found"
        }, status_code=404)
    
    session_data = active_sessions[session_id]
    frame_data = session_data["body_parts"].get(frame_number, {})
    
    return JSONResponse({
        "session_id": session_id,
        "frame_number": frame_number,
        "body_parts": frame_data,
        "corrections": [c for c in session_data["corrections"] if c["frame_number"] == frame_number]
    })

@app.post("/correct-body-part")
async def correct_body_part(correction: BodyPartCorrection):
    """Apply user correction to body part position"""
    try:
        if correction.session_id not in active_sessions:
            return JSONResponse({
                "status": "error",
                "message": "Session not found"
            }, status_code=404)
        
        session_data = active_sessions[correction.session_id]
        
        # Apply correction
        if correction.frame_number not in session_data["body_parts"]:
            session_data["body_parts"][correction.frame_number] = {}
        
        session_data["body_parts"][correction.frame_number][correction.body_part] = correction.position
        
        # Store correction for history
        session_data["corrections"].append(correction.dict())
        
        return JSONResponse({
            "status": "success",
            "message": "Correction applied successfully"
        })
        
    except Exception as e:
        logger.error(f"Error applying correction: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Error applying correction: {str(e)}"
        }, status_code=500)

@app.post("/generate-funscript")
async def generate_funscript(request: FunscriptRequest):
    """Generate funscript file based on body part interactions"""
    try:
        if request.session_id not in active_sessions:
            return JSONResponse({
                "status": "error",
                "message": "Session not found"
            }, status_code=404)
        
        session_data = active_sessions[request.session_id]
        
        # Generate funscript
        funscript_data = funscript_generator.generate_funscript(
            body_parts_data=session_data["body_parts"],
            target_part=request.target_body_part,
            interaction_parts=request.interaction_parts,
            sensitivity=request.sensitivity
        )
        
        # Save funscript file
        output_path = f"outputs/{request.session_id}.funscript"
        with open(output_path, 'w') as f:
            json.dump(funscript_data, f, indent=2)
        
        return JSONResponse({
            "status": "success",
            "message": "Funscript generated successfully",
            "download_url": f"/download-funscript/{request.session_id}"
        })
        
    except Exception as e:
        logger.error(f"Error generating funscript: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Error generating funscript: {str(e)}"
        }, status_code=500)

@app.get("/download-funscript/{session_id}")
async def download_funscript(session_id: str):
    """Download generated funscript file"""
    file_path = f"outputs/{session_id}.funscript"
    
    if not os.path.exists(file_path):
        return JSONResponse({
            "status": "error",
            "message": "Funscript file not found"
        }, status_code=404)
    
    return FileResponse(
        file_path,
        media_type="application/json",
        filename=f"{session_id}.funscript"
    )

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send current session status
            if session_id in active_sessions:
                session_data = active_sessions[session_id]
                await websocket.send_json({
                    "type": "status_update",
                    "data": {
                        "session_id": session_id,
                        "status": session_data["status"],
                        "current_frame": session_data.get("current_frame", 0),
                        "total_frames": session_data.get("total_frames", 0)
                    }
                })
            
            await asyncio.sleep(0.1)  # Send updates every 100ms
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

# Mount static files
app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )