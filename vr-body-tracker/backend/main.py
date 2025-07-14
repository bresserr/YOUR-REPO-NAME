from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import base64

from body_detector import BodyPartDetector
from funscript_generator import FunscriptGenerator

app = FastAPI(title="VR Body Tracker")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class VideoProcessor:
    def __init__(self):
        self.detector = BodyPartDetector()
        self.generator = FunscriptGenerator()
        self.current_video = None
        self.video_capture = None
        self.is_processing = False
        self.is_paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0
        self.position_history = []
        self.movement_history = []
        self.interaction_history = []
        self.manual_corrections = {}
        self.websocket_clients = []
        
processor = VideoProcessor()

# Mount static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('../frontend/index.html')

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a VR video for processing"""
    try:
        # Save uploaded file
        video_id = str(uuid.uuid4())
        video_path = f"../output/{video_id}_{file.filename}"
        
        os.makedirs("../output", exist_ok=True)
        
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Open video
        processor.video_capture = cv2.VideoCapture(video_path)
        processor.current_video = video_path
        processor.total_frames = int(processor.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        processor.fps = processor.video_capture.get(cv2.CAP_PROP_FPS)
        processor.generator.fps = processor.fps
        
        return JSONResponse({
            "video_id": video_id,
            "filename": file.filename,
            "total_frames": processor.total_frames,
            "fps": processor.fps,
            "duration": processor.total_frames / processor.fps
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time video processing and preview"""
    await websocket.accept()
    processor.websocket_clients.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "start_processing":
                await process_video(websocket)
            
            elif data["type"] == "pause":
                processor.is_paused = True
            
            elif data["type"] == "resume":
                processor.is_paused = False
            
            elif data["type"] == "seek":
                frame_number = data["frame"]
                processor.current_frame = frame_number
                processor.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            elif data["type"] == "correct_position":
                # Manual correction of body part position
                frame = data["frame"]
                body_part = data["body_part"]
                position = (data["x"], data["y"])
                
                if frame not in processor.manual_corrections:
                    processor.manual_corrections[frame] = {}
                processor.manual_corrections[frame][body_part] = position
                
                await websocket.send_json({
                    "type": "correction_saved",
                    "frame": frame,
                    "body_part": body_part
                })
            
            elif data["type"] == "generate_funscript":
                await generate_funscript(websocket)
    
    except WebSocketDisconnect:
        processor.websocket_clients.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in processor.websocket_clients:
            processor.websocket_clients.remove(websocket)

async def process_video(websocket: WebSocket):
    """Process video frames and send results via WebSocket"""
    if not processor.video_capture:
        await websocket.send_json({"type": "error", "message": "No video loaded"})
        return
    
    processor.is_processing = True
    processor.position_history = []
    processor.movement_history = []
    processor.interaction_history = []
    
    while processor.is_processing and processor.current_frame < processor.total_frames:
        if processor.is_paused:
            await asyncio.sleep(0.1)
            continue
        
        ret, frame = processor.video_capture.read()
        if not ret:
            break
        
        # Detect body parts
        detected_parts = processor.detector.detect_body_parts(frame, is_vr=True)
        
        # Apply manual corrections if any
        if processor.current_frame in processor.manual_corrections:
            corrections = processor.manual_corrections[processor.current_frame]
            for part, pos in corrections.items():
                detected_parts[part] = pos
        
        # Track movement
        processor.position_history.append(detected_parts)
        movements = processor.detector.track_movement(processor.position_history[-10:])
        processor.movement_history.append(movements)
        
        # Detect interactions
        interactions = processor.detector.detect_interactions(detected_parts)
        processor.interaction_history.append(interactions)
        
        # Draw annotations on frame
        annotated_frame = draw_annotations(frame, detected_parts, interactions)
        
        # Convert frame to base64 for sending via WebSocket
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send frame data
        await websocket.send_json({
            "type": "frame_update",
            "frame_number": processor.current_frame,
            "frame_data": frame_base64,
            "detected_parts": {k: list(v) if v else None for k, v in detected_parts.items()},
            "movements": movements,
            "interactions": [list(i) for i in interactions],
            "progress": processor.current_frame / processor.total_frames
        })
        
        processor.current_frame += 1
        
        # Small delay to prevent overwhelming the client
        await asyncio.sleep(0.01)
    
    processor.is_processing = False
    await websocket.send_json({"type": "processing_complete"})

def draw_annotations(frame: np.ndarray, detected_parts: Dict[str, Tuple[int, int]], 
                    interactions: List[Tuple[str, str, float]]) -> np.ndarray:
    """Draw body part annotations on the frame"""
    annotated = frame.copy()
    height, width = frame.shape[:2]
    
    # For VR, we'll annotate the left view
    left_width = width // 2
    
    # Colors for different body parts
    colors = {
        'head': (255, 0, 0),      # Red
        'mouth': (255, 100, 100),  # Light red
        'hand_left': (0, 255, 0),  # Green
        'hand_right': (0, 255, 0), # Green
        'breasts': (255, 0, 255),  # Magenta
        'pelvis': (0, 0, 255),     # Blue
        'genitals': (255, 255, 0)  # Yellow
    }
    
    # Draw body parts
    for part, pos in detected_parts.items():
        if pos:
            color = colors.get(part, (255, 255, 255))
            cv2.circle(annotated, pos, 8, color, -1)
            cv2.putText(annotated, part, (pos[0] + 10, pos[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw interaction lines
    for interaction in interactions:
        if len(interaction) >= 3:
            part1, part2, distance = interaction
            if part1 in detected_parts and part2 in detected_parts:
                pos1 = detected_parts[part1]
                pos2 = detected_parts[part2]
                if pos1 and pos2:
                    cv2.line(annotated, pos1, pos2, (0, 255, 255), 2)
                    mid_point = ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2)
                    cv2.putText(annotated, f"{distance:.0f}px", mid_point,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    return annotated

async def generate_funscript(websocket: WebSocket):
    """Generate funscript based on analyzed data"""
    try:
        # Generate funscript from interaction history
        actions = processor.generator.analyze_interactions(
            processor.interaction_history,
            processor.movement_history
        )
        
        # Add metadata
        metadata = {
            "creator": "VR Body Tracker",
            "created": datetime.now().isoformat(),
            "video": os.path.basename(processor.current_video) if processor.current_video else "unknown",
            "duration": int((processor.total_frames / processor.fps) * 1000),
            "fps": processor.fps
        }
        
        # Save funscript
        output_filename = f"../output/generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.funscript"
        processor.generator.export_funscript(output_filename, metadata)
        
        await websocket.send_json({
            "type": "funscript_generated",
            "filename": output_filename,
            "actions_count": len(actions)
        })
        
        return FileResponse(output_filename, filename=os.path.basename(output_filename))
    
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to generate funscript: {str(e)}"
        })

@app.get("/download_funscript/{filename}")
async def download_funscript(filename: str):
    """Download generated funscript"""
    file_path = f"../output/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)