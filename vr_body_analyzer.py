import os
import sys
import cv2
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import json
import time
from typing import List, Dict, Tuple, Optional
import mediapipe as mp
from PIL import Image
import io
import base64
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorRTBodyDetector:
    """TensorRT optimized body part detection for VR videos"""
    
    def __init__(self, model_path: str = None):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = cuda.Stream()
        
        # MediaPipe pose detection as fallback/comparison
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Body part mappings
        self.body_parts = {
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'torso': [11, 12, 23, 24],
            'left_arm': [11, 13, 15, 17, 19, 21],
            'right_arm': [12, 14, 16, 18, 20, 22],
            'left_leg': [23, 25, 27, 29, 31],
            'right_leg': [24, 26, 28, 30, 32]
        }
        
        if model_path and os.path.exists(model_path):
            self.load_tensorrt_engine(model_path)
    
    def load_tensorrt_engine(self, engine_path: str):
        """Load TensorRT engine from file"""
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.trt_logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            logger.info(f"TensorRT engine loaded successfully from {engine_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            return False
    
    def build_tensorrt_engine(self, onnx_path: str, engine_path: str):
        """Build TensorRT engine from ONNX model"""
        try:
            builder = trt.Builder(self.trt_logger)
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.trt_logger)
            
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    return False
            
            self.engine = builder.build_engine(network, config)
            
            if self.engine:
                with open(engine_path, 'wb') as f:
                    f.write(self.engine.serialize())
                self.context = self.engine.create_execution_context()
                logger.info(f"TensorRT engine built and saved to {engine_path}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to build TensorRT engine: {e}")
            return False
    
    def detect_body_parts_mediapipe(self, frame: np.ndarray) -> Dict:
        """Detect body parts using MediaPipe (CPU/GPU fallback)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        detections = {
            'landmarks': [],
            'body_parts': {},
            'confidence': 0.0,
            'segmentation_mask': None
        }
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            detections['landmarks'] = landmarks
            detections['confidence'] = np.mean([lm['visibility'] for lm in landmarks])
            
            # Group landmarks by body parts
            for part_name, indices in self.body_parts.items():
                part_landmarks = [landmarks[i] for i in indices if i < len(landmarks)]
                detections['body_parts'][part_name] = part_landmarks
            
            # Segmentation mask
            if results.segmentation_mask is not None:
                detections['segmentation_mask'] = results.segmentation_mask
        
        return detections
    
    def detect_body_parts_tensorrt(self, frame: np.ndarray) -> Dict:
        """Detect body parts using TensorRT optimized model"""
        if not self.engine or not self.context:
            logger.warning("TensorRT engine not available, falling back to MediaPipe")
            return self.detect_body_parts_mediapipe(frame)
        
        try:
            # Preprocess frame for TensorRT model
            input_shape = self.engine.get_binding_shape(0)
            preprocessed = self.preprocess_frame(frame, input_shape)
            
            # Allocate GPU memory
            d_input = cuda.mem_alloc(preprocessed.nbytes)
            output_shape = self.engine.get_binding_shape(1)
            output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize
            d_output = cuda.mem_alloc(output_size)
            
            # Copy input to GPU
            cuda.memcpy_htod_async(d_input, preprocessed, self.stream)
            
            # Run inference
            self.context.execute_async_v2([int(d_input), int(d_output)], self.stream.handle)
            
            # Copy output back to CPU
            output = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(output, d_output, self.stream)
            self.stream.synchronize()
            
            # Post-process results
            detections = self.postprocess_tensorrt_output(output, frame.shape)
            
            # Clean up GPU memory
            d_input.free()
            d_output.free()
            
            return detections
            
        except Exception as e:
            logger.error(f"TensorRT inference failed: {e}")
            return self.detect_body_parts_mediapipe(frame)
    
    def preprocess_frame(self, frame: np.ndarray, input_shape: Tuple) -> np.ndarray:
        """Preprocess frame for model input"""
        height, width = input_shape[2], input_shape[3]
        resized = cv2.resize(frame, (width, height))
        normalized = resized.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        return np.ascontiguousarray(batched)
    
    def postprocess_tensorrt_output(self, output: np.ndarray, original_shape: Tuple) -> Dict:
        """Post-process TensorRT model output"""
        # This would depend on the specific model architecture
        # For now, return a basic structure
        return {
            'landmarks': [],
            'body_parts': {},
            'confidence': 0.8,
            'segmentation_mask': None
        }

class VRVideoProcessor:
    """Process 3D VR videos for body part detection"""
    
    def __init__(self, detector: TensorRTBodyDetector):
        self.detector = detector
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def process_vr_video(self, video_path: str, output_path: str = None) -> List[Dict]:
        """Process VR video and detect body parts in each frame"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {width}x{height}, {frame_count} frames, {fps} FPS")
        
        # For VR videos, we might need to handle stereoscopic format
        # Assuming side-by-side stereo for now
        eye_width = width // 2 if width > height else width
        
        results = []
        frame_idx = 0
        
        # Setup output video writer if needed
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process left eye (assuming side-by-side)
                left_eye = frame[:, :eye_width]
                detections = self.detector.detect_body_parts_tensorrt(left_eye)
                
                # Add frame metadata
                detections['frame_idx'] = frame_idx
                detections['timestamp'] = frame_idx / fps
                detections['vr_format'] = 'side_by_side'
                detections['eye'] = 'left'
                
                results.append(detections)
                
                # Draw annotations on frame for output video
                if out_writer:
                    annotated_frame = self.draw_body_parts(frame, detections)
                    out_writer.write(annotated_frame)
                
                frame_idx += 1
                
                # Log progress
                if frame_idx % 30 == 0:
                    progress = (frame_idx / frame_count) * 100
                    logger.info(f"Processed {frame_idx}/{frame_count} frames ({progress:.1f}%)")
        
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
        
        return results
    
    def draw_body_parts(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Draw body part annotations on frame"""
        annotated = frame.copy()
        
        # Draw landmarks
        if detections['landmarks']:
            h, w = frame.shape[:2]
            for landmark in detections['landmarks']:
                x = int(landmark['x'] * w)
                y = int(landmark['y'] * h)
                confidence = landmark.get('visibility', 1.0)
                
                if confidence > 0.5:
                    cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)
        
        # Draw body part connections
        connections = [
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (24, 26), (25, 27), (26, 28)  # Legs
        ]
        
        landmarks = detections['landmarks']
        if len(landmarks) > max(max(conn) for conn in connections):
            h, w = frame.shape[:2]
            for start_idx, end_idx in connections:
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                if (start_point.get('visibility', 0) > 0.5 and 
                    end_point.get('visibility', 0) > 0.5):
                    start_pos = (int(start_point['x'] * w), int(start_point['y'] * h))
                    end_pos = (int(end_point['x'] * w), int(end_point['y'] * h))
                    cv2.line(annotated, start_pos, end_pos, (255, 0, 0), 2)
        
        return annotated

# Flask web application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vr_body_analyzer_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
detector = TensorRTBodyDetector()
processor = VRVideoProcessor(detector)
processing_queue = queue.Queue()
results_storage = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and any(file.filename.lower().endswith(ext) for ext in processor.supported_formats):
        # Save uploaded file
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = f"{int(time.time())}_{file.filename}"
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        
        # Start processing in background
        job_id = str(int(time.time()))
        processing_queue.put({
            'job_id': job_id,
            'filepath': filepath,
            'filename': file.filename
        })
        
        # Start processing thread
        thread = threading.Thread(target=process_video_worker, args=(job_id, filepath))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'Video upload successful, processing started'
        })
    
    return jsonify({'error': 'Invalid file format'}), 400

def process_video_worker(job_id: str, filepath: str):
    """Background worker for video processing"""
    try:
        socketio.emit('processing_update', {
            'job_id': job_id,
            'status': 'started',
            'message': 'Processing started...'
        })
        
        # Process video
        results = processor.process_vr_video(filepath)
        
        # Store results
        results_storage[job_id] = {
            'results': results,
            'status': 'completed',
            'processed_at': time.time()
        }
        
        # Generate summary statistics
        total_frames = len(results)
        avg_confidence = np.mean([r['confidence'] for r in results])
        detected_frames = sum(1 for r in results if r['landmarks'])
        
        socketio.emit('processing_update', {
            'job_id': job_id,
            'status': 'completed',
            'message': 'Processing completed successfully',
            'summary': {
                'total_frames': total_frames,
                'detected_frames': detected_frames,
                'average_confidence': round(avg_confidence, 3),
                'detection_rate': round(detected_frames / total_frames * 100, 1)
            }
        })
        
    except Exception as e:
        logger.error(f"Processing failed for job {job_id}: {e}")
        socketio.emit('processing_update', {
            'job_id': job_id,
            'status': 'error',
            'message': f'Processing failed: {str(e)}'
        })

@app.route('/results/<job_id>')
def get_results(job_id: str):
    """Get processing results for a job"""
    if job_id not in results_storage:
        return jsonify({'error': 'Job not found'}), 404
    
    data = results_storage[job_id]
    return jsonify(data)

@app.route('/live_demo')
def live_demo():
    """Live camera demo page"""
    return render_template('live_demo.html')

def generate_camera_frames():
    """Generate frames from camera for live demo"""
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Detect body parts
            detections = detector.detect_body_parts_tensorrt(frame)
            
            # Draw annotations
            annotated_frame = processor.draw_body_parts(frame, detections)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route for live demo"""
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA not available, using CPU mode")
    
    # Start the web application
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)