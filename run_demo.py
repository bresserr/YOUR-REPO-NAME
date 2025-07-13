#!/usr/bin/env python3
"""
Quick demo runner for VR Body Analyzer
This script provides an easy way to test the application with sample data.
"""

import os
import sys
import time
import subprocess
import threading
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_modules = [
        'flask', 'cv2', 'numpy', 'torch', 'mediapipe'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def create_sample_video():
    """Create a sample VR video for testing if none exists"""
    import cv2
    import numpy as np
    
    sample_dir = Path("samples")
    sample_dir.mkdir(exist_ok=True)
    
    sample_path = sample_dir / "sample_vr_video.mp4"
    
    if sample_path.exists():
        print(f"Sample video already exists: {sample_path}")
        return str(sample_path)
    
    print("Creating sample VR video...")
    
    # Create a simple test video (side-by-side format)
    width, height = 1920, 1080
    fps = 30
    duration = 5  # seconds
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(sample_path), fourcc, fps, (width, height))
    
    for frame_num in range(fps * duration):
        # Create a frame with moving objects
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background
        frame[:] = (20, 30, 40)
        
        # Left eye view
        left_center_x = int(width * 0.25 + 100 * np.sin(frame_num * 0.1))
        left_center_y = int(height * 0.5 + 50 * np.cos(frame_num * 0.1))
        cv2.circle(frame, (left_center_x, left_center_y), 50, (0, 255, 0), -1)
        
        # Right eye view (slightly offset)
        right_center_x = int(width * 0.75 + 100 * np.sin(frame_num * 0.1))
        right_center_y = int(height * 0.5 + 50 * np.cos(frame_num * 0.1))
        cv2.circle(frame, (right_center_x, right_center_y), 50, (0, 255, 0), -1)
        
        # Add some text
        cv2.putText(frame, f"Frame {frame_num}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Left Eye", (int(width * 0.125), height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Right Eye", (int(width * 0.625), height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Divider line
        cv2.line(frame, (width // 2, 0), (width // 2, height), (100, 100, 100), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Sample video created: {sample_path}")
    return str(sample_path)

def start_application():
    """Start the VR Body Analyzer application"""
    print("Starting VR Body Analyzer...")
    
    # Import and run the application
    try:
        from vr_body_analyzer import app, socketio
        
        def run_app():
            socketio.run(app, host='0.0.0.0', port=5000, debug=False)
        
        app_thread = threading.Thread(target=run_app, daemon=True)
        app_thread.start()
        
        return True
        
    except ImportError as e:
        print(f"Failed to import application: {e}")
        return False

def open_browser():
    """Open web browser to the application"""
    time.sleep(2)  # Wait for server to start
    
    urls = [
        "http://localhost:5000",
        "http://127.0.0.1:5000"
    ]
    
    for url in urls:
        try:
            webbrowser.open(url)
            print(f"✓ Opened browser: {url}")
            return True
        except Exception as e:
            print(f"Failed to open {url}: {e}")
    
    return False

def print_demo_instructions():
    """Print instructions for using the demo"""
    print("\n" + "="*60)
    print("VR Body Analyzer Demo Instructions")
    print("="*60)
    print()
    print("1. WEB INTERFACE:")
    print("   - Main page: Upload and analyze VR videos")
    print("   - Live Demo: Real-time camera body part detection")
    print()
    print("2. TESTING WITH SAMPLE VIDEO:")
    print("   - A sample VR video has been created in ./samples/")
    print("   - Upload this video to test the analysis pipeline")
    print()
    print("3. LIVE DEMO:")
    print("   - Click 'Live Demo' in the navigation")
    print("   - Allow camera access when prompted")
    print("   - Click 'Start Detection' to begin analysis")
    print()
    print("4. CONTROLS:")
    print("   - Adjust confidence threshold (0.1 - 0.9)")
    print("   - Switch between TensorRT/MediaPipe modes")
    print("   - Toggle landmarks and connections display")
    print("   - Capture frames and export data")
    print()
    print("5. SUPPORTED FORMATS:")
    print("   - Video: MP4, AVI, MOV, MKV, WebM")
    print("   - VR: Side-by-side, top-bottom, monoscopic")
    print()
    print("6. STOPPING THE DEMO:")
    print("   - Press Ctrl+C in this terminal")
    print("   - Or close the browser and terminal")
    print()
    print("="*60)

def main():
    """Main demo function"""
    print("VR Body Analyzer - Quick Demo")
    print("="*40)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        return False
    
    print("✓ All dependencies available")
    
    # Create sample video
    print("\n2. Setting up sample data...")
    sample_video = create_sample_video()
    
    # Start application
    print("\n3. Starting application...")
    if not start_application():
        print("✗ Failed to start application")
        return False
    
    print("✓ Application started successfully")
    
    # Open browser
    print("\n4. Opening web browser...")
    browser_opened = open_browser()
    
    if not browser_opened:
        print("Please manually open: http://localhost:5000")
    
    # Print instructions
    print_demo_instructions()
    
    # Keep the demo running
    try:
        print("\nDemo is running... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nDemo stopped. Thank you for trying VR Body Analyzer!")
        return True

if __name__ == "__main__":
    main()