#!/usr/bin/env python3

import sys
import os
import logging
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        import mediapipe as mp
        print(f"✅ MediaPipe: {mp.__version__}")
        
        import gradio as gr
        print(f"✅ Gradio: {gr.__version__}")
        
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import scipy
        print(f"✅ SciPy: {scipy.__version__}")
        
        try:
            import tensorrt as trt
            print(f"✅ TensorRT: {trt.__version__}")
        except ImportError:
            print("⚠️  TensorRT not available (optional)")
        
        try:
            import pycuda.driver as cuda
            print("✅ PyCUDA available")
        except ImportError:
            print("⚠️  PyCUDA not available (optional)")
        
        print("✅ All core imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\n🔥 Testing CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"   Compute capability: {torch.cuda.get_device_capability(0)}")
            return True
        else:
            print("⚠️  CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_video_processing():
    """Test video processing capabilities"""
    print("\n🎥 Testing video processing...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a dummy video frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test basic OpenCV operations
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print("✅ Basic video processing works")
        return True
        
    except Exception as e:
        print(f"❌ Video processing test failed: {e}")
        return False

def test_body_detection():
    """Test body part detection"""
    print("\n🧠 Testing body part detection...")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        
        # Create a dummy image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test detection (will likely find nothing in a black image)
        results = pose.process(image)
        
        pose.close()
        
        print("✅ Body part detection initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Body detection test failed: {e}")
        return False

def test_modules():
    """Test custom modules"""
    print("\n📦 Testing custom modules...")
    
    try:
        # Test if our custom modules can be imported
        from video_processor import VideoProcessor
        from body_part_detector import BodyPartDetector
        from movement_analyzer import MovementAnalyzer
        from funscript_generator import FunscriptGenerator
        from tensorrt_optimizer import TensorRTOptimizer
        from gui_components import GUIComponents
        
        print("✅ All custom modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Custom module import failed: {e}")
        return False

def test_main_application():
    """Test main application initialization"""
    print("\n🚀 Testing main application...")
    
    try:
        from vr_video_analyzer import VRVideoAnalyzer
        
        # Initialize analyzer
        analyzer = VRVideoAnalyzer()
        
        print("✅ Main application initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Main application test failed: {e}")
        return False

def test_funscript_generation():
    """Test funscript generation"""
    print("\n🎮 Testing funscript generation...")
    
    try:
        from funscript_generator import FunscriptGenerator
        
        generator = FunscriptGenerator()
        
        # Test with dummy data
        dummy_data = {
            "movement_patterns": {
                "stroke_patterns": [
                    {
                        "timestamp": 0.0,
                        "duration": 2.0,
                        "positions": [10, 50, 90, 50, 10],
                        "avg_intensity": 0.5,
                        "max_intensity": 1.0,
                        "type": "hand_left-genitals"
                    }
                ],
                "intensity_curve": [],
                "rhythm_analysis": {"pattern_type": "regular"}
            },
            "total_frames": 60,
            "duration": 2.0
        }
        
        # Test funscript generation
        success, message = generator.generate_script(dummy_data, "test_output.funscript")
        
        if success:
            print(f"✅ Funscript generation successful: {message}")
            # Clean up test file
            if os.path.exists("test_output.funscript"):
                os.remove("test_output.funscript")
            return True
        else:
            print(f"❌ Funscript generation failed: {message}")
            return False
            
    except Exception as e:
        print(f"❌ Funscript generation test failed: {e}")
        return False

def test_web_interface():
    """Test web interface components"""
    print("\n🌐 Testing web interface...")
    
    try:
        from gui_components import GUIComponents
        
        gui = GUIComponents()
        
        # Test component creation
        video_player = gui.create_video_player()
        body_selector = gui.create_body_part_selector()
        settings = gui.create_settings_panel()
        
        print("✅ Web interface components created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 VR Video Analyzer Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_cuda,
        test_video_processing,
        test_body_detection,
        test_modules,
        test_main_application,
        test_funscript_generation,
        test_web_interface
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\nYou can now run the application with:")
        print("  python3 vr_video_analyzer.py")
        return True
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\nTry running the installation script again:")
        print("  ./install_dependencies.sh")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)