# VR Video Analysis Application

A comprehensive 3D VR video analysis application that uses AI-powered body part detection, tracking, and generates funscript files for haptic devices. The application features real-time pose estimation, interactive annotation capabilities, and automated script generation based on body part interactions.

## 🚀 Features

### Core Functionality
- **3D VR Video Support**: Side-by-side (SBS), over-under (OU), 360-degree mono/stereo formats
- **AI-Powered Detection**: Advanced body part detection using MediaPipe and TensorRT
- **Real-time Tracking**: Kalman filter-based tracking with temporal consistency
- **Interactive Annotation**: Manual correction of detection results with confidence scoring
- **Funscript Generation**: Automated haptic script generation based on body part interactions

### Supported Body Parts
- Head and facial features
- Mouth and lips
- Left and right hands
- Breasts
- Pelvis region
- Genital detection (penis/vagina)

### Technical Features
- **GPU Acceleration**: CUDA and TensorRT optimization for real-time processing
- **Web-based UI**: Modern, responsive interface with dark theme
- **WebSocket Communication**: Real-time updates and progress tracking
- **Batch Processing**: Efficient video analysis with memory optimization
- **Export Formats**: .funscript files compatible with haptic devices

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- CUDA Toolkit 11.7+
- TensorRT 8.0+
- FFmpeg

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vr-video-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional system dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

   # macOS
   brew install ffmpeg
   ```

4. **Setup CUDA (if using GPU acceleration)**
   ```bash
   # Verify CUDA installation
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Model Setup

The application requires pre-trained models for optimal performance:

1. **Download base models** (optional - will use MediaPipe defaults):
   ```bash
   mkdir -p models
   # Add your custom TensorRT models here
   ```

2. **Configure model paths** in `src/utils/config.py` if using custom models

## 🚀 Usage

### Starting the Application

1. **Run the server**
   ```bash
   python main.py
   ```

2. **Access the web interface**
   Open your browser and navigate to `http://localhost:8000`

### Basic Workflow

1. **Upload Video**
   - Drag and drop a VR video file or click to browse
   - Supported formats: MP4, AVI, MOV, MKV, WebM
   - VR formats: SBS, OU, 360°

2. **Configure Body Parts**
   - Toggle visibility and tracking for each body part
   - Adjust detection confidence in settings

3. **Start Analysis**
   - Click "Start Analysis" to begin processing
   - Monitor progress in real-time
   - View detection results in the video player

4. **Manual Annotation** (optional)
   - Enable annotation mode
   - Click on video to correct body part positions
   - Adjust confidence levels for corrections

5. **Generate Funscript**
   - Select target body part (penis/vagina)
   - Choose interaction parts (hands, mouth)
   - Adjust sensitivity settings
   - Generate and download .funscript file

### Advanced Configuration

#### Settings Panel
Access advanced settings through the settings button:

**Detection Settings:**
- Confidence threshold: 0.1-1.0
- NMS threshold: 0.1-1.0
- Maximum detections per frame

**Tracking Settings:**
- Algorithm: Kalman Filter, Particle Filter, Optical Flow
- Max lost frames before track deletion
- Smoothing factor for temporal consistency

**Funscript Settings:**
- Sampling rate: 10-200 Hz
- Smoothing window size
- Interaction distance threshold

**Performance Settings:**
- GPU device selection
- Batch size for processing
- Memory fraction allocation

## 🏗️ Architecture

### Backend Components

#### Main Application (`main.py`)
- FastAPI-based web server
- WebSocket support for real-time updates
- Session management and file handling

#### Core Modules
- **Video Processor** (`src/video_processor.py`): VR video format handling and frame extraction
- **Pose Detector** (`src/pose_detector.py`): MediaPipe and TensorRT-based body part detection
- **Body Tracker** (`src/body_tracker.py`): Kalman filter tracking with temporal consistency
- **Funscript Generator** (`src/funscript_generator.py`): Haptic script generation from motion data

#### Utilities
- **Configuration** (`src/utils/config.py`): Centralized application settings
- **CUDA Utils** (`src/utils/cuda_utils.py`): GPU acceleration and memory management

### Frontend Components

#### Web Interface (`src/ui/web_interface.py`)
- Modern HTML5/CSS3/JavaScript interface
- Real-time video player with annotation overlay
- Interactive controls and settings panels

#### JavaScript Modules
- **Main App** (`main.js`): Application orchestration and state management
- **Video Player** (`video-player.js`): Video playback controls and timeline
- **Annotation Manager** (`annotation.js`): Interactive annotation tools
- **Controls Manager** (`controls.js`): UI controls and settings
- **WebSocket Manager** (`websocket.js`): Real-time communication

## 🔧 API Reference

### REST Endpoints

#### Video Management
- `POST /upload-video` - Upload video file
- `POST /analyze-video` - Start video analysis
- `GET /session/{session_id}/status` - Get analysis progress
- `GET /session/{session_id}/frame/{frame_number}` - Get frame data

#### Annotation
- `POST /correct-body-part` - Apply manual correction
- `GET /session/{session_id}/annotations` - Get all annotations

#### Funscript Generation
- `POST /generate-funscript` - Generate haptic script
- `GET /download-funscript/{session_id}` - Download generated script

### WebSocket Events

#### Client → Server
- `session_update` - Update session parameters
- `frame_request` - Request specific frame data

#### Server → Client
- `status_update` - Analysis progress updates
- `frame_data` - Body part detection results
- `error` - Error notifications
- `analysis_complete` - Analysis completion

## 🎯 Funscript Generation

The application generates .funscript files compatible with haptic devices like The Handy. The generation process analyzes:

### Motion Analysis
- **Velocity tracking** of target body parts
- **Acceleration patterns** for intensity mapping
- **Direction changes** for rhythm detection
- **Interaction proximity** between body parts

### Script Parameters
- **Target part**: Primary body part for tracking (penis/vagina)
- **Interaction parts**: Parts that interact with target (hands, mouth)
- **Sensitivity**: Motion amplification factor (0.1-2.0)
- **Sampling rate**: Output frequency (10-200 Hz)

### Output Format
```json
{
  "version": "1.0",
  "inverted": false,
  "range": 100,
  "actions": [
    {"at": 0, "pos": 50},
    {"at": 100, "pos": 75},
    ...
  ],
  "metadata": {
    "creator": "VR Video Analyzer",
    "description": "Generated from penis interactions with hand_1, hand_2, mouth",
    "duration": 300000,
    "fps": 30.0
  }
}
```

## 🔬 Technical Details

### Performance Optimization
- **GPU Acceleration**: CUDA kernels for image processing
- **TensorRT Optimization**: Model quantization and graph optimization
- **Memory Management**: Efficient buffer allocation and cleanup
- **Batch Processing**: Multiple frame processing for throughput

### Video Format Support
- **Standard formats**: MP4, AVI, MOV, MKV, WebM
- **VR formats**: Automatic detection and processing
- **Stereo handling**: Left/right eye separation for 3D content
- **360° support**: Equirectangular projection handling

### Detection Accuracy
- **MediaPipe backbone**: Industry-standard pose estimation
- **Custom training**: Specialized models for adult content detection
- **Confidence scoring**: Reliability metrics for each detection
- **Temporal consistency**: Kalman filtering for smooth tracking

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size or memory fraction in settings
# Check GPU memory usage: nvidia-smi
```

**Video Upload Fails:**
```bash
# Check file format compatibility
# Verify file size limits (2GB default)
# Ensure sufficient disk space
```

**Poor Detection Quality:**
```bash
# Adjust confidence thresholds
# Use manual annotation for corrections
# Verify video quality and lighting
```

**WebSocket Connection Issues:**
```bash
# Check firewall settings
# Verify port 8000 accessibility
# Check browser console for errors
```

### Logging
The application uses structured logging:
```bash
# View logs in terminal
tail -f logs/application.log

# Enable debug logging
export DEBUG=true
python main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is designed for educational and research purposes. Users are responsible for ensuring compliance with applicable laws and regulations regarding adult content processing and haptic device usage.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section
- Review the GitHub issues
- Contact the development team

## 🚧 Future Enhancements

- **Multi-person detection**: Support for multiple individuals
- **Advanced VR formats**: Support for additional VR video formats
- **Machine learning improvements**: Enhanced detection models
- **Real-time processing**: Live video stream analysis
- **Mobile support**: Responsive design for mobile devices
- **Cloud deployment**: Containerized deployment options

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Compatibility**: Python 3.8+, CUDA 11.7+, TensorRT 8.0+
