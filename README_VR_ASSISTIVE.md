# VR Assistive Technology - Body Part Tracking Application

A comprehensive assistive technology application for 3D VR video analysis with advanced body part detection and tracking capabilities. This application is specifically designed to help disabled individuals by providing automated analysis and generating funscript files for compatible devices.

## Features

### 🔍 Advanced Body Part Detection
- **Multi-part tracking**: Head, mouth, hands, breasts, pelvis, and genitals
- **CUDA/TensorRT acceleration** for real-time processing
- **MediaPipe integration** for robust pose estimation
- **High-precision tracking** with confidence scoring

### 🎯 Interactive Correction System
- **Manual correction** of tracking points
- **Frame-by-frame editing** capabilities
- **Visual feedback** with color-coded body parts
- **Undo/redo** functionality for corrections

### 📊 Real-time Analytics
- **Speed tracking** of body part movements
- **Depth analysis** for interaction assessment
- **Intensity measurement** of interactions
- **Live metrics** dashboard with visual indicators

### 🎮 Funscript Generation
- **Automated funscript** creation from tracking data
- **Customizable parameters** for speed, depth, and intensity
- **The Handy device** compatible output format
- **Smooth interpolation** and optimization

### 🌐 Web-based Interface
- **Responsive design** for all devices
- **Real-time WebSocket** communication
- **Intuitive controls** with keyboard shortcuts
- **Accessibility features** for assistive technology

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for performance)
- 8GB+ RAM
- Modern web browser with WebSocket support

### System Requirements

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0

# For CUDA support (optional but recommended)
sudo apt-get install nvidia-cuda-toolkit
```

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd vr-assistive-technology
```

2. **Install dependencies**
```bash
pip install -r requirements_vr_assist.txt
```

3. **Setup CUDA/TensorRT (optional)**
```bash
# Install TensorRT (if not already installed)
pip install tensorrt
```

4. **Create necessary directories**
```bash
mkdir -p uploads output models
```

5. **Download pre-trained models**
```bash
# MediaPipe models will be downloaded automatically
# YOLO models will be downloaded on first run
```

## Usage

### Starting the Application

1. **Launch the server**
```bash
python vr_assist_app.py
```

2. **Open web browser**
Navigate to `http://localhost:8000`

### Basic Workflow

#### 1. Video Upload
- Click "Upload" and select your VR video file
- Supported formats: MP4, AVI, MOV, MKV, WebM
- The application will automatically detect VR format

#### 2. Configure Tracking
- Enable/disable body parts to track
- Adjust detection sensitivity if needed
- Set tracking parameters

#### 3. Start Analysis
- Click "Start" to begin processing
- Monitor progress in real-time
- View live tracking overlays

#### 4. Manual Corrections
- Pause at any frame
- Click on incorrect tracking points
- Drag to correct position
- Apply corrections

#### 5. Generate Funscript
- Set generation parameters:
  - **Speed Multiplier**: Adjusts response to movement speed
  - **Depth Sensitivity**: Controls penetration depth mapping
  - **Intensity**: Overall interaction intensity scaling
- Click "Generate Funscript"

#### 6. Download Results
- Download generated `.funscript` file
- Compatible with The Handy and similar devices

### Advanced Features

#### Keyboard Shortcuts
- `Ctrl+Space`: Start/Pause analysis
- `Ctrl+G`: Generate funscript
- `Ctrl+D`: Download funscript
- `Arrow keys`: Frame navigation
- `+/-`: Zoom in/out

#### Timeline Navigation
- Click anywhere on timeline to jump to frame
- Drag timeline handle for precise navigation
- Real-time preview during scrubbing

#### Zoom and Pan
- Zoom controls for detailed inspection
- Pan around zoomed video
- Reset zoom to fit view

## Technical Details

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   FastAPI App   │    │  Body Detection │
│                 │◄──►│                 │◄──►│                 │
│  • HTML/CSS/JS  │    │  • WebSocket    │    │  • MediaPipe    │
│  • Canvas       │    │  • REST API     │    │  • CUDA/TensorRT│
│  • Timeline     │    │  • File Upload  │    │  • YOLO         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▲
                                │
                ┌─────────────────▼─────────────────┐
                │        Processing Pipeline       │
                │                                  │
                │  ┌─────────────┐  ┌─────────────┐│
                │  │   Video     │  │  Tracking   ││
                │  │  Processor  │  │  Manager    ││
                │  └─────────────┘  └─────────────┘│
                │                                  │
                │  ┌─────────────┐  ┌─────────────┐│
                │  │ Funscript   │  │ Interaction ││
                │  │ Generator   │  │ Analyzer    ││
                │  └─────────────┘  └─────────────┘│
                └──────────────────────────────────┘
```

### Body Part Detection

The application uses a multi-model approach for robust detection:

1. **MediaPipe Pose**: Primary pose estimation
2. **MediaPipe Hands**: Specialized hand tracking
3. **MediaPipe Face**: Face detection and landmarks
4. **YOLO**: Additional object detection
5. **Custom Models**: Specialized intimate part detection

### Tracking Algorithm

```python
# Simplified tracking flow
for frame in video:
    # 1. Detect body parts
    detections = detector.detect_body_parts(frame)
    
    # 2. Update tracking
    tracking_manager.update_tracking(frame_number, detections)
    
    # 3. Analyze interactions
    interactions = tracking_manager.analyze_interactions(frame_number)
    
    # 4. Generate funscript actions
    if interactions:
        funscript_generator.process_interactions(interactions)
```

### Performance Optimization

- **CUDA acceleration** for neural network inference
- **TensorRT optimization** for production deployment
- **Frame skipping** for real-time processing
- **Multithreading** for parallel processing
- **Memory management** for large video files

## Configuration

### Environment Variables

```bash
# Optional configuration
export VR_ASSIST_CUDA_DEVICE=0          # CUDA device ID
export VR_ASSIST_BATCH_SIZE=4           # Processing batch size
export VR_ASSIST_CONFIDENCE_THRESHOLD=0.5  # Detection confidence
export VR_ASSIST_MAX_WORKERS=4          # Processing workers
```

### Config File

Create `config.json` for advanced settings:

```json
{
  "detection": {
    "use_tensorrt": true,
    "confidence_threshold": 0.5,
    "smooth_landmarks": true,
    "max_num_hands": 2
  },
  "tracking": {
    "interaction_distance_threshold": 50,
    "speed_smoothing_window": 5,
    "depth_analysis_window": 10
  },
  "funscript": {
    "min_interval": 100,
    "max_speed": 500,
    "smoothing_window": 5
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": false
  }
}
```

## Privacy and Ethics

This application is designed specifically for assistive technology purposes. Please ensure:

- **Consent**: Only process videos with explicit consent
- **Privacy**: Keep all data local and secure
- **Legal compliance**: Follow applicable laws and regulations
- **Ethical use**: Use only for intended assistive purposes

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size
   - Use smaller video resolution
   - Enable frame skipping

2. **Slow processing**
   - Ensure CUDA is properly installed
   - Check GPU utilization
   - Reduce video quality if needed

3. **WebSocket connection issues**
   - Check firewall settings
   - Verify port availability
   - Try different browser

4. **Model loading errors**
   - Ensure internet connection for initial download
   - Check model file permissions
   - Verify CUDA version compatibility

### Performance Tips

- **Use GPU acceleration** when available
- **Process shorter video segments** for faster results
- **Adjust detection confidence** based on video quality
- **Enable frame skipping** for real-time preview
- **Use SSD storage** for better I/O performance

## Contributing

This is an assistive technology project. Contributions should focus on:

- **Accessibility improvements**
- **Performance optimizations**
- **Bug fixes and stability**
- **Documentation updates**
- **Testing and validation**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For technical support or questions:

- Check the troubleshooting section
- Review the configuration options
- Open an issue with detailed information
- Include system specifications and error logs

## Disclaimer

This application is intended for assistive technology use only. Users are responsible for ensuring appropriate and legal use of the software. The developers are not responsible for misuse or any resulting consequences.

---

**Note**: This application processes sensitive content for assistive technology purposes. Please ensure proper data handling and privacy protection according to applicable laws and regulations.