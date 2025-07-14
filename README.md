# VR Video Analyzer

A comprehensive VR video analysis tool that detects body parts, tracks movements, and generates funscripts for interactive devices. Built with Python, TensorRT/CUDA acceleration, and a modern web interface.

## Features

- **🎥 VR Video Processing**: Support for 3D VR videos with automatic format detection
- **🧠 Body Part Detection**: Advanced detection of head, mouth, hands, breasts, pelvis, and genitals using MediaPipe
- **📊 Movement Analysis**: Real-time tracking and analysis of body part movements and interactions
- **🎮 Funscript Generation**: Automatic generation of .funscript files for compatible devices
- **⚡ TensorRT Optimization**: CUDA-accelerated inference for faster processing
- **🌐 Web Interface**: Modern, responsive web UI built with Gradio
- **✏️ Manual Correction**: Click-to-correct system for improving detection accuracy
- **📈 Real-time Visualization**: Live preview with highlighted body parts and tracking data

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for TensorRT optimization)
- 4GB+ RAM
- FFmpeg for video processing

## Installation

### Quick Install (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/vrvideoanalyzer/vr-video-analyzer.git
cd vr-video-analyzer
```

2. Run the installation script:
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

3. Activate the virtual environment:
```bash
source vr_analyzer_env/bin/activate
```

### Manual Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Install TensorRT (if using CUDA):
```bash
pip install tensorrt pycuda
```

## Usage

### Basic Usage

1. Start the application:
```bash
python vr_video_analyzer.py
```

2. Open your web browser to `http://localhost:7860`

3. Upload a VR video file and click "Load Video"

4. Navigate through frames using the slider or input field

5. Correct body part positions by:
   - Selecting a body part from the dropdown
   - Clicking on the video where it should be positioned

6. Generate a funscript:
   - Set the output path
   - Click "Generate Funscript"

### Advanced Usage

#### Command Line Options

```bash
python vr_video_analyzer.py --help
```

#### Python API

```python
from vr_video_analyzer import VRVideoAnalyzer

# Initialize analyzer
analyzer = VRVideoAnalyzer()

# Load video
success, message = analyzer.load_video("path/to/video.mp4")

# Process specific frame
detection = analyzer.process_frame(frame_id=100)

# Analyze movements
analysis = analyzer.analyze_movements()

# Generate funscript
success, message = analyzer.generate_funscript("output.funscript")
```

## Configuration

### Detection Settings

- **Confidence Threshold**: Minimum confidence for body part detection (0.1-1.0)
- **TensorRT Optimization**: Enable CUDA acceleration (requires compatible GPU)
- **Smooth Tracking**: Apply smoothing to tracked body parts

### Funscript Settings

- **Minimum Action Interval**: Minimum time between actions (50-500ms)
- **Intensity Multiplier**: Multiply detected intensity (0.1-2.0)
- **Smoothing Factor**: Amount of smoothing to apply (0.0-1.0)

## Architecture

### Core Components

1. **Video Processor** (`video_processor.py`): Handles video loading and frame extraction
2. **Body Part Detector** (`body_part_detector.py`): MediaPipe-based detection system
3. **Movement Analyzer** (`movement_analyzer.py`): Tracks movements and interactions
4. **Funscript Generator** (`funscript_generator.py`): Converts analysis to funscript format
5. **TensorRT Optimizer** (`tensorrt_optimizer.py`): CUDA acceleration and optimization
6. **GUI Components** (`gui_components.py`): Web interface components

### Data Flow

```
Video Input → Frame Extraction → Body Part Detection → Movement Analysis → Funscript Generation
                                        ↓
                                Manual Corrections ← User Interface
```

## Performance Optimization

### TensorRT Acceleration

The application supports TensorRT optimization for faster inference:

```python
# Enable TensorRT optimization
analyzer.tensorrt_optimizer.optimize_model(
    model=detection_model,
    input_shapes={"input": (1, 3, 640, 480)},
    model_name="body_detector"
)
```

### Memory Management

- Frame caching with configurable size limits
- Automatic GPU memory management
- Background processing for smooth UI experience

## File Formats

### Supported Video Formats

- MP4, AVI, MOV, MKV
- VR 360° videos (2:1 aspect ratio)
- Standard videos
- Various codecs (H.264, H.265, etc.)

### Output Formats

- **Funscript** (.funscript): Standard format for interactive devices
- **CSV** (.csv): Comma-separated values for analysis
- **JSON** (.json): Structured data format

## API Reference

### VRVideoAnalyzer Class

#### Methods

- `load_video(path)`: Load video file
- `process_frame(frame_id)`: Process single frame
- `get_annotated_frame(frame_id)`: Get frame with annotations
- `correct_body_part(frame_id, part, x, y)`: Manual correction
- `analyze_movements()`: Analyze movement patterns
- `generate_funscript(output_path)`: Generate funscript file

### Body Part Detection

#### Detected Body Parts

- **Head**: Face center point
- **Mouth**: Mouth center from face mesh
- **Hands**: Left and right hand positions
- **Breasts**: Estimated chest area
- **Pelvis**: Hip midpoint
- **Genitals**: Estimated genital area

#### Detection Confidence

Each detection includes:
- Position coordinates (x, y, z)
- Confidence score (0.0-1.0)
- Frame ID and timestamp

## Troubleshooting

### Common Issues

1. **CUDA Not Available**
   - Install CUDA toolkit
   - Verify GPU compatibility
   - Check PyTorch CUDA installation

2. **Video Loading Failed**
   - Check file format support
   - Verify file permissions
   - Install required codecs

3. **Memory Issues**
   - Reduce frame cache size
   - Lower video resolution
   - Use smaller batch sizes

4. **Performance Issues**
   - Enable TensorRT optimization
   - Reduce confidence threshold
   - Process fewer frames per second

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black .

# Type checking
mypy .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for body part detection
- [TensorRT](https://developer.nvidia.com/tensorrt) for GPU acceleration
- [Gradio](https://gradio.app/) for the web interface
- [OpenCV](https://opencv.org/) for video processing

## Support

- 📧 Email: support@vrvideoanalyzer.com
- 🐛 Issues: [GitHub Issues](https://github.com/vrvideoanalyzer/vr-video-analyzer/issues)
- 📖 Documentation: [Read the Docs](https://vrvideoanalyzer.readthedocs.io/)
- 💬 Discord: [Join our community](https://discord.gg/vrvideoanalyzer)

## Changelog

### Version 1.0.0
- Initial release
- Basic body part detection
- Movement analysis
- Funscript generation
- Web interface
- TensorRT optimization

---

**Note**: This application is designed for adult content analysis and should be used responsibly and in accordance with local laws and regulations.
