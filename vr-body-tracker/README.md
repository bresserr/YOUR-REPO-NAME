# VR Body Tracker - 3D VR Video Analysis & Funscript Generator

A sophisticated application for analyzing 3D VR videos, detecting and tracking body parts, and generating funscripts for interactive devices like The Handy.

## Features

- **3D VR Video Support**: Processes side-by-side VR video formats
- **Body Part Detection**: Detects and tracks multiple body parts using TensorRT/CUDA accelerated AI models
  - Head
  - Mouth  
  - Hands (left and right)
  - Breasts
  - Pelvis
  - Genitals
- **Live Preview**: Real-time visualization with highlighted body parts
- **Manual Correction**: Pause and manually correct body part positions
- **Interaction Detection**: Tracks interactions between body parts
- **Funscript Generation**: Automatically generates funscripts based on detected movements and interactions
- **Web Interface**: Modern, responsive web GUI

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- TensorRT 8.6+
- CUDA Toolkit 11.x+
- cuDNN 8.x+

## Installation

1. Clone the repository:
```bash
cd vr-body-tracker
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install CUDA and TensorRT (if not already installed):
   - Download and install CUDA Toolkit from NVIDIA
   - Download and install TensorRT from NVIDIA Developer
   - Ensure CUDA and TensorRT paths are in your system PATH

## Usage

1. Start the application:
```bash
cd backend
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Upload a VR video:
   - Click the upload area or drag & drop a VR video file
   - Supported formats: MP4, AVI, MOV, etc.

4. Analyze the video:
   - Click "Start Analysis" to begin processing
   - The application will detect and track body parts in real-time
   - Watch the live preview with highlighted body parts

5. Manual corrections (optional):
   - Click "Correction Mode" to enable manual corrections
   - Select the body part to correct
   - Click on the video preview to set the new position
   - The correction will be applied to that frame

6. Generate funscript:
   - After analysis is complete, click "Generate Funscript"
   - Configure generation options if desired
   - Download the generated .funscript file

## How It Works

1. **Video Processing**: The application splits VR video frames into left/right views
2. **Body Detection**: Uses MediaPipe and custom models to detect body landmarks
3. **Tracking**: Tracks movement and calculates speeds for each body part
4. **Interaction Analysis**: Detects when body parts interact based on proximity
5. **Funscript Generation**: Converts interactions into device commands:
   - Distance determines stroke depth
   - Movement speed affects intensity
   - Different body parts trigger different patterns

## Funscript Patterns

The generator creates different patterns based on detected interactions:
- **Hand interactions**: Standard stroking patterns
- **Mouth interactions**: Deeper, more intense patterns
- **Pelvis interactions**: Rhythmic patterns
- **Multiple interactions**: Combined complex patterns

## Output

Generated funscripts follow the standard format:
```json
{
  "version": "1.0",
  "inverted": false,
  "range": 100,
  "actions": [
    {"at": 0, "pos": 0},
    {"at": 100, "pos": 100}
  ]
}
```

## Performance Tips

- Use a powerful NVIDIA GPU for best performance
- Lower resolution videos process faster
- Adjust detection confidence thresholds if needed
- Enable GPU memory growth in TensorFlow/PyTorch

## Troubleshooting

- **CUDA not found**: Ensure CUDA is installed and in PATH
- **TensorRT errors**: Check TensorRT installation and version compatibility
- **Slow processing**: Verify GPU is being used (check nvidia-smi)
- **WebSocket disconnection**: Check firewall settings

## Privacy & Security

- All processing is done locally on your machine
- No data is sent to external servers
- Videos and generated files are stored locally

## License

This project is for educational and personal use only.