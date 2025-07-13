"""
Web Interface Module for VR Video Analysis Application
Generates HTML/CSS/JavaScript for the frontend user interface
"""

import os
from typing import Dict, Any, List
from pathlib import Path


class WebInterface:
    """
    Generates web interface components for the VR video analysis application
    """
    
    def __init__(self):
        self.base_template_path = Path(__file__).parent / "templates"
        self.static_path = Path(__file__).parent / "static"
        
        # Ensure directories exist
        self.base_template_path.mkdir(exist_ok=True)
        self.static_path.mkdir(exist_ok=True)
        (self.static_path / "css").mkdir(exist_ok=True)
        (self.static_path / "js").mkdir(exist_ok=True)
        (self.static_path / "images").mkdir(exist_ok=True)
        
        # Create static files
        self._create_static_files()
    
    def get_main_page(self) -> str:
        """Generate the main application page"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VR Video Analysis - 3D Body Part Tracker</title>
    <link rel="stylesheet" href="/static/css/main.css">
    <link rel="stylesheet" href="/static/css/video-player.css">
    <link rel="stylesheet" href="/static/css/annotation.css">
    <link rel="stylesheet" href="/static/css/controls.css">
    <link rel="icon" href="/static/images/favicon.ico" type="image/x-icon">
</head>
<body>
    {self._get_header()}
    
    <div class="app-container">
        {self._get_sidebar()}
        {self._get_main_content()}
        {self._get_annotation_panel()}
    </div>
    
    {self._get_modals()}
    {self._get_footer()}
    
    <script src="/static/js/websocket.js"></script>
    <script src="/static/js/video-player.js"></script>
    <script src="/static/js/annotation.js"></script>
    <script src="/static/js/controls.js"></script>
    <script src="/static/js/main.js"></script>
</body>
</html>
        """
    
    def _get_header(self) -> str:
        """Generate the header component"""
        return """
        <header class="app-header">
            <div class="header-content">
                <div class="logo">
                    <h1>VR Video Analyzer</h1>
                    <span class="tagline">3D Body Part Detection & Tracking</span>
                </div>
                <div class="header-controls">
                    <button id="settings-btn" class="btn btn-secondary">
                        <i class="icon-settings"></i> Settings
                    </button>
                    <button id="help-btn" class="btn btn-secondary">
                        <i class="icon-help"></i> Help
                    </button>
                </div>
            </div>
        </header>
        """
    
    def _get_sidebar(self) -> str:
        """Generate the sidebar component"""
        return """
        <aside class="sidebar">
            <div class="sidebar-section">
                <h3>Project</h3>
                <div class="project-info">
                    <div class="project-name">No project loaded</div>
                    <div class="project-status">Ready</div>
                </div>
            </div>
            
            <div class="sidebar-section">
                <h3>Video Upload</h3>
                <div class="upload-area" id="upload-area">
                    <div class="upload-content">
                        <i class="icon-upload"></i>
                        <p>Drop VR video file here or click to browse</p>
                        <input type="file" id="video-upload" accept="video/*" style="display: none;">
                    </div>
                </div>
                <div class="upload-progress" id="upload-progress" style="display: none;">
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                    <div class="progress-text">Uploading...</div>
                </div>
            </div>
            
            <div class="sidebar-section">
                <h3>Body Parts</h3>
                <div class="body-parts-list">
                    <div class="body-part-item" data-part="head">
                        <div class="part-color" style="background-color: #ff6b6b;"></div>
                        <span class="part-name">Head</span>
                        <div class="part-controls">
                            <button class="btn-toggle" data-action="toggle-visibility">👁</button>
                            <button class="btn-toggle" data-action="toggle-tracking">🎯</button>
                        </div>
                    </div>
                    <div class="body-part-item" data-part="mouth">
                        <div class="part-color" style="background-color: #4ecdc4;"></div>
                        <span class="part-name">Mouth</span>
                        <div class="part-controls">
                            <button class="btn-toggle" data-action="toggle-visibility">👁</button>
                            <button class="btn-toggle" data-action="toggle-tracking">🎯</button>
                        </div>
                    </div>
                    <div class="body-part-item" data-part="hand_1">
                        <div class="part-color" style="background-color: #45b7d1;"></div>
                        <span class="part-name">Hand 1</span>
                        <div class="part-controls">
                            <button class="btn-toggle" data-action="toggle-visibility">👁</button>
                            <button class="btn-toggle" data-action="toggle-tracking">🎯</button>
                        </div>
                    </div>
                    <div class="body-part-item" data-part="hand_2">
                        <div class="part-color" style="background-color: #96ceb4;"></div>
                        <span class="part-name">Hand 2</span>
                        <div class="part-controls">
                            <button class="btn-toggle" data-action="toggle-visibility">👁</button>
                            <button class="btn-toggle" data-action="toggle-tracking">🎯</button>
                        </div>
                    </div>
                    <div class="body-part-item" data-part="breasts">
                        <div class="part-color" style="background-color: #ffeaa7;"></div>
                        <span class="part-name">Breasts</span>
                        <div class="part-controls">
                            <button class="btn-toggle" data-action="toggle-visibility">👁</button>
                            <button class="btn-toggle" data-action="toggle-tracking">🎯</button>
                        </div>
                    </div>
                    <div class="body-part-item" data-part="pelvis">
                        <div class="part-color" style="background-color: #dda0dd;"></div>
                        <span class="part-name">Pelvis</span>
                        <div class="part-controls">
                            <button class="btn-toggle" data-action="toggle-visibility">👁</button>
                            <button class="btn-toggle" data-action="toggle-tracking">🎯</button>
                        </div>
                    </div>
                    <div class="body-part-item" data-part="genitals">
                        <div class="part-color" style="background-color: #fd79a8;"></div>
                        <span class="part-name">Genitals</span>
                        <div class="part-controls">
                            <button class="btn-toggle" data-action="toggle-visibility">👁</button>
                            <button class="btn-toggle" data-action="toggle-tracking">🎯</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="sidebar-section">
                <h3>Analysis</h3>
                <button id="start-analysis" class="btn btn-primary" disabled>
                    <i class="icon-play"></i> Start Analysis
                </button>
                <div class="analysis-progress" id="analysis-progress" style="display: none;">
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                    <div class="progress-text">Analyzing...</div>
                </div>
            </div>
            
            <div class="sidebar-section">
                <h3>Funscript</h3>
                <div class="funscript-settings">
                    <div class="setting-group">
                        <label>Target Part:</label>
                        <select id="target-part">
                            <option value="penis">Penis</option>
                            <option value="vagina">Vagina</option>
                            <option value="genitals">Genitals</option>
                        </select>
                    </div>
                    <div class="setting-group">
                        <label>Interaction Parts:</label>
                        <div class="checkbox-group">
                            <label><input type="checkbox" value="hand_1" checked> Hand 1</label>
                            <label><input type="checkbox" value="hand_2" checked> Hand 2</label>
                            <label><input type="checkbox" value="mouth" checked> Mouth</label>
                        </div>
                    </div>
                    <div class="setting-group">
                        <label>Sensitivity:</label>
                        <input type="range" id="sensitivity-slider" min="0.1" max="2.0" step="0.1" value="1.0">
                        <span id="sensitivity-value">1.0</span>
                    </div>
                </div>
                <button id="generate-funscript" class="btn btn-success" disabled>
                    <i class="icon-download"></i> Generate Funscript
                </button>
            </div>
        </aside>
        """
    
    def _get_main_content(self) -> str:
        """Generate the main content area"""
        return """
        <main class="main-content">
            <div class="video-container">
                <div class="video-wrapper">
                    <video id="main-video" class="video-player" controls>
                        <source src="" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <canvas id="annotation-canvas" class="annotation-overlay"></canvas>
                </div>
                
                <div class="video-controls">
                    <div class="control-group">
                        <button id="play-pause" class="btn btn-primary">
                            <i class="icon-play"></i>
                        </button>
                        <button id="step-backward" class="btn btn-secondary">
                            <i class="icon-step-backward"></i>
                        </button>
                        <button id="step-forward" class="btn btn-secondary">
                            <i class="icon-step-forward"></i>
                        </button>
                    </div>
                    
                    <div class="timeline-container">
                        <input type="range" id="timeline-slider" min="0" max="100" value="0">
                        <div class="timeline-markers" id="timeline-markers"></div>
                    </div>
                    
                    <div class="time-display">
                        <span id="current-time">00:00</span>
                        <span>/</span>
                        <span id="total-time">00:00</span>
                    </div>
                    
                    <div class="control-group">
                        <button id="speed-control" class="btn btn-secondary">1x</button>
                        <button id="annotation-mode" class="btn btn-secondary">
                            <i class="icon-edit"></i>
                        </button>
                        <button id="fullscreen" class="btn btn-secondary">
                            <i class="icon-fullscreen"></i>
                        </button>
                    </div>
                </div>
            </div>
            
            <div class="status-bar">
                <div class="status-left">
                    <span id="connection-status" class="status-indicator">
                        <i class="icon-wifi"></i> Connected
                    </span>
                    <span id="processing-status" class="status-indicator">
                        <i class="icon-process"></i> Ready
                    </span>
                </div>
                <div class="status-right">
                    <span id="fps-counter">FPS: 0</span>
                    <span id="frame-counter">Frame: 0/0</span>
                </div>
            </div>
        </main>
        """
    
    def _get_annotation_panel(self) -> str:
        """Generate the annotation panel"""
        return """
        <div class="annotation-panel" id="annotation-panel">
            <div class="panel-header">
                <h3>Manual Annotation</h3>
                <button id="close-annotation" class="btn btn-secondary">×</button>
            </div>
            
            <div class="panel-content">
                <div class="annotation-instructions">
                    <p>Click on the video to mark body part positions. Use the tools below to refine the tracking.</p>
                </div>
                
                <div class="annotation-tools">
                    <div class="tool-group">
                        <label>Selected Part:</label>
                        <select id="selected-part">
                            <option value="head">Head</option>
                            <option value="mouth">Mouth</option>
                            <option value="hand_1">Hand 1</option>
                            <option value="hand_2">Hand 2</option>
                            <option value="breasts">Breasts</option>
                            <option value="pelvis">Pelvis</option>
                            <option value="genitals">Genitals</option>
                        </select>
                    </div>
                    
                    <div class="tool-group">
                        <label>Marker Size:</label>
                        <input type="range" id="marker-size" min="5" max="20" value="10">
                    </div>
                    
                    <div class="tool-group">
                        <label>Confidence:</label>
                        <input type="range" id="confidence-slider" min="0.1" max="1.0" step="0.1" value="0.8">
                        <span id="confidence-value">0.8</span>
                    </div>
                </div>
                
                <div class="annotation-actions">
                    <button id="save-annotation" class="btn btn-primary">
                        <i class="icon-save"></i> Save
                    </button>
                    <button id="delete-annotation" class="btn btn-danger">
                        <i class="icon-delete"></i> Delete
                    </button>
                    <button id="clear-frame" class="btn btn-warning">
                        <i class="icon-clear"></i> Clear Frame
                    </button>
                </div>
                
                <div class="annotation-history">
                    <h4>Recent Annotations</h4>
                    <div id="annotation-list" class="annotation-list">
                        <!-- Annotations will be populated here -->
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _get_modals(self) -> str:
        """Generate modal dialogs"""
        return """
        <!-- Settings Modal -->
        <div id="settings-modal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Settings</h2>
                    <button class="close-modal">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="settings-tabs">
                        <div class="tab-buttons">
                            <button class="tab-button active" data-tab="detection">Detection</button>
                            <button class="tab-button" data-tab="tracking">Tracking</button>
                            <button class="tab-button" data-tab="funscript">Funscript</button>
                            <button class="tab-button" data-tab="performance">Performance</button>
                        </div>
                        
                        <div class="tab-content active" data-tab="detection">
                            <div class="setting-group">
                                <label>Detection Confidence:</label>
                                <input type="range" id="detection-confidence" min="0.1" max="1.0" step="0.1" value="0.5">
                                <span class="setting-value">0.5</span>
                            </div>
                            <div class="setting-group">
                                <label>NMS Threshold:</label>
                                <input type="range" id="nms-threshold" min="0.1" max="1.0" step="0.1" value="0.4">
                                <span class="setting-value">0.4</span>
                            </div>
                            <div class="setting-group">
                                <label>Max Detections:</label>
                                <input type="number" id="max-detections" min="1" max="20" value="10">
                            </div>
                        </div>
                        
                        <div class="tab-content" data-tab="tracking">
                            <div class="setting-group">
                                <label>Tracking Algorithm:</label>
                                <select id="tracking-algorithm">
                                    <option value="kalman">Kalman Filter</option>
                                    <option value="particle">Particle Filter</option>
                                    <option value="optical_flow">Optical Flow</option>
                                </select>
                            </div>
                            <div class="setting-group">
                                <label>Max Lost Frames:</label>
                                <input type="number" id="max-lost-frames" min="1" max="30" value="10">
                            </div>
                            <div class="setting-group">
                                <label>Smoothing Factor:</label>
                                <input type="range" id="smoothing-factor" min="0.1" max="1.0" step="0.1" value="0.3">
                                <span class="setting-value">0.3</span>
                            </div>
                        </div>
                        
                        <div class="tab-content" data-tab="funscript">
                            <div class="setting-group">
                                <label>Sampling Rate (Hz):</label>
                                <input type="number" id="sampling-rate" min="10" max="200" value="100">
                            </div>
                            <div class="setting-group">
                                <label>Smoothing Window:</label>
                                <input type="number" id="smoothing-window" min="1" max="20" value="5">
                            </div>
                            <div class="setting-group">
                                <label>Interaction Distance:</label>
                                <input type="range" id="interaction-distance" min="0.01" max="0.2" step="0.01" value="0.05">
                                <span class="setting-value">0.05m</span>
                            </div>
                        </div>
                        
                        <div class="tab-content" data-tab="performance">
                            <div class="setting-group">
                                <label>GPU Device:</label>
                                <select id="gpu-device">
                                    <option value="0">CUDA:0</option>
                                    <option value="cpu">CPU</option>
                                </select>
                            </div>
                            <div class="setting-group">
                                <label>Batch Size:</label>
                                <input type="number" id="batch-size" min="1" max="16" value="4">
                            </div>
                            <div class="setting-group">
                                <label>Memory Fraction:</label>
                                <input type="range" id="memory-fraction" min="0.1" max="1.0" step="0.1" value="0.8">
                                <span class="setting-value">0.8</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" id="cancel-settings">Cancel</button>
                    <button class="btn btn-primary" id="save-settings">Save Settings</button>
                </div>
            </div>
        </div>
        
        <!-- Help Modal -->
        <div id="help-modal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Help & Documentation</h2>
                    <button class="close-modal">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="help-content">
                        <h3>Getting Started</h3>
                        <ol>
                            <li>Upload a VR video file using the upload area</li>
                            <li>Wait for the video to load and process</li>
                            <li>Click "Start Analysis" to begin body part detection</li>
                            <li>Use the annotation tools to correct any mistakes</li>
                            <li>Generate funscript file for your haptic device</li>
                        </ol>
                        
                        <h3>Body Part Detection</h3>
                        <p>The system automatically detects various body parts including head, mouth, hands, breasts, pelvis, and genitals. You can toggle visibility and tracking for each part.</p>
                        
                        <h3>Manual Annotation</h3>
                        <p>Click the annotation mode button to manually correct body part positions. Click on the video to place markers and adjust their confidence levels.</p>
                        
                        <h3>Funscript Generation</h3>
                        <p>Select a target body part and interaction parts, then adjust sensitivity to generate haptic control scripts for compatible devices.</p>
                        
                        <h3>Supported Formats</h3>
                        <ul>
                            <li>MP4, AVI, MOV, MKV, WebM</li>
                            <li>Side-by-side (SBS) stereo</li>
                            <li>Over-under (OU) stereo</li>
                            <li>360-degree mono and stereo</li>
                        </ul>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-primary" id="close-help">Close</button>
                </div>
            </div>
        </div>
        """
    
    def _get_footer(self) -> str:
        """Generate the footer component"""
        return """
        <footer class="app-footer">
            <div class="footer-content">
                <span>&copy; 2024 VR Video Analyzer. Advanced 3D Body Part Detection & Tracking.</span>
                <div class="footer-links">
                    <a href="#" id="about-link">About</a>
                    <a href="#" id="privacy-link">Privacy</a>
                    <a href="#" id="support-link">Support</a>
                </div>
            </div>
        </footer>
        """
    
    def _create_static_files(self):
        """Create static CSS and JavaScript files"""
        # Create main CSS file
        main_css = """
        /* Main CSS for VR Video Analyzer */
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --background-color: #1a1a1a;
            --surface-color: #2a2a2a;
            --text-color: #ecf0f1;
            --text-muted: #95a5a6;
            --border-color: #34495e;
            --shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
            overflow-x: hidden;
        }
        
        .app-header {
            background-color: var(--surface-color);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            box-shadow: var(--shadow);
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo h1 {
            color: var(--secondary-color);
            font-size: 1.8rem;
            margin-bottom: 0.2rem;
        }
        
        .tagline {
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        
        .app-container {
            display: flex;
            min-height: calc(100vh - 120px);
        }
        
        .sidebar {
            width: 320px;
            background-color: var(--surface-color);
            border-right: 1px solid var(--border-color);
            padding: 1rem;
            overflow-y: auto;
        }
        
        .sidebar-section {
            margin-bottom: 2rem;
        }
        
        .sidebar-section h3 {
            color: var(--secondary-color);
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }
        
        .main-content {
            flex: 1;
            padding: 1rem;
            display: flex;
            flex-direction: column;
        }
        
        .video-container {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .video-wrapper {
            position: relative;
            flex: 1;
            background-color: #000;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }
        
        .video-player {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        .annotation-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 10;
        }
        
        .annotation-overlay.active {
            pointer-events: all;
            cursor: crosshair;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn-primary {
            background-color: var(--secondary-color);
            color: white;
        }
        
        .btn-primary:hover {
            background-color: #2980b9;
        }
        
        .btn-secondary {
            background-color: var(--border-color);
            color: var(--text-color);
        }
        
        .btn-secondary:hover {
            background-color: #4a5f7a;
        }
        
        .btn-success {
            background-color: var(--success-color);
            color: white;
        }
        
        .btn-success:hover {
            background-color: #229954;
        }
        
        .btn-warning {
            background-color: var(--warning-color);
            color: white;
        }
        
        .btn-danger {
            background-color: var(--danger-color);
            color: white;
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .upload-area {
            border: 2px dashed var(--border-color);
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: var(--secondary-color);
            background-color: rgba(52, 152, 219, 0.1);
        }
        
        .upload-area.dragover {
            border-color: var(--secondary-color);
            background-color: rgba(52, 152, 219, 0.2);
        }
        
        .body-parts-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .body-part-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            transition: all 0.3s ease;
        }
        
        .body-part-item:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        .part-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        .part-name {
            flex: 1;
            font-size: 0.9rem;
        }
        
        .part-controls {
            display: flex;
            gap: 0.25rem;
        }
        
        .btn-toggle {
            background: none;
            border: none;
            font-size: 1rem;
            cursor: pointer;
            opacity: 0.6;
            transition: opacity 0.3s ease;
        }
        
        .btn-toggle:hover {
            opacity: 1;
        }
        
        .btn-toggle.active {
            opacity: 1;
        }
        
        .progress-bar {
            width: 100%;
            height: 4px;
            background-color: var(--border-color);
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }
        
        .progress-fill {
            height: 100%;
            background-color: var(--secondary-color);
            transition: width 0.3s ease;
            width: 0%;
        }
        
        .progress-text {
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
        }
        
        .modal-content {
            background-color: var(--surface-color);
            margin: 5% auto;
            padding: 0;
            border-radius: 8px;
            width: 80%;
            max-width: 800px;
            box-shadow: var(--shadow);
        }
        
        .modal-header {
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .modal-body {
            padding: 2rem;
        }
        
        .modal-footer {
            padding: 1rem 2rem;
            border-top: 1px solid var(--border-color);
            display: flex;
            justify-content: flex-end;
            gap: 1rem;
        }
        
        .close-modal {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--text-muted);
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 1rem;
            background-color: var(--surface-color);
            border-top: 1px solid var(--border-color);
            font-size: 0.8rem;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-right: 1rem;
        }
        
        .app-footer {
            background-color: var(--surface-color);
            border-top: 1px solid var(--border-color);
            padding: 1rem 2rem;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        
        .footer-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .footer-links {
            display: flex;
            gap: 1rem;
        }
        
        .footer-links a {
            color: var(--text-muted);
            text-decoration: none;
            transition: color 0.3s ease;
        }
        
        .footer-links a:hover {
            color: var(--secondary-color);
        }
        
        @media (max-width: 768px) {
            .app-container {
                flex-direction: column;
            }
            
            .sidebar {
                width: 100%;
                order: 2;
            }
            
            .main-content {
                order: 1;
            }
            
            .modal-content {
                width: 95%;
                margin: 10% auto;
            }
        }
        """
        
        # Write CSS file
        with open(self.static_path / "css" / "main.css", "w") as f:
            f.write(main_css)
        
        # Create main JavaScript file
        main_js = """
        // Main JavaScript for VR Video Analyzer
        
        class VRVideoAnalyzer {
            constructor() {
                this.sessionId = null;
                this.websocket = null;
                this.currentFrame = 0;
                this.totalFrames = 0;
                this.isAnalyzing = false;
                this.bodyParts = {};
                this.annotationMode = false;
                
                this.init();
            }
            
            init() {
                this.initializeElements();
                this.bindEvents();
                this.setupWebSocket();
                this.loadSettings();
            }
            
            initializeElements() {
                this.videoPlayer = document.getElementById('main-video');
                this.uploadArea = document.getElementById('upload-area');
                this.fileInput = document.getElementById('video-upload');
                this.analyzeBtn = document.getElementById('start-analysis');
                this.annotationCanvas = document.getElementById('annotation-canvas');
                this.timelineSlider = document.getElementById('timeline-slider');
                this.currentTimeDisplay = document.getElementById('current-time');
                this.totalTimeDisplay = document.getElementById('total-time');
            }
            
            bindEvents() {
                // Upload events
                this.uploadArea.addEventListener('click', () => {
                    this.fileInput.click();
                });
                
                this.uploadArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    this.uploadArea.classList.add('dragover');
                });
                
                this.uploadArea.addEventListener('dragleave', () => {
                    this.uploadArea.classList.remove('dragover');
                });
                
                this.uploadArea.addEventListener('drop', (e) => {
                    e.preventDefault();
                    this.uploadArea.classList.remove('dragover');
                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        this.handleVideoUpload(files[0]);
                    }
                });
                
                this.fileInput.addEventListener('change', (e) => {
                    if (e.target.files.length > 0) {
                        this.handleVideoUpload(e.target.files[0]);
                    }
                });
                
                // Analysis events
                this.analyzeBtn.addEventListener('click', () => {
                    this.startAnalysis();
                });
                
                // Video events
                this.videoPlayer.addEventListener('loadedmetadata', () => {
                    this.updateVideoInfo();
                });
                
                this.videoPlayer.addEventListener('timeupdate', () => {
                    this.updateTimeDisplay();
                });
                
                // Timeline events
                this.timelineSlider.addEventListener('input', () => {
                    this.seekToPosition();
                });
                
                // Modal events
                this.bindModalEvents();
            }
            
            bindModalEvents() {
                // Settings modal
                document.getElementById('settings-btn').addEventListener('click', () => {
                    this.showModal('settings-modal');
                });
                
                // Help modal
                document.getElementById('help-btn').addEventListener('click', () => {
                    this.showModal('help-modal');
                });
                
                // Close modal buttons
                document.querySelectorAll('.close-modal').forEach(btn => {
                    btn.addEventListener('click', () => {
                        this.hideModals();
                    });
                });
                
                // Modal backdrop click
                document.querySelectorAll('.modal').forEach(modal => {
                    modal.addEventListener('click', (e) => {
                        if (e.target === modal) {
                            this.hideModals();
                        }
                    });
                });
            }
            
            setupWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId || 'temp'}`;
                
                this.websocket = new WebSocket(wsUrl);
                
                this.websocket.onopen = () => {
                    console.log('WebSocket connected');
                    this.updateConnectionStatus(true);
                };
                
                this.websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                };
                
                this.websocket.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.updateConnectionStatus(false);
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus(false);
                };
            }
            
            handleWebSocketMessage(data) {
                switch(data.type) {
                    case 'status_update':
                        this.updateAnalysisProgress(data.data);
                        break;
                    case 'frame_data':
                        this.updateFrameData(data.data);
                        break;
                    case 'error':
                        this.showError(data.message);
                        break;
                }
            }
            
            async handleVideoUpload(file) {
                const formData = new FormData();
                formData.append('file', file);
                
                this.showUploadProgress(true);
                
                try {
                    const response = await fetch('/upload-video', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        this.sessionId = result.session_id;
                        this.videoPlayer.src = `/uploads/${result.session_id}_${file.name}`;
                        this.analyzeBtn.disabled = false;
                        this.showNotification('Video uploaded successfully!', 'success');
                    } else {
                        this.showError(result.message);
                    }
                } catch (error) {
                    this.showError('Failed to upload video: ' + error.message);
                } finally {
                    this.showUploadProgress(false);
                }
            }
            
            async startAnalysis() {
                if (!this.sessionId) {
                    this.showError('Please upload a video first');
                    return;
                }
                
                this.isAnalyzing = true;
                this.analyzeBtn.disabled = true;
                this.showAnalysisProgress(true);
                
                try {
                    const response = await fetch('/analyze-video', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            video_path: `/uploads/${this.sessionId}`,
                            body_parts: this.getSelectedBodyParts()
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        this.showNotification('Analysis started!', 'success');
                    } else {
                        this.showError(result.message);
                        this.isAnalyzing = false;
                        this.analyzeBtn.disabled = false;
                        this.showAnalysisProgress(false);
                    }
                } catch (error) {
                    this.showError('Failed to start analysis: ' + error.message);
                    this.isAnalyzing = false;
                    this.analyzeBtn.disabled = false;
                    this.showAnalysisProgress(false);
                }
            }
            
            getSelectedBodyParts() {
                const parts = [];
                document.querySelectorAll('.body-part-item').forEach(item => {
                    const partName = item.dataset.part;
                    const isVisible = item.querySelector('[data-action="toggle-visibility"]').classList.contains('active');
                    const isTracked = item.querySelector('[data-action="toggle-tracking"]').classList.contains('active');
                    
                    if (isVisible && isTracked) {
                        parts.push(partName);
                    }
                });
                return parts;
            }
            
            updateAnalysisProgress(data) {
                const progress = data.progress || 0;
                const progressBar = document.querySelector('#analysis-progress .progress-fill');
                const progressText = document.querySelector('#analysis-progress .progress-text');
                
                if (progressBar) {
                    progressBar.style.width = `${progress * 100}%`;
                }
                
                if (progressText) {
                    progressText.textContent = `Analyzing... ${Math.round(progress * 100)}%`;
                }
                
                if (progress >= 1.0) {
                    this.isAnalyzing = false;
                    this.analyzeBtn.disabled = false;
                    this.showAnalysisProgress(false);
                    document.getElementById('generate-funscript').disabled = false;
                    this.showNotification('Analysis completed!', 'success');
                }
            }
            
            updateFrameData(data) {
                this.currentFrame = data.frame_number;
                this.bodyParts = data.body_parts;
                this.updateBodyPartDisplay();
            }
            
            updateBodyPartDisplay() {
                // Update body part indicators in the UI
                document.querySelectorAll('.body-part-item').forEach(item => {
                    const partName = item.dataset.part;
                    const partData = this.bodyParts[partName];
                    
                    if (partData) {
                        item.classList.add('detected');
                        const confidence = partData.confidence || 0;
                        item.style.opacity = 0.5 + (confidence * 0.5);
                    } else {
                        item.classList.remove('detected');
                        item.style.opacity = 0.5;
                    }
                });
            }
            
            updateVideoInfo() {
                this.totalFrames = Math.floor(this.videoPlayer.duration * 30); // Assume 30 FPS
                this.totalTimeDisplay.textContent = this.formatTime(this.videoPlayer.duration);
                this.timelineSlider.max = this.videoPlayer.duration;
            }
            
            updateTimeDisplay() {
                this.currentTimeDisplay.textContent = this.formatTime(this.videoPlayer.currentTime);
                this.timelineSlider.value = this.videoPlayer.currentTime;
                
                // Update frame counter
                const frameNumber = Math.floor(this.videoPlayer.currentTime * 30);
                document.getElementById('frame-counter').textContent = `Frame: ${frameNumber}/${this.totalFrames}`;
            }
            
            seekToPosition() {
                this.videoPlayer.currentTime = this.timelineSlider.value;
            }
            
            formatTime(seconds) {
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            }
            
            showModal(modalId) {
                document.getElementById(modalId).style.display = 'block';
            }
            
            hideModals() {
                document.querySelectorAll('.modal').forEach(modal => {
                    modal.style.display = 'none';
                });
            }
            
            showNotification(message, type = 'info') {
                // Create notification element
                const notification = document.createElement('div');
                notification.className = `notification notification-${type}`;
                notification.textContent = message;
                
                // Add to page
                document.body.appendChild(notification);
                
                // Auto-remove after 3 seconds
                setTimeout(() => {
                    notification.remove();
                }, 3000);
            }
            
            showError(message) {
                this.showNotification(message, 'error');
            }
            
            showUploadProgress(show) {
                const progressDiv = document.getElementById('upload-progress');
                progressDiv.style.display = show ? 'block' : 'none';
            }
            
            showAnalysisProgress(show) {
                const progressDiv = document.getElementById('analysis-progress');
                progressDiv.style.display = show ? 'block' : 'none';
            }
            
            updateConnectionStatus(connected) {
                const statusIndicator = document.getElementById('connection-status');
                if (connected) {
                    statusIndicator.innerHTML = '<i class="icon-wifi"></i> Connected';
                    statusIndicator.style.color = 'var(--success-color)';
                } else {
                    statusIndicator.innerHTML = '<i class="icon-wifi-off"></i> Disconnected';
                    statusIndicator.style.color = 'var(--danger-color)';
                }
            }
            
            loadSettings() {
                // Load settings from localStorage
                const settings = localStorage.getItem('vr-analyzer-settings');
                if (settings) {
                    const parsed = JSON.parse(settings);
                    this.applySettings(parsed);
                }
            }
            
            applySettings(settings) {
                // Apply settings to the application
                console.log('Applying settings:', settings);
            }
        }
        
        // Initialize application when DOM is loaded
        document.addEventListener('DOMContentLoaded', () => {
            window.vrAnalyzer = new VRVideoAnalyzer();
        });
        """
        
        # Write JavaScript file
        with open(self.static_path / "js" / "main.js", "w") as f:
            f.write(main_js)
        
        # Create additional CSS files
        self._create_video_player_css()
        self._create_annotation_css()
        self._create_controls_css()
        
        # Create additional JavaScript files
        self._create_video_player_js()
        self._create_annotation_js()
        self._create_controls_js()
        self._create_websocket_js()
    
    def _create_video_player_css(self):
        """Create video player specific CSS"""
        css_content = """
        .video-controls {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background-color: rgba(0, 0, 0, 0.8);
            border-radius: 0 0 8px 8px;
        }
        
        .control-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .timeline-container {
            flex: 1;
            position: relative;
            margin: 0 1rem;
        }
        
        .timeline-slider {
            width: 100%;
            height: 4px;
            background-color: rgba(255, 255, 255, 0.3);
            border-radius: 2px;
            outline: none;
            appearance: none;
        }
        
        .timeline-slider::-webkit-slider-thumb {
            width: 12px;
            height: 12px;
            background-color: var(--secondary-color);
            border-radius: 50%;
            cursor: pointer;
            appearance: none;
        }
        
        .timeline-markers {
            position: absolute;
            top: -2px;
            left: 0;
            right: 0;
            height: 8px;
            pointer-events: none;
        }
        
        .timeline-marker {
            position: absolute;
            width: 2px;
            height: 100%;
            background-color: var(--accent-color);
            border-radius: 1px;
        }
        
        .time-display {
            font-size: 0.9rem;
            color: var(--text-color);
            min-width: 80px;
            text-align: center;
        }
        """
        
        with open(self.static_path / "css" / "video-player.css", "w") as f:
            f.write(css_content)
    
    def _create_annotation_css(self):
        """Create annotation specific CSS"""
        css_content = """
        .annotation-panel {
            position: fixed;
            top: 0;
            right: -400px;
            width: 400px;
            height: 100vh;
            background-color: var(--surface-color);
            border-left: 1px solid var(--border-color);
            z-index: 100;
            transition: right 0.3s ease;
            display: flex;
            flex-direction: column;
        }
        
        .annotation-panel.active {
            right: 0;
        }
        
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        .panel-content {
            flex: 1;
            padding: 1rem;
            overflow-y: auto;
        }
        
        .annotation-instructions {
            background-color: rgba(52, 152, 219, 0.1);
            border-left: 4px solid var(--secondary-color);
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        
        .annotation-tools {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .tool-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .tool-group label {
            font-size: 0.9rem;
            color: var(--text-muted);
        }
        
        .tool-group select,
        .tool-group input {
            padding: 0.5rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        .annotation-actions {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .annotation-actions .btn {
            flex: 1;
            padding: 0.5rem;
            font-size: 0.8rem;
        }
        
        .annotation-history {
            border-top: 1px solid var(--border-color);
            padding-top: 1rem;
        }
        
        .annotation-history h4 {
            margin-bottom: 0.5rem;
            color: var(--text-color);
        }
        
        .annotation-list {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .annotation-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            font-size: 0.8rem;
        }
        
        .annotation-item:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        .annotation-marker {
            position: absolute;
            width: 12px;
            height: 12px;
            border: 2px solid #fff;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            cursor: pointer;
            z-index: 20;
        }
        
        .annotation-marker.selected {
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.3);
        }
        
        .annotation-marker.low-confidence {
            opacity: 0.6;
        }
        """
        
        with open(self.static_path / "css" / "annotation.css", "w") as f:
            f.write(css_content)
    
    def _create_controls_css(self):
        """Create controls specific CSS"""
        css_content = """
        .settings-tabs {
            display: flex;
            flex-direction: column;
        }
        
        .tab-buttons {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1rem;
        }
        
        .tab-button {
            flex: 1;
            padding: 0.75rem;
            border: none;
            background-color: transparent;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.3s ease;
            border-bottom: 2px solid transparent;
        }
        
        .tab-button.active {
            color: var(--secondary-color);
            border-bottom-color: var(--secondary-color);
        }
        
        .tab-button:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .setting-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .setting-group label {
            font-size: 0.9rem;
            color: var(--text-muted);
        }
        
        .setting-group input,
        .setting-group select {
            padding: 0.5rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        .setting-group input[type="range"] {
            background-color: transparent;
        }
        
        .setting-value {
            font-size: 0.8rem;
            color: var(--text-color);
            background-color: rgba(255, 255, 255, 0.05);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            min-width: 50px;
            text-align: center;
        }
        
        .checkbox-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .checkbox-group label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
            cursor: pointer;
        }
        
        .checkbox-group input[type="checkbox"] {
            width: auto;
            margin: 0;
        }
        
        .funscript-settings {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .help-content {
            line-height: 1.6;
        }
        
        .help-content h3 {
            color: var(--secondary-color);
            margin-bottom: 0.5rem;
            margin-top: 1rem;
        }
        
        .help-content h3:first-child {
            margin-top: 0;
        }
        
        .help-content ol,
        .help-content ul {
            margin-left: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .help-content li {
            margin-bottom: 0.5rem;
        }
        
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem;
            border-radius: 4px;
            color: white;
            font-size: 0.9rem;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        }
        
        .notification-success {
            background-color: var(--success-color);
        }
        
        .notification-error {
            background-color: var(--danger-color);
        }
        
        .notification-info {
            background-color: var(--secondary-color);
        }
        
        .notification-warning {
            background-color: var(--warning-color);
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        """
        
        with open(self.static_path / "css" / "controls.css", "w") as f:
            f.write(css_content)
    
    def _create_video_player_js(self):
        """Create video player JavaScript"""
        js_content = """
        // Video Player functionality
        class VideoPlayer {
            constructor() {
                this.video = document.getElementById('main-video');
                this.playPauseBtn = document.getElementById('play-pause');
                this.stepBackBtn = document.getElementById('step-backward');
                this.stepForwardBtn = document.getElementById('step-forward');
                this.speedBtn = document.getElementById('speed-control');
                this.fullscreenBtn = document.getElementById('fullscreen');
                
                this.speeds = [0.25, 0.5, 1, 1.5, 2];
                this.currentSpeedIndex = 2; // 1x speed
                
                this.init();
            }
            
            init() {
                this.bindEvents();
            }
            
            bindEvents() {
                this.playPauseBtn.addEventListener('click', () => {
                    this.togglePlayPause();
                });
                
                this.stepBackBtn.addEventListener('click', () => {
                    this.stepFrame(-1);
                });
                
                this.stepForwardBtn.addEventListener('click', () => {
                    this.stepFrame(1);
                });
                
                this.speedBtn.addEventListener('click', () => {
                    this.cycleSpeed();
                });
                
                this.fullscreenBtn.addEventListener('click', () => {
                    this.toggleFullscreen();
                });
                
                this.video.addEventListener('play', () => {
                    this.updatePlayPauseButton(true);
                });
                
                this.video.addEventListener('pause', () => {
                    this.updatePlayPauseButton(false);
                });
                
                // Keyboard shortcuts
                document.addEventListener('keydown', (e) => {
                    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                        return;
                    }
                    
                    switch(e.code) {
                        case 'Space':
                            e.preventDefault();
                            this.togglePlayPause();
                            break;
                        case 'ArrowLeft':
                            e.preventDefault();
                            this.stepFrame(-1);
                            break;
                        case 'ArrowRight':
                            e.preventDefault();
                            this.stepFrame(1);
                            break;
                        case 'KeyF':
                            e.preventDefault();
                            this.toggleFullscreen();
                            break;
                    }
                });
            }
            
            togglePlayPause() {
                if (this.video.paused) {
                    this.video.play();
                } else {
                    this.video.pause();
                }
            }
            
            stepFrame(direction) {
                const frameTime = 1 / 30; // Assume 30 FPS
                this.video.currentTime += direction * frameTime;
            }
            
            cycleSpeed() {
                this.currentSpeedIndex = (this.currentSpeedIndex + 1) % this.speeds.length;
                const speed = this.speeds[this.currentSpeedIndex];
                this.video.playbackRate = speed;
                this.speedBtn.textContent = `${speed}x`;
            }
            
            toggleFullscreen() {
                const container = document.querySelector('.video-wrapper');
                
                if (!document.fullscreenElement) {
                    container.requestFullscreen();
                } else {
                    document.exitFullscreen();
                }
            }
            
            updatePlayPauseButton(playing) {
                const icon = this.playPauseBtn.querySelector('i');
                if (playing) {
                    icon.className = 'icon-pause';
                } else {
                    icon.className = 'icon-play';
                }
            }
        }
        
        // Initialize when DOM is loaded
        document.addEventListener('DOMContentLoaded', () => {
            window.videoPlayer = new VideoPlayer();
        });
        """
        
        with open(self.static_path / "js" / "video-player.js", "w") as f:
            f.write(js_content)
    
    def _create_annotation_js(self):
        """Create annotation JavaScript"""
        js_content = """
        // Annotation functionality
        class AnnotationManager {
            constructor() {
                this.canvas = document.getElementById('annotation-canvas');
                this.ctx = this.canvas.getContext('2d');
                this.video = document.getElementById('main-video');
                this.panel = document.getElementById('annotation-panel');
                this.annotationMode = false;
                this.selectedPart = 'head';
                this.annotations = {};
                this.currentAnnotations = [];
                
                this.init();
            }
            
            init() {
                this.bindEvents();
                this.setupCanvas();
            }
            
            bindEvents() {
                document.getElementById('annotation-mode').addEventListener('click', () => {
                    this.toggleAnnotationMode();
                });
                
                document.getElementById('close-annotation').addEventListener('click', () => {
                    this.closeAnnotationPanel();
                });
                
                this.canvas.addEventListener('click', (e) => {
                    if (this.annotationMode) {
                        this.addAnnotation(e);
                    }
                });
                
                this.canvas.addEventListener('mousemove', (e) => {
                    if (this.annotationMode) {
                        this.updateCursor(e);
                    }
                });
                
                document.getElementById('selected-part').addEventListener('change', (e) => {
                    this.selectedPart = e.target.value;
                });
                
                document.getElementById('save-annotation').addEventListener('click', () => {
                    this.saveAnnotations();
                });
                
                document.getElementById('delete-annotation').addEventListener('click', () => {
                    this.deleteSelectedAnnotation();
                });
                
                document.getElementById('clear-frame').addEventListener('click', () => {
                    this.clearFrameAnnotations();
                });
                
                // Update canvas size when video loads
                this.video.addEventListener('loadedmetadata', () => {
                    this.setupCanvas();
                });
                
                // Update annotations when video time changes
                this.video.addEventListener('timeupdate', () => {
                    this.updateAnnotationDisplay();
                });
            }
            
            setupCanvas() {
                const rect = this.video.getBoundingClientRect();
                this.canvas.width = this.video.videoWidth || rect.width;
                this.canvas.height = this.video.videoHeight || rect.height;
                this.redrawAnnotations();
            }
            
            toggleAnnotationMode() {
                this.annotationMode = !this.annotationMode;
                
                if (this.annotationMode) {
                    this.canvas.classList.add('active');
                    this.panel.classList.add('active');
                    document.getElementById('annotation-mode').classList.add('active');
                } else {
                    this.canvas.classList.remove('active');
                    this.panel.classList.remove('active');
                    document.getElementById('annotation-mode').classList.remove('active');
                }
            }
            
            closeAnnotationPanel() {
                this.annotationMode = false;
                this.canvas.classList.remove('active');
                this.panel.classList.remove('active');
                document.getElementById('annotation-mode').classList.remove('active');
            }
            
            addAnnotation(event) {
                const rect = this.canvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                
                // Convert to video coordinates
                const videoX = (x / rect.width) * this.canvas.width;
                const videoY = (y / rect.height) * this.canvas.height;
                
                const annotation = {
                    id: Date.now(),
                    part: this.selectedPart,
                    x: videoX,
                    y: videoY,
                    confidence: parseFloat(document.getElementById('confidence-slider').value),
                    frame: Math.floor(this.video.currentTime * 30),
                    timestamp: this.video.currentTime
                };
                
                this.currentAnnotations.push(annotation);
                this.redrawAnnotations();
                this.updateAnnotationList();
            }
            
            updateCursor(event) {
                const rect = this.canvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                
                // Show preview marker
                this.redrawAnnotations();
                
                this.ctx.strokeStyle = '#3498db';
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(x, y, 8, 0, 2 * Math.PI);
                this.ctx.stroke();
            }
            
            redrawAnnotations() {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                
                for (const annotation of this.currentAnnotations) {
                    this.drawAnnotation(annotation);
                }
            }
            
            drawAnnotation(annotation) {
                const color = this.getPartColor(annotation.part);
                const alpha = annotation.confidence;
                
                this.ctx.fillStyle = color;
                this.ctx.globalAlpha = alpha;
                this.ctx.beginPath();
                this.ctx.arc(annotation.x, annotation.y, 8, 0, 2 * Math.PI);
                this.ctx.fill();
                
                // Draw outline
                this.ctx.strokeStyle = '#fff';
                this.ctx.lineWidth = 2;
                this.ctx.globalAlpha = 1;
                this.ctx.stroke();
                
                // Draw label
                this.ctx.fillStyle = '#fff';
                this.ctx.font = '12px Arial';
                this.ctx.fillText(annotation.part, annotation.x + 12, annotation.y - 8);
            }
            
            getPartColor(part) {
                const colors = {
                    'head': '#ff6b6b',
                    'mouth': '#4ecdc4',
                    'hand_1': '#45b7d1',
                    'hand_2': '#96ceb4',
                    'breasts': '#ffeaa7',
                    'pelvis': '#dda0dd',
                    'genitals': '#fd79a8'
                };
                return colors[part] || '#ffffff';
            }
            
            updateAnnotationDisplay() {
                const currentFrame = Math.floor(this.video.currentTime * 30);
                
                // Load annotations for current frame
                this.currentAnnotations = this.annotations[currentFrame] || [];
                this.redrawAnnotations();
                this.updateAnnotationList();
            }
            
            updateAnnotationList() {
                const list = document.getElementById('annotation-list');
                list.innerHTML = '';
                
                for (const annotation of this.currentAnnotations) {
                    const item = document.createElement('div');
                    item.className = 'annotation-item';
                    item.innerHTML = `
                        <span>${annotation.part}</span>
                        <span>${annotation.confidence.toFixed(1)}</span>
                        <button onclick="annotationManager.deleteAnnotation(${annotation.id})">×</button>
                    `;
                    list.appendChild(item);
                }
            }
            
            deleteAnnotation(id) {
                this.currentAnnotations = this.currentAnnotations.filter(a => a.id !== id);
                this.redrawAnnotations();
                this.updateAnnotationList();
            }
            
            deleteSelectedAnnotation() {
                if (this.currentAnnotations.length > 0) {
                    this.currentAnnotations.pop();
                    this.redrawAnnotations();
                    this.updateAnnotationList();
                }
            }
            
            clearFrameAnnotations() {
                this.currentAnnotations = [];
                this.redrawAnnotations();
                this.updateAnnotationList();
            }
            
            async saveAnnotations() {
                const currentFrame = Math.floor(this.video.currentTime * 30);
                this.annotations[currentFrame] = [...this.currentAnnotations];
                
                // Send to server
                for (const annotation of this.currentAnnotations) {
                    try {
                        await fetch('/correct-body-part', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                session_id: window.vrAnalyzer.sessionId,
                                frame_number: annotation.frame,
                                body_part: annotation.part,
                                position: {
                                    x: annotation.x,
                                    y: annotation.y,
                                    z: 0
                                }
                            })
                        });
                    } catch (error) {
                        console.error('Error saving annotation:', error);
                    }
                }
                
                window.vrAnalyzer.showNotification('Annotations saved!', 'success');
            }
        }
        
        // Initialize when DOM is loaded
        document.addEventListener('DOMContentLoaded', () => {
            window.annotationManager = new AnnotationManager();
        });
        """
        
        with open(self.static_path / "js" / "annotation.js", "w") as f:
            f.write(js_content)
    
    def _create_controls_js(self):
        """Create controls JavaScript"""
        js_content = """
        // Controls functionality
        class ControlsManager {
            constructor() {
                this.init();
            }
            
            init() {
                this.bindEvents();
                this.initializeSettings();
            }
            
            bindEvents() {
                // Body part toggle events
                document.querySelectorAll('.body-part-item').forEach(item => {
                    const visibilityBtn = item.querySelector('[data-action="toggle-visibility"]');
                    const trackingBtn = item.querySelector('[data-action="toggle-tracking"]');
                    
                    visibilityBtn.addEventListener('click', () => {
                        this.toggleBodyPartVisibility(item.dataset.part, visibilityBtn);
                    });
                    
                    trackingBtn.addEventListener('click', () => {
                        this.toggleBodyPartTracking(item.dataset.part, trackingBtn);
                    });
                });
                
                // Funscript generation
                document.getElementById('generate-funscript').addEventListener('click', () => {
                    this.generateFunscript();
                });
                
                // Settings modal tabs
                document.querySelectorAll('.tab-button').forEach(btn => {
                    btn.addEventListener('click', () => {
                        this.switchTab(btn.dataset.tab);
                    });
                });
                
                // Settings save
                document.getElementById('save-settings').addEventListener('click', () => {
                    this.saveSettings();
                });
                
                // Sensitivity slider
                document.getElementById('sensitivity-slider').addEventListener('input', (e) => {
                    document.getElementById('sensitivity-value').textContent = e.target.value;
                });
                
                // Confidence slider
                document.getElementById('confidence-slider').addEventListener('input', (e) => {
                    document.getElementById('confidence-value').textContent = e.target.value;
                });
                
                // Settings value displays
                document.querySelectorAll('input[type="range"]').forEach(slider => {
                    slider.addEventListener('input', (e) => {
                        const valueDisplay = e.target.parentElement.querySelector('.setting-value');
                        if (valueDisplay) {
                            let value = e.target.value;
                            if (e.target.id === 'interaction-distance') {
                                value += 'm';
                            }
                            valueDisplay.textContent = value;
                        }
                    });
                });
            }
            
            toggleBodyPartVisibility(partName, button) {
                button.classList.toggle('active');
                const isVisible = button.classList.contains('active');
                
                // Update UI
                const partItem = button.closest('.body-part-item');
                if (isVisible) {
                    partItem.style.opacity = '1';
                } else {
                    partItem.style.opacity = '0.5';
                }
                
                // Notify other components
                this.notifyBodyPartToggle(partName, 'visibility', isVisible);
            }
            
            toggleBodyPartTracking(partName, button) {
                button.classList.toggle('active');
                const isTracking = button.classList.contains('active');
                
                // Update UI
                const partItem = button.closest('.body-part-item');
                if (isTracking) {
                    partItem.classList.add('tracking');
                } else {
                    partItem.classList.remove('tracking');
                }
                
                // Notify other components
                this.notifyBodyPartToggle(partName, 'tracking', isTracking);
            }
            
            notifyBodyPartToggle(partName, action, state) {
                // Dispatch custom event
                const event = new CustomEvent('bodyPartToggle', {
                    detail: {
                        partName,
                        action,
                        state
                    }
                });
                document.dispatchEvent(event);
            }
            
            async generateFunscript() {
                if (!window.vrAnalyzer.sessionId) {
                    window.vrAnalyzer.showError('Please complete video analysis first');
                    return;
                }
                
                const targetPart = document.getElementById('target-part').value;
                const interactionParts = [];
                
                document.querySelectorAll('.checkbox-group input[type="checkbox"]:checked').forEach(checkbox => {
                    interactionParts.push(checkbox.value);
                });
                
                const sensitivity = parseFloat(document.getElementById('sensitivity-slider').value);
                
                try {
                    const response = await fetch('/generate-funscript', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            session_id: window.vrAnalyzer.sessionId,
                            target_body_part: targetPart,
                            interaction_parts: interactionParts,
                            sensitivity: sensitivity
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        window.vrAnalyzer.showNotification('Funscript generated successfully!', 'success');
                        
                        // Create download link
                        const link = document.createElement('a');
                        link.href = result.download_url;
                        link.download = `${window.vrAnalyzer.sessionId}.funscript`;
                        link.click();
                    } else {
                        window.vrAnalyzer.showError(result.message);
                    }
                } catch (error) {
                    window.vrAnalyzer.showError('Failed to generate funscript: ' + error.message);
                }
            }
            
            switchTab(tabName) {
                // Update tab buttons
                document.querySelectorAll('.tab-button').forEach(btn => {
                    btn.classList.remove('active');
                });
                document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
                
                // Update tab content
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                document.querySelector(`.tab-content[data-tab="${tabName}"]`).classList.add('active');
            }
            
            saveSettings() {
                const settings = {
                    detection: {
                        confidence: parseFloat(document.getElementById('detection-confidence').value),
                        nmsThreshold: parseFloat(document.getElementById('nms-threshold').value),
                        maxDetections: parseInt(document.getElementById('max-detections').value)
                    },
                    tracking: {
                        algorithm: document.getElementById('tracking-algorithm').value,
                        maxLostFrames: parseInt(document.getElementById('max-lost-frames').value),
                        smoothingFactor: parseFloat(document.getElementById('smoothing-factor').value)
                    },
                    funscript: {
                        samplingRate: parseInt(document.getElementById('sampling-rate').value),
                        smoothingWindow: parseInt(document.getElementById('smoothing-window').value),
                        interactionDistance: parseFloat(document.getElementById('interaction-distance').value)
                    },
                    performance: {
                        gpuDevice: document.getElementById('gpu-device').value,
                        batchSize: parseInt(document.getElementById('batch-size').value),
                        memoryFraction: parseFloat(document.getElementById('memory-fraction').value)
                    }
                };
                
                // Save to localStorage
                localStorage.setItem('vr-analyzer-settings', JSON.stringify(settings));
                
                // Apply settings
                window.vrAnalyzer.applySettings(settings);
                
                // Close modal
                window.vrAnalyzer.hideModals();
                
                window.vrAnalyzer.showNotification('Settings saved!', 'success');
            }
            
            initializeSettings() {
                // Set default values for body part toggles
                document.querySelectorAll('.body-part-item').forEach(item => {
                    const visibilityBtn = item.querySelector('[data-action="toggle-visibility"]');
                    const trackingBtn = item.querySelector('[data-action="toggle-tracking"]');
                    
                    // Enable by default
                    visibilityBtn.classList.add('active');
                    trackingBtn.classList.add('active');
                });
            }
        }
        
        // Initialize when DOM is loaded
        document.addEventListener('DOMContentLoaded', () => {
            window.controlsManager = new ControlsManager();
        });
        """
        
        with open(self.static_path / "js" / "controls.js", "w") as f:
            f.write(js_content)
    
    def _create_websocket_js(self):
        """Create WebSocket JavaScript"""
        js_content = """
        // WebSocket communication
        class WebSocketManager {
            constructor() {
                this.socket = null;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.reconnectDelay = 1000;
                this.messageHandlers = new Map();
                
                this.init();
            }
            
            init() {
                this.connect();
                this.registerHandlers();
            }
            
            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const sessionId = window.vrAnalyzer?.sessionId || 'temp';
                const url = `${protocol}//${window.location.host}/ws/${sessionId}`;
                
                try {
                    this.socket = new WebSocket(url);
                    
                    this.socket.onopen = () => {
                        console.log('WebSocket connected');
                        this.reconnectAttempts = 0;
                        this.updateConnectionStatus(true);
                    };
                    
                    this.socket.onmessage = (event) => {
                        this.handleMessage(event.data);
                    };
                    
                    this.socket.onclose = (event) => {
                        console.log('WebSocket disconnected');
                        this.updateConnectionStatus(false);
                        this.handleReconnect();
                    };
                    
                    this.socket.onerror = (error) => {
                        console.error('WebSocket error:', error);
                        this.updateConnectionStatus(false);
                    };
                    
                } catch (error) {
                    console.error('Failed to create WebSocket:', error);
                    this.updateConnectionStatus(false);
                }
            }
            
            handleMessage(data) {
                try {
                    const message = JSON.parse(data);
                    
                    // Call registered handlers
                    if (this.messageHandlers.has(message.type)) {
                        this.messageHandlers.get(message.type)(message.data);
                    } else {
                        console.log('Unhandled message type:', message.type);
                    }
                    
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            }
            
            registerHandlers() {
                this.messageHandlers.set('status_update', (data) => {
                    this.handleStatusUpdate(data);
                });
                
                this.messageHandlers.set('frame_data', (data) => {
                    this.handleFrameData(data);
                });
                
                this.messageHandlers.set('error', (data) => {
                    this.handleError(data);
                });
                
                this.messageHandlers.set('analysis_complete', (data) => {
                    this.handleAnalysisComplete(data);
                });
            }
            
            handleStatusUpdate(data) {
                if (window.vrAnalyzer) {
                    window.vrAnalyzer.updateAnalysisProgress(data);
                }
            }
            
            handleFrameData(data) {
                if (window.vrAnalyzer) {
                    window.vrAnalyzer.updateFrameData(data);
                }
                
                // Update annotation display
                if (window.annotationManager) {
                    window.annotationManager.updateAnnotationDisplay();
                }
            }
            
            handleError(data) {
                if (window.vrAnalyzer) {
                    window.vrAnalyzer.showError(data.message);
                }
            }
            
            handleAnalysisComplete(data) {
                if (window.vrAnalyzer) {
                    window.vrAnalyzer.showNotification('Analysis completed!', 'success');
                    document.getElementById('generate-funscript').disabled = false;
                }
            }
            
            handleReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    
                    setTimeout(() => {
                        console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                        this.connect();
                    }, this.reconnectDelay * this.reconnectAttempts);
                } else {
                    console.error('Max reconnection attempts reached');
                }
            }
            
            send(message) {
                if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                    this.socket.send(JSON.stringify(message));
                } else {
                    console.error('WebSocket not connected');
                }
            }
            
            updateConnectionStatus(connected) {
                const statusElement = document.getElementById('connection-status');
                if (statusElement) {
                    if (connected) {
                        statusElement.innerHTML = '<i class="icon-wifi"></i> Connected';
                        statusElement.style.color = 'var(--success-color)';
                    } else {
                        statusElement.innerHTML = '<i class="icon-wifi-off"></i> Disconnected';
                        statusElement.style.color = 'var(--danger-color)';
                    }
                }
            }
            
            disconnect() {
                if (this.socket) {
                    this.socket.close();
                    this.socket = null;
                }
            }
        }
        
        // Initialize when DOM is loaded
        document.addEventListener('DOMContentLoaded', () => {
            window.webSocketManager = new WebSocketManager();
        });
        """
        
        with open(self.static_path / "js" / "websocket.js", "w") as f:
            f.write(js_content)