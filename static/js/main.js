/**
 * VR Assistive Technology - Main JavaScript
 * Handles WebSocket communication, video display, tracking visualization, and user interactions
 */

class VRAssistiveTechnology {
    constructor() {
        this.ws = null;
        this.canvas = null;
        this.ctx = null;
        this.videoInfo = null;
        this.currentFrame = 0;
        this.isProcessing = false;
        this.trackingData = {};
        this.correctionMode = false;
        this.correctionData = null;
        this.zoomLevel = 1;
        this.panX = 0;
        this.panY = 0;
        
        // Body part colors
        this.bodyPartColors = {
            'head': '#dc3545',
            'mouth': '#ffc107',
            'hand_1': '#28a745',
            'hand_2': '#17a2b8',
            'breasts': '#007bff',
            'pelvis': '#6c757d',
            'genitals': '#343a40'
        };
        
        this.init();
    }
    
    init() {
        this.setupElements();
        this.setupEventListeners();
        this.connectWebSocket();
        this.initCanvas();
    }
    
    setupElements() {
        // Get DOM elements
        this.elements = {
            videoFile: document.getElementById('videoFile'),
            uploadBtn: document.getElementById('uploadBtn'),
            startBtn: document.getElementById('startBtn'),
            pauseBtn: document.getElementById('pauseBtn'),
            stopBtn: document.getElementById('stopBtn'),
            progressBar: document.getElementById('progressBar'),
            statusBar: document.getElementById('statusBar'),
            statusText: document.getElementById('statusText'),
            videoCanvas: document.getElementById('videoCanvas'),
            videoContainer: document.getElementById('videoContainer'),
            videoPlaceholder: document.getElementById('videoPlaceholder'),
            timeline: document.getElementById('timeline'),
            timelineProgress: document.getElementById('timelineProgress'),
            timelineHandle: document.getElementById('timelineHandle'),
            currentTime: document.getElementById('currentTime'),
            totalTime: document.getElementById('totalTime'),
            currentFrame: document.getElementById('currentFrame'),
            totalFrames: document.getElementById('totalFrames'),
            generateBtn: document.getElementById('generateBtn'),
            downloadBtn: document.getElementById('downloadBtn'),
            speedMultiplier: document.getElementById('speedMultiplier'),
            depthMultiplier: document.getElementById('depthMultiplier'),
            intensityMultiplier: document.getElementById('intensityMultiplier'),
            speedMetric: document.getElementById('speedMetric'),
            depthMetric: document.getElementById('depthMetric'),
            intensityMetric: document.getElementById('intensityMetric'),
            interactionMetric: document.getElementById('interactionMetric'),
            speedBar: document.getElementById('speedBar'),
            depthBar: document.getElementById('depthBar'),
            intensityBar: document.getElementById('intensityBar'),
            interactionBar: document.getElementById('interactionBar'),
            correctionModal: document.getElementById('correctionModal'),
            correctionBodyPart: document.getElementById('correctionBodyPart'),
            correctionFrame: document.getElementById('correctionFrame'),
            applyCorrectionBtn: document.getElementById('applyCorrectionBtn'),
            zoomInBtn: document.getElementById('zoomInBtn'),
            zoomOutBtn: document.getElementById('zoomOutBtn'),
            resetZoomBtn: document.getElementById('resetZoomBtn')
        };
        
        this.canvas = this.elements.videoCanvas;
        this.ctx = this.canvas.getContext('2d');
    }
    
    setupEventListeners() {
        // Upload button
        this.elements.uploadBtn.addEventListener('click', () => this.uploadVideo());
        
        // Control buttons
        this.elements.startBtn.addEventListener('click', () => this.startAnalysis());
        this.elements.pauseBtn.addEventListener('click', () => this.pauseAnalysis());
        this.elements.stopBtn.addEventListener('click', () => this.stopAnalysis());
        
        // Generation and download
        this.elements.generateBtn.addEventListener('click', () => this.generateFunscript());
        this.elements.downloadBtn.addEventListener('click', () => this.downloadFunscript());
        
        // Parameter sliders
        this.elements.speedMultiplier.addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = e.target.value;
        });
        this.elements.depthMultiplier.addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = e.target.value;
        });
        this.elements.intensityMultiplier.addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = e.target.value;
        });
        
        // Timeline interaction
        this.elements.timeline.addEventListener('click', (e) => this.handleTimelineClick(e));
        this.elements.timelineHandle.addEventListener('mousedown', (e) => this.handleTimelineDrag(e));
        
        // Canvas interaction
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleCanvasMouseMove(e));
        
        // Zoom controls
        this.elements.zoomInBtn.addEventListener('click', () => this.zoomIn());
        this.elements.zoomOutBtn.addEventListener('click', () => this.zoomOut());
        this.elements.resetZoomBtn.addEventListener('click', () => this.resetZoom());
        
        // Correction modal
        this.elements.applyCorrectionBtn.addEventListener('click', () => this.applyCorrectionFromModal());
        
        // Body part tracking checkboxes
        const trackingCheckboxes = document.querySelectorAll('[id^="track"]');
        trackingCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => this.updateTrackingSettings());
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
        
        // Window resize
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateStatus('Connected to server', 'success');
        };
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateStatus('Disconnected from server', 'danger');
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Connection error', 'danger');
        };
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'status':
                this.updateProcessingStatus(message.is_processing, message.has_video);
                break;
            case 'analysis_update':
                this.updateAnalysisData(message.data);
                break;
            case 'analysis_complete':
                this.handleAnalysisComplete();
                break;
            case 'error':
                this.updateStatus(message.message, 'danger');
                break;
            case 'pong':
                // Handle ping response
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }
    
    async uploadVideo() {
        const file = this.elements.videoFile.files[0];
        if (!file) {
            this.updateStatus('Please select a video file', 'warning');
            return;
        }
        
        this.updateStatus('Uploading video...', 'info');
        
        const formData = new FormData();
        formData.append('video', file);
        
        try {
            const response = await fetch('/upload-video', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.videoInfo = result.video_info;
                this.updateStatus('Video uploaded successfully', 'success');
                this.elements.startBtn.disabled = false;
                this.elements.videoPlaceholder.style.display = 'none';
                this.updateVideoInfo();
            } else {
                this.updateStatus(`Upload failed: ${result.error}`, 'danger');
            }
        } catch (error) {
            this.updateStatus(`Upload error: ${error.message}`, 'danger');
        }
    }
    
    async startAnalysis() {
        this.updateStatus('Starting analysis...', 'info');
        
        try {
            const response = await fetch('/start-analysis', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isProcessing = true;
                this.updateStatus('Analysis in progress...', 'info');
                this.elements.startBtn.disabled = true;
                this.elements.pauseBtn.disabled = false;
                this.elements.stopBtn.disabled = false;
            } else {
                this.updateStatus(`Start failed: ${result.error}`, 'danger');
            }
        } catch (error) {
            this.updateStatus(`Start error: ${error.message}`, 'danger');
        }
    }
    
    async pauseAnalysis() {
        try {
            const response = await fetch('/pause-analysis', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isProcessing = false;
                this.updateStatus('Analysis paused', 'warning');
                this.elements.startBtn.disabled = false;
                this.elements.pauseBtn.disabled = true;
            }
        } catch (error) {
            this.updateStatus(`Pause error: ${error.message}`, 'danger');
        }
    }
    
    async stopAnalysis() {
        this.isProcessing = false;
        this.updateStatus('Analysis stopped', 'info');
        this.elements.startBtn.disabled = false;
        this.elements.pauseBtn.disabled = true;
        this.elements.stopBtn.disabled = true;
        this.elements.generateBtn.disabled = false;
    }
    
    updateAnalysisData(data) {
        this.currentFrame = data.frame_number;
        this.trackingData[data.frame_number] = data;
        
        // Update progress
        if (this.videoInfo) {
            const progress = (data.frame_number / this.videoInfo.frame_count) * 100;
            this.elements.progressBar.style.width = `${progress}%`;
            this.elements.progressBar.textContent = `${Math.round(progress)}%`;
        }
        
        // Update timeline
        this.updateTimeline();
        
        // Update metrics
        this.updateMetrics(data.interactions);
        
        // Update canvas
        this.updateCanvas(data);
    }
    
    updateCanvas(data) {
        if (!this.canvas || !this.ctx) return;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw tracking points
        data.body_parts.forEach(bodyPart => {
            if (this.isBodyPartEnabled(bodyPart.name)) {
                this.drawTrackingPoints(bodyPart);
            }
        });
        
        // Draw interaction lines
        this.drawInteractionLines(data.interactions);
    }
    
    drawTrackingPoints(bodyPart) {
        const color = this.bodyPartColors[bodyPart.name] || '#ffffff';
        
        bodyPart.points.forEach(point => {
            const x = point.x * this.zoomLevel + this.panX;
            const y = point.y * this.zoomLevel + this.panY;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, 6, 0, 2 * Math.PI);
            this.ctx.fillStyle = color;
            this.ctx.fill();
            this.ctx.strokeStyle = point.is_corrected ? '#ffc107' : '#ffffff';
            this.ctx.lineWidth = point.is_corrected ? 3 : 2;
            this.ctx.stroke();
        });
    }
    
    drawInteractionLines(interactions) {
        // Draw lines between interacting body parts
        Object.entries(interactions).forEach(([key, value]) => {
            if (value > 0.1) { // Only draw significant interactions
                // This would draw interaction lines between body parts
                // Implementation depends on specific interaction types
            }
        });
    }
    
    updateMetrics(interactions) {
        const speed = interactions.speed || 0;
        const depth = interactions.depth || 0;
        const intensity = interactions.intensity || 0;
        const interactionCount = interactions.interaction_count || 0;
        
        // Update metric values
        this.elements.speedMetric.textContent = speed.toFixed(1);
        this.elements.depthMetric.textContent = depth.toFixed(1);
        this.elements.intensityMetric.textContent = intensity.toFixed(1);
        this.elements.interactionMetric.textContent = interactionCount;
        
        // Update metric bars
        this.elements.speedBar.style.width = `${Math.min(speed * 100, 100)}%`;
        this.elements.depthBar.style.width = `${Math.min(depth * 100, 100)}%`;
        this.elements.intensityBar.style.width = `${Math.min(intensity * 100, 100)}%`;
        this.elements.interactionBar.style.width = `${Math.min(interactionCount * 10, 100)}%`;
    }
    
    updateTimeline() {
        if (!this.videoInfo) return;
        
        const progress = (this.currentFrame / this.videoInfo.frame_count) * 100;
        this.elements.timelineProgress.style.width = `${progress}%`;
        this.elements.timelineHandle.style.left = `${progress}%`;
        
        // Update time display
        const currentTime = this.currentFrame / this.videoInfo.fps;
        const totalTime = this.videoInfo.duration;
        
        this.elements.currentTime.textContent = this.formatTime(currentTime);
        this.elements.totalTime.textContent = this.formatTime(totalTime);
        this.elements.currentFrame.textContent = this.currentFrame;
        this.elements.totalFrames.textContent = this.videoInfo.frame_count;
    }
    
    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    
    handleCanvasClick(event) {
        if (!this.correctionMode) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left - this.panX) / this.zoomLevel;
        const y = (event.clientY - rect.top - this.panY) / this.zoomLevel;
        
        // Apply correction
        this.applyCorrection(x, y);
    }
    
    async applyCorrection(x, y) {
        if (!this.correctionData) return;
        
        const correctionRequest = {
            frame_number: this.correctionData.frame,
            body_part: this.correctionData.bodyPart,
            position: { x, y }
        };
        
        try {
            const response = await fetch('/correct-tracking', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(correctionRequest)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.updateStatus('Tracking corrected', 'success');
                this.exitCorrectionMode();
            } else {
                this.updateStatus(`Correction failed: ${result.error}`, 'danger');
            }
        } catch (error) {
            this.updateStatus(`Correction error: ${error.message}`, 'danger');
        }
    }
    
    async generateFunscript() {
        this.updateStatus('Generating funscript...', 'info');
        
        try {
            const response = await fetch('/generate-funscript', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.updateStatus(`Funscript generated with ${result.actions_count} actions`, 'success');
                this.elements.downloadBtn.disabled = false;
            } else {
                this.updateStatus(`Generation failed: ${result.error}`, 'danger');
            }
        } catch (error) {
            this.updateStatus(`Generation error: ${error.message}`, 'danger');
        }
    }
    
    async downloadFunscript() {
        try {
            const response = await fetch('/download-funscript');
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'generated.funscript';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.updateStatus('Funscript downloaded', 'success');
            } else {
                this.updateStatus('Download failed', 'danger');
            }
        } catch (error) {
            this.updateStatus(`Download error: ${error.message}`, 'danger');
        }
    }
    
    initCanvas() {
        this.resizeCanvas();
    }
    
    resizeCanvas() {
        const container = this.elements.videoContainer;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }
    
    updateStatus(message, type = 'info') {
        this.elements.statusText.textContent = message;
        this.elements.statusBar.className = `alert alert-${type}`;
    }
    
    updateVideoInfo() {
        if (!this.videoInfo) return;
        
        this.elements.totalTime.textContent = this.formatTime(this.videoInfo.duration);
        this.elements.totalFrames.textContent = this.videoInfo.frame_count;
    }
    
    isBodyPartEnabled(bodyPart) {
        const checkbox = document.getElementById(`track${bodyPart.charAt(0).toUpperCase() + bodyPart.slice(1).replace('_', '')}`);
        return checkbox ? checkbox.checked : true;
    }
    
    updateTrackingSettings() {
        // Update tracking settings based on checkboxes
        // This would send settings to the server
    }
    
    handleKeyboardShortcuts(event) {
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case ' ':
                    event.preventDefault();
                    if (this.isProcessing) {
                        this.pauseAnalysis();
                    } else {
                        this.startAnalysis();
                    }
                    break;
                case 'g':
                    event.preventDefault();
                    this.generateFunscript();
                    break;
                case 'd':
                    event.preventDefault();
                    this.downloadFunscript();
                    break;
            }
        }
    }
    
    zoomIn() {
        this.zoomLevel = Math.min(this.zoomLevel * 1.2, 5);
        this.updateCanvasTransform();
    }
    
    zoomOut() {
        this.zoomLevel = Math.max(this.zoomLevel / 1.2, 0.5);
        this.updateCanvasTransform();
    }
    
    resetZoom() {
        this.zoomLevel = 1;
        this.panX = 0;
        this.panY = 0;
        this.updateCanvasTransform();
    }
    
    updateCanvasTransform() {
        // Update canvas transform for zoom and pan
        this.ctx.setTransform(this.zoomLevel, 0, 0, this.zoomLevel, this.panX, this.panY);
    }
    
    handleAnalysisComplete() {
        this.isProcessing = false;
        this.updateStatus('Analysis completed', 'success');
        this.elements.startBtn.disabled = false;
        this.elements.pauseBtn.disabled = true;
        this.elements.stopBtn.disabled = true;
        this.elements.generateBtn.disabled = false;
    }
    
    exitCorrectionMode() {
        this.correctionMode = false;
        this.correctionData = null;
        this.elements.videoContainer.classList.remove('correction-mode');
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(this.elements.correctionModal);
        if (modal) {
            modal.hide();
        }
    }
    
    handleTimelineClick(event) {
        const rect = this.elements.timeline.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const percentage = clickX / rect.width;
        
        if (this.videoInfo) {
            const targetFrame = Math.floor(percentage * this.videoInfo.frame_count);
            this.seekToFrame(targetFrame);
        }
    }
    
    seekToFrame(frame) {
        this.currentFrame = frame;
        this.updateTimeline();
        // Update canvas with frame data if available
        if (this.trackingData[frame]) {
            this.updateCanvas(this.trackingData[frame]);
        }
    }
    
    handleTimelineDrag(event) {
        let isDragging = true;
        
        const handleMouseMove = (e) => {
            if (!isDragging) return;
            
            const rect = this.elements.timeline.getBoundingClientRect();
            const percentage = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            
            if (this.videoInfo) {
                const targetFrame = Math.floor(percentage * this.videoInfo.frame_count);
                this.seekToFrame(targetFrame);
            }
        };
        
        const handleMouseUp = () => {
            isDragging = false;
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
        
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    }
    
    updateProcessingStatus(isProcessing, hasVideo) {
        this.isProcessing = isProcessing;
        this.elements.startBtn.disabled = !hasVideo || isProcessing;
        this.elements.pauseBtn.disabled = !isProcessing;
        this.elements.stopBtn.disabled = !isProcessing;
    }
    
    handleCanvasMouseMove(event) {
        // Handle mouse move for tooltips or other interactions
        if (this.correctionMode) {
            const rect = this.canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            // Update cursor or crosshair position
            this.updateCorrectionCursor(x, y);
        }
    }
    
    updateCorrectionCursor(x, y) {
        // Update visual feedback for correction mode
        let crosshair = document.querySelector('.correction-crosshair');
        if (!crosshair) {
            crosshair = document.createElement('div');
            crosshair.className = 'correction-crosshair';
            this.elements.videoContainer.appendChild(crosshair);
        }
        
        crosshair.style.left = `${x}px`;
        crosshair.style.top = `${y}px`;
    }
    
    applyCorrectionFromModal() {
        // This would be triggered by the modal's apply button
        // The actual correction is handled by canvas click
        this.correctionMode = true;
        this.elements.videoContainer.classList.add('correction-mode');
        
        // Close modal but keep correction mode active
        const modal = bootstrap.Modal.getInstance(this.elements.correctionModal);
        if (modal) {
            modal.hide();
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.vrApp = new VRAssistiveTechnology();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (window.vrApp && window.vrApp.ws) {
        if (document.hidden) {
            // Page is hidden, potentially close WebSocket
        } else {
            // Page is visible, ensure WebSocket is connected
            if (window.vrApp.ws.readyState === WebSocket.CLOSED) {
                window.vrApp.connectWebSocket();
            }
        }
    }
});

// Handle before unload
window.addEventListener('beforeunload', () => {
    if (window.vrApp && window.vrApp.ws) {
        window.vrApp.ws.close();
    }
});