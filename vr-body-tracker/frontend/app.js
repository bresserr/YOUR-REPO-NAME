class VRBodyTracker {
    constructor() {
        this.ws = null;
        this.videoInfo = null;
        this.currentFrame = 0;
        this.isPlaying = false;
        this.isAnalyzing = false;
        this.correctionMode = false;
        this.canvas = document.getElementById('videoCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
    }
    
    initializeElements() {
        // Get DOM elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.videoInfoSection = document.getElementById('videoInfo');
        this.previewSection = document.getElementById('previewSection');
        this.playBtn = document.getElementById('playBtn');
        this.pauseBtn = document.getElementById('pauseBtn');
        this.progressBar = document.getElementById('progressBar');
        this.currentTimeEl = document.getElementById('currentTime');
        this.totalTimeEl = document.getElementById('totalTime');
        this.startAnalysisBtn = document.getElementById('startAnalysis');
        this.correctionModeBtn = document.getElementById('correctionMode');
        this.correctionModal = document.getElementById('correctionModal');
        this.generateFunscriptBtn = document.getElementById('generateFunscript');
    }
    
    setupEventListeners() {
        // File upload
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        
        // Drag and drop
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
                this.uploadFile(files[0]);
            }
        });
        
        // Playback controls
        this.playBtn.addEventListener('click', () => this.play());
        this.pauseBtn.addEventListener('click', () => this.pause());
        this.progressBar.addEventListener('input', (e) => this.seek(e.target.value));
        
        // Analysis controls
        this.startAnalysisBtn.addEventListener('click', () => this.startAnalysis());
        this.correctionModeBtn.addEventListener('click', () => this.toggleCorrectionMode());
        
        // Canvas click for corrections
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        
        // Correction modal
        document.getElementById('cancelCorrection').addEventListener('click', () => {
            this.correctionModal.style.display = 'none';
            this.correctionMode = false;
        });
        
        // Funscript generation
        this.generateFunscriptBtn.addEventListener('click', () => this.generateFunscript());
    }
    
    connectWebSocket() {
        this.ws = new WebSocket('ws://localhost:8000/ws');
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            // Reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'frame_update':
                this.updateFrame(data);
                break;
            
            case 'processing_complete':
                this.onProcessingComplete();
                break;
            
            case 'correction_saved':
                this.showNotification('Correction saved for ' + data.body_part);
                break;
            
            case 'funscript_generated':
                this.onFunscriptGenerated(data);
                break;
            
            case 'error':
                this.showError(data.message);
                break;
        }
    }
    
    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            await this.uploadFile(file);
        }
    }
    
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/upload_video', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Upload failed');
            }
            
            const data = await response.json();
            this.videoInfo = data;
            this.displayVideoInfo(data);
            this.showPreviewSection();
            
        } catch (error) {
            this.showError('Failed to upload video: ' + error.message);
        }
    }
    
    displayVideoInfo(info) {
        document.getElementById('videoFilename').textContent = info.filename;
        document.getElementById('videoDuration').textContent = info.duration.toFixed(2);
        document.getElementById('videoFPS').textContent = info.fps.toFixed(2);
        document.getElementById('videoFrames').textContent = info.total_frames;
        
        this.videoInfoSection.style.display = 'block';
        
        // Update time display
        this.totalTimeEl.textContent = this.formatTime(info.duration);
        this.progressBar.max = info.total_frames;
    }
    
    showPreviewSection() {
        this.previewSection.style.display = 'block';
        
        // Set canvas size
        this.canvas.width = 1280;
        this.canvas.height = 720;
    }
    
    updateFrame(data) {
        // Update progress
        this.currentFrame = data.frame_number;
        this.progressBar.value = this.currentFrame;
        
        const currentTime = this.currentFrame / this.videoInfo.fps;
        this.currentTimeEl.textContent = this.formatTime(currentTime);
        
        // Draw frame
        const img = new Image();
        img.onload = () => {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
        };
        img.src = 'data:image/jpeg;base64,' + data.frame_data;
        
        // Update progress bar
        if (data.progress !== undefined) {
            const percent = (data.progress * 100).toFixed(1);
            document.getElementById('progressPercent').textContent = percent;
            document.getElementById('progressFill').style.width = percent + '%';
        }
        
        // Update interaction stats
        if (data.interactions && data.interactions.length > 0) {
            this.updateInteractionStats(data.interactions);
        }
    }
    
    updateInteractionStats(interactions) {
        const statsSection = document.getElementById('interactionStats');
        const interactionList = document.getElementById('interactionList');
        
        statsSection.style.display = 'block';
        
        const interactionText = interactions.map(i => {
            return `${i[0]} ↔ ${i[1]} (${i[2].toFixed(0)}px)`;
        }).join(', ');
        
        interactionList.textContent = interactionText;
    }
    
    play() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'resume' }));
            this.isPlaying = true;
            this.playBtn.style.display = 'none';
            this.pauseBtn.style.display = 'block';
        }
    }
    
    pause() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'pause' }));
            this.isPlaying = false;
            this.playBtn.style.display = 'block';
            this.pauseBtn.style.display = 'none';
        }
    }
    
    seek(frame) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ 
                type: 'seek', 
                frame: parseInt(frame) 
            }));
        }
    }
    
    startAnalysis() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN && this.videoInfo) {
            this.ws.send(JSON.stringify({ type: 'start_processing' }));
            this.isAnalyzing = true;
            this.startAnalysisBtn.disabled = true;
            this.startAnalysisBtn.textContent = 'Analyzing...';
            
            document.getElementById('analysisProgress').style.display = 'block';
        }
    }
    
    onProcessingComplete() {
        this.isAnalyzing = false;
        this.startAnalysisBtn.disabled = false;
        this.startAnalysisBtn.textContent = 'Start Analysis';
        
        document.getElementById('funscriptControls').style.display = 'block';
        this.showNotification('Analysis complete!');
    }
    
    toggleCorrectionMode() {
        this.correctionMode = !this.correctionMode;
        this.correctionModeBtn.classList.toggle('active', this.correctionMode);
        
        if (this.correctionMode) {
            this.correctionModal.style.display = 'flex';
        } else {
            this.correctionModal.style.display = 'none';
        }
    }
    
    handleCanvasClick(event) {
        if (!this.correctionMode) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Scale to canvas coordinates
        const canvasX = (x / rect.width) * this.canvas.width;
        const canvasY = (y / rect.height) * this.canvas.height;
        
        const bodyPart = document.getElementById('bodyPartSelect').value;
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'correct_position',
                frame: this.currentFrame,
                body_part: bodyPart,
                x: Math.round(canvasX),
                y: Math.round(canvasY)
            }));
        }
        
        this.correctionModal.style.display = 'none';
        this.correctionMode = false;
    }
    
    generateFunscript() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'generate_funscript' }));
            this.generateFunscriptBtn.disabled = true;
            this.generateFunscriptBtn.textContent = 'Generating...';
        }
    }
    
    onFunscriptGenerated(data) {
        this.generateFunscriptBtn.disabled = false;
        this.generateFunscriptBtn.textContent = 'Generate Funscript';
        
        const downloadSection = document.getElementById('downloadSection');
        const downloadLink = document.getElementById('downloadLink');
        
        downloadSection.style.display = 'block';
        downloadLink.href = `/download_funscript/${data.filename.split('/').pop()}`;
        
        this.showNotification(`Funscript generated with ${data.actions_count} actions!`);
    }
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    showNotification(message) {
        // Simple notification implementation
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #4caf50;
            color: white;
            padding: 15px 25px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.3s ease-out;
            z-index: 1001;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    showError(message) {
        const notification = document.createElement('div');
        notification.className = 'error-notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #f44336;
            color: white;
            padding: 15px 25px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.3s ease-out;
            z-index: 1001;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
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

@keyframes slideOut {
    from {
        transform: translateX(0);
        opacity: 1;
    }
    to {
        transform: translateX(100%);
        opacity: 0;
    }
}

.correction-btn.active {
    background-color: rgba(255, 100, 100, 0.9);
}
`;
document.head.appendChild(style);

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new VRBodyTracker();
});