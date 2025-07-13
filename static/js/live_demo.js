// VR Body Analyzer - Live Demo JavaScript

let isDetectionActive = false;
let detectionSettings = {
    confidenceThreshold: 0.5,
    mode: 'auto',
    showLandmarks: true,
    showConnections: true
};
let fpsCounter = 0;
let lastFrameTime = Date.now();
let isRecording = false;
let recordedFrames = [];

// Initialize live demo
document.addEventListener('DOMContentLoaded', function() {
    initializeControls();
    startFPSCounter();
    updateDetectionStats();
});

// Initialize controls and event listeners
function initializeControls() {
    // Video feed error handling
    const videoFeed = document.getElementById('videoFeed');
    videoFeed.addEventListener('error', function() {
        showNotification('Camera connection failed', 'error');
    });
    
    videoFeed.addEventListener('load', function() {
        showNotification('Camera connected successfully', 'success');
    });
}

// Start detection
function startDetection() {
    isDetectionActive = true;
    
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'inline-block';
    
    showNotification('Body part detection started', 'success');
    
    // Simulate detection activity
    updateActiveBodyParts(['head', 'torso', 'left_arm', 'right_arm']);
    updateConfidence(0.85);
}

// Stop detection
function stopDetection() {
    isDetectionActive = false;
    
    document.getElementById('startBtn').style.display = 'inline-block';
    document.getElementById('stopBtn').style.display = 'none';
    
    showNotification('Body part detection stopped', 'info');
    
    // Reset stats
    updateActiveBodyParts([]);
    updateConfidence(0);
}

// Update confidence threshold
function updateThreshold(value) {
    detectionSettings.confidenceThreshold = parseFloat(value);
    document.getElementById('thresholdValue').textContent = value;
    
    // Apply new threshold if detection is active
    if (isDetectionActive) {
        showNotification(`Confidence threshold updated to ${value}`, 'info');
    }
}

// Change detection mode
function changeDetectionMode() {
    const mode = document.getElementById('detectionMode').value;
    detectionSettings.mode = mode;
    
    let modeText = '';
    switch(mode) {
        case 'tensorrt':
            modeText = 'TensorRT (GPU) mode enabled';
            break;
        case 'mediapipe':
            modeText = 'MediaPipe (CPU) mode enabled';
            break;
        case 'auto':
            modeText = 'Auto-selection mode enabled';
            break;
    }
    
    showNotification(modeText, 'info');
}

// Toggle landmarks display
function toggleLandmarks() {
    detectionSettings.showLandmarks = document.getElementById('showLandmarks').checked;
    showNotification(
        `Landmarks ${detectionSettings.showLandmarks ? 'enabled' : 'disabled'}`, 
        'info'
    );
}

// Toggle connections display
function toggleConnections() {
    detectionSettings.showConnections = document.getElementById('showConnections').checked;
    showNotification(
        `Connections ${detectionSettings.showConnections ? 'enabled' : 'disabled'}`, 
        'info'
    );
}

// Capture current frame
function captureFrame() {
    if (!isDetectionActive) {
        showNotification('Start detection first to capture frames', 'warning');
        return;
    }
    
    // Simulate frame capture
    const capturedData = {
        timestamp: new Date().toISOString(),
        confidence: Math.random() * 0.4 + 0.6, // Random confidence 0.6-1.0
        landmarks: generateMockLandmarks(),
        bodyParts: ['head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
    };
    
    displayCapturedFrame(capturedData);
    
    const modal = new bootstrap.Modal(document.getElementById('captureModal'));
    modal.show();
    
    showNotification('Frame captured successfully', 'success');
}

// Display captured frame in modal
function displayCapturedFrame(data) {
    const canvas = document.getElementById('capturedFrame');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = 640;
    canvas.height = 480;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw mock body representation
    drawMockBody(ctx, data.landmarks);
    
    // Update landmarks list
    updateLandmarksList(data);
}

// Draw mock body on canvas
function drawMockBody(ctx, landmarks) {
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    // Draw border
    ctx.strokeStyle = '#dee2e6';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    if (!landmarks || landmarks.length === 0) {
        // No detection
        ctx.fillStyle = '#6c757d';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No body detected', ctx.canvas.width / 2, ctx.canvas.height / 2);
        return;
    }
    
    // Draw landmarks
    ctx.fillStyle = '#28a745';
    landmarks.forEach(landmark => {
        const x = landmark.x * ctx.canvas.width;
        const y = landmark.y * ctx.canvas.height;
        
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();
    });
    
    // Draw connections
    ctx.strokeStyle = '#007bff';
    ctx.lineWidth = 3;
    
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Head
        [5, 6], [5, 7], [6, 8], // Arms
        [9, 10], [10, 11], [11, 12] // Legs
    ];
    
    connections.forEach(([start, end]) => {
        if (start < landmarks.length && end < landmarks.length) {
            const startPoint = landmarks[start];
            const endPoint = landmarks[end];
            
            ctx.beginPath();
            ctx.moveTo(startPoint.x * ctx.canvas.width, startPoint.y * ctx.canvas.height);
            ctx.lineTo(endPoint.x * ctx.canvas.width, endPoint.y * ctx.canvas.height);
            ctx.stroke();
        }
    });
}

// Update landmarks list in capture modal
function updateLandmarksList(data) {
    const container = document.getElementById('landmarksList');
    
    let html = `
        <p><strong>Capture Time:</strong> ${new Date(data.timestamp).toLocaleTimeString()}</p>
        <p><strong>Overall Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
        <p><strong>Body Parts Detected:</strong></p>
        <ul class="list-unstyled">
    `;
    
    data.bodyParts.forEach(part => {
        const confidence = Math.random() * 0.3 + 0.7; // Random confidence
        html += `
            <li class="mb-1">
                <span class="badge bg-primary me-2">${part.replace('_', ' ')}</span>
                <small>${(confidence * 100).toFixed(1)}%</small>
            </li>
        `;
    });
    
    html += '</ul>';
    container.innerHTML = html;
}

// Save captured frame
function saveCapturedFrame() {
    const canvas = document.getElementById('capturedFrame');
    
    // Convert canvas to blob and download
    canvas.toBlob(function(blob) {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `captured_frame_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.png`;
        link.click();
        
        showNotification('Frame saved successfully', 'success');
    });
}

// Start recording
function startRecording() {
    if (!isDetectionActive) {
        showNotification('Start detection first to begin recording', 'warning');
        return;
    }
    
    isRecording = !isRecording;
    const button = event.target;
    
    if (isRecording) {
        button.innerHTML = '<i class="fas fa-stop me-2"></i>Stop Recording';
        button.className = 'btn btn-outline-danger btn-sm w-100 mb-2';
        recordedFrames = [];
        showNotification('Recording started', 'success');
    } else {
        button.innerHTML = '<i class="fas fa-record-vinyl me-2"></i>Start Recording';
        button.className = 'btn btn-outline-primary btn-sm w-100 mb-2';
        showNotification(`Recording stopped. ${recordedFrames.length} frames recorded`, 'info');
    }
}

// Export landmarks data
function exportLandmarks() {
    if (!isDetectionActive) {
        showNotification('No landmark data to export', 'warning');
        return;
    }
    
    // Generate mock landmark data
    const landmarkData = {
        timestamp: new Date().toISOString(),
        detectionSettings: detectionSettings,
        landmarks: generateMockLandmarks(),
        summary: {
            totalLandmarks: 33,
            visibleLandmarks: 28,
            averageConfidence: 0.82
        }
    };
    
    const dataStr = JSON.stringify(landmarkData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `landmarks_${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    
    showNotification('Landmarks exported successfully', 'success');
}

// Generate analysis report
function generateReport() {
    const report = {
        sessionInfo: {
            timestamp: new Date().toISOString(),
            duration: isDetectionActive ? '5 minutes' : '0 minutes',
            detectionMode: detectionSettings.mode,
            confidenceThreshold: detectionSettings.confidenceThreshold
        },
        statistics: {
            framesProcessed: isDetectionActive ? 1500 : 0,
            averageFPS: fpsCounter,
            detectionRate: isDetectionActive ? 94.2 : 0,
            averageConfidence: isDetectionActive ? 0.84 : 0
        },
        bodyPartsDetected: {
            head: isDetectionActive ? 98.5 : 0,
            torso: isDetectionActive ? 96.8 : 0,
            arms: isDetectionActive ? 89.2 : 0,
            legs: isDetectionActive ? 87.6 : 0
        },
        recommendations: [
            'Ensure adequate lighting for better detection',
            'Maintain distance of 1-2 meters from camera',
            'Use contrasting background for optimal results'
        ]
    };
    
    const dataStr = JSON.stringify(report, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `analysis_report_${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    
    showNotification('Analysis report generated', 'success');
}

// Update active body parts display
function updateActiveBodyParts(parts) {
    const container = document.getElementById('activeBodyParts');
    
    const allParts = ['head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg'];
    
    container.innerHTML = allParts.map(part => {
        const isActive = parts.includes(part);
        const badgeClass = isActive ? 'bg-success' : 'bg-secondary';
        return `<span class="badge ${badgeClass} me-1">${part.replace('_', ' ')}</span>`;
    }).join('');
}

// Update confidence display
function updateConfidence(confidence) {
    const percentage = Math.round(confidence * 100);
    
    document.getElementById('confidenceBar').style.width = percentage + '%';
    document.getElementById('confidenceText').textContent = percentage + '%';
    
    // Update bar color based on confidence
    const bar = document.getElementById('confidenceBar');
    bar.className = 'progress-bar';
    
    if (confidence > 0.8) {
        bar.classList.add('bg-success');
    } else if (confidence > 0.6) {
        bar.classList.add('bg-warning');
    } else {
        bar.classList.add('bg-danger');
    }
}

// Start FPS counter
function startFPSCounter() {
    setInterval(function() {
        if (isDetectionActive) {
            // Simulate realistic FPS
            fpsCounter = Math.floor(Math.random() * 5) + 25; // 25-30 FPS
            document.getElementById('processingFps').textContent = fpsCounter;
            document.getElementById('fpsCounter').textContent = `FPS: ${fpsCounter}`;
            
            // Update confidence randomly during detection
            const newConfidence = Math.random() * 0.3 + 0.6; // 0.6-0.9
            updateConfidence(newConfidence);
            
            // Randomly update active body parts
            if (Math.random() > 0.7) {
                const randomParts = ['head', 'torso', 'left_arm', 'right_arm']
                    .filter(() => Math.random() > 0.3);
                updateActiveBodyParts(randomParts);
            }
        } else {
            document.getElementById('processingFps').textContent = '--';
            document.getElementById('fpsCounter').textContent = 'FPS: --';
        }
    }, 1000);
}

// Update detection stats periodically
function updateDetectionStats() {
    setInterval(function() {
        if (isDetectionActive && isRecording) {
            recordedFrames.push({
                timestamp: Date.now(),
                confidence: Math.random() * 0.4 + 0.6
            });
        }
    }, 100);
}

// Generate mock landmarks for testing
function generateMockLandmarks() {
    const landmarks = [];
    
    // Generate 33 pose landmarks (MediaPipe format)
    for (let i = 0; i < 33; i++) {
        landmarks.push({
            x: Math.random() * 0.6 + 0.2, // 0.2 to 0.8
            y: Math.random() * 0.6 + 0.2, // 0.2 to 0.8
            z: Math.random() * 0.1 - 0.05, // -0.05 to 0.05
            visibility: Math.random() * 0.4 + 0.6 // 0.6 to 1.0
        });
    }
    
    return landmarks;
}

// Show notification
function showNotification(message, type = 'info') {
    const alertClass = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    }[type] || 'alert-info';
    
    const alertHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show position-fixed" 
             style="top: 20px; right: 20px; z-index: 9999; min-width: 300px;" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', alertHtml);
    
    // Auto-dismiss after 3 seconds
    setTimeout(() => {
        const alerts = document.querySelectorAll('.alert');
        const lastAlert = alerts[alerts.length - 1];
        if (lastAlert) {
            const bsAlert = new bootstrap.Alert(lastAlert);
            bsAlert.close();
        }
    }, 3000);
}