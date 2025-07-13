// VR Body Analyzer - Main Application JavaScript

let socket;
let currentJobId = null;
let detectionChart = null;
let currentResults = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeSocketIO();
    initializeFileUpload();
    initializeDragAndDrop();
});

// Initialize Socket.IO connection
function initializeSocketIO() {
    socket = io();
    
    socket.on('connect', function() {
        console.log('Connected to server');
    });
    
    socket.on('processing_update', function(data) {
        handleProcessingUpdate(data);
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
    });
}

// Initialize file upload functionality
function initializeFileUpload() {
    const fileInput = document.getElementById('videoFile');
    
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            validateAndUploadFile(file);
        }
    });
}

// Initialize drag and drop functionality
function initializeDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');
    
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            validateAndUploadFile(files[0]);
        }
    });
    
    uploadArea.addEventListener('click', function() {
        document.getElementById('videoFile').click();
    });
}

// Validate and upload file
function validateAndUploadFile(file) {
    const supportedFormats = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!supportedFormats.includes(fileExtension)) {
        showAlert('Invalid file format. Please upload a video file.', 'danger');
        return;
    }
    
    const maxSize = 500 * 1024 * 1024; // 500MB
    if (file.size > maxSize) {
        showAlert('File size too large. Please upload a file smaller than 500MB.', 'warning');
        return;
    }
    
    uploadFile(file);
}

// Upload file to server
function uploadFile(file) {
    const formData = new FormData();
    formData.append('video', file);
    
    showUploadProgress();
    
    const xhr = new XMLHttpRequest();
    
    // Track upload progress
    xhr.upload.addEventListener('progress', function(e) {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            updateProgressBar(percentComplete);
        }
    });
    
    xhr.addEventListener('load', function() {
        if (xhr.status === 200) {
            const response = JSON.parse(xhr.responseText);
            currentJobId = response.job_id;
            showProcessingArea();
            updateUploadStatus('Upload complete, processing started...');
        } else {
            const errorResponse = JSON.parse(xhr.responseText);
            showAlert(errorResponse.error || 'Upload failed', 'danger');
            hideUploadProgress();
        }
    });
    
    xhr.addEventListener('error', function() {
        showAlert('Network error during upload', 'danger');
        hideUploadProgress();
    });
    
    xhr.open('POST', '/upload');
    xhr.send(formData);
}

// Handle processing updates from server
function handleProcessingUpdate(data) {
    if (data.job_id !== currentJobId) return;
    
    switch (data.status) {
        case 'started':
            updateProcessingStatus(data.message);
            break;
        case 'completed':
            hideProcessingArea();
            fetchAndDisplayResults(data.job_id);
            if (data.summary) {
                displaySummaryStats(data.summary);
            }
            break;
        case 'error':
            hideProcessingArea();
            showAlert(data.message, 'danger');
            break;
    }
}

// Fetch and display results
async function fetchAndDisplayResults(jobId) {
    try {
        const response = await fetch(`/results/${jobId}`);
        const data = await response.json();
        
        if (response.ok) {
            currentResults = data.results;
            displayResults(data.results);
            showResultsCard();
        } else {
            showAlert(data.error || 'Failed to fetch results', 'danger');
        }
    } catch (error) {
        showAlert('Error fetching results: ' + error.message, 'danger');
    }
}

// Display summary statistics
function displaySummaryStats(summary) {
    const statsContainer = document.getElementById('summaryStats');
    
    statsContainer.innerHTML = `
        <div class="col-md-3">
            <div class="stat-item text-center">
                <h3 class="text-primary">${summary.total_frames}</h3>
                <small class="text-muted">Total Frames</small>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stat-item text-center">
                <h3 class="text-success">${summary.detected_frames}</h3>
                <small class="text-muted">Detected Frames</small>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stat-item text-center">
                <h3 class="text-info">${summary.average_confidence}</h3>
                <small class="text-muted">Avg Confidence</small>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stat-item text-center">
                <h3 class="text-warning">${summary.detection_rate}%</h3>
                <small class="text-muted">Detection Rate</small>
            </div>
        </div>
    `;
}

// Display detailed results
function displayResults(results) {
    const tableBody = document.getElementById('resultsTable');
    
    tableBody.innerHTML = '';
    
    results.forEach((result, index) => {
        const row = document.createElement('tr');
        
        const bodyPartsDetected = Object.keys(result.body_parts || {}).filter(
            part => result.body_parts[part].length > 0
        );
        
        row.innerHTML = `
            <td>${result.frame_idx}</td>
            <td>${result.timestamp ? result.timestamp.toFixed(2) : 0}s</td>
            <td>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar ${getConfidenceColor(result.confidence)}" 
                         style="width: ${(result.confidence * 100).toFixed(1)}%">
                        ${(result.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            </td>
            <td>
                ${bodyPartsDetected.map(part => 
                    `<span class="badge bg-primary me-1">${part}</span>`
                ).join('')}
            </td>
            <td>
                <button class="btn btn-sm btn-outline-primary" 
                        onclick="visualizeBodyParts(${index})">
                    <i class="fas fa-eye"></i> View
                </button>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // Create detection timeline chart
    createDetectionChart(results);
}

// Get confidence color based on value
function getConfidenceColor(confidence) {
    if (confidence > 0.8) return 'bg-success';
    if (confidence > 0.6) return 'bg-warning';
    return 'bg-danger';
}

// Create detection timeline chart
function createDetectionChart(results) {
    const ctx = document.getElementById('detectionChart').getContext('2d');
    
    if (detectionChart) {
        detectionChart.destroy();
    }
    
    const labels = results.map(r => r.frame_idx);
    const confidenceData = results.map(r => r.confidence);
    
    detectionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Detection Confidence',
                data: confidenceData,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Body Part Detection Confidence Over Time'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Confidence'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Frame Number'
                    }
                }
            }
        }
    });
}

// Visualize body parts for a specific frame
function visualizeBodyParts(frameIndex) {
    if (!currentResults || frameIndex >= currentResults.length) return;
    
    const result = currentResults[frameIndex];
    const modal = new bootstrap.Modal(document.getElementById('bodyPartsModal'));
    
    // Update modal content
    document.querySelector('#bodyPartsModal .modal-title').textContent = 
        `Frame ${result.frame_idx} - Body Parts Visualization`;
    
    // Draw body parts on canvas
    drawBodyPartsVisualization(result);
    
    // Update body parts list
    updateBodyPartsList(result.body_parts);
    
    modal.show();
}

// Draw body parts visualization on canvas
function drawBodyPartsVisualization(result) {
    const canvas = document.getElementById('bodyPartsCanvas');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw stick figure representation
    if (result.landmarks && result.landmarks.length > 0) {
        const landmarks = result.landmarks;
        
        // Scale landmarks to canvas size
        const scaleX = canvas.width;
        const scaleY = canvas.height;
        
        // Draw landmarks
        ctx.fillStyle = '#28a745';
        landmarks.forEach(landmark => {
            if (landmark.visibility > 0.5) {
                const x = landmark.x * scaleX;
                const y = landmark.y * scaleY;
                
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, 2 * Math.PI);
                ctx.fill();
            }
        });
        
        // Draw connections
        const connections = [
            [11, 12], [11, 13], [12, 14], [13, 15], [14, 16],  // Arms
            [11, 23], [12, 24], [23, 24],  // Torso
            [23, 25], [24, 26], [25, 27], [26, 28]  // Legs
        ];
        
        ctx.strokeStyle = '#007bff';
        ctx.lineWidth = 2;
        
        connections.forEach(([start, end]) => {
            if (start < landmarks.length && end < landmarks.length) {
                const startLandmark = landmarks[start];
                const endLandmark = landmarks[end];
                
                if (startLandmark.visibility > 0.5 && endLandmark.visibility > 0.5) {
                    ctx.beginPath();
                    ctx.moveTo(startLandmark.x * scaleX, startLandmark.y * scaleY);
                    ctx.lineTo(endLandmark.x * scaleX, endLandmark.y * scaleY);
                    ctx.stroke();
                }
            }
        });
    } else {
        // No landmarks detected
        ctx.fillStyle = '#6c757d';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No body parts detected', canvas.width / 2, canvas.height / 2);
    }
}

// Update body parts list
function updateBodyPartsList(bodyParts) {
    const container = document.getElementById('bodyPartsList');
    container.innerHTML = '';
    
    if (!bodyParts || Object.keys(bodyParts).length === 0) {
        container.innerHTML = '<div class="col-12"><p class="text-muted">No body parts detected</p></div>';
        return;
    }
    
    Object.entries(bodyParts).forEach(([partName, landmarks]) => {
        if (landmarks && landmarks.length > 0) {
            const col = document.createElement('div');
            col.className = 'col-md-6 mb-2';
            
            const avgConfidence = landmarks.reduce((sum, lm) => 
                sum + (lm.visibility || 0), 0) / landmarks.length;
            
            col.innerHTML = `
                <div class="card">
                    <div class="card-body p-2">
                        <h6 class="card-title">${partName.replace('_', ' ').toUpperCase()}</h6>
                        <div class="progress" style="height: 15px;">
                            <div class="progress-bar ${getConfidenceColor(avgConfidence)}" 
                                 style="width: ${(avgConfidence * 100).toFixed(1)}%">
                                ${(avgConfidence * 100).toFixed(1)}%
                            </div>
                        </div>
                        <small class="text-muted">${landmarks.length} landmarks</small>
                    </div>
                </div>
            `;
            
            container.appendChild(col);
        }
    });
}

// Download results as JSON
function downloadResults() {
    if (!currentResults) {
        showAlert('No results to download', 'warning');
        return;
    }
    
    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `vr_body_analysis_${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
}

// Export results as CSV
function exportCSV() {
    if (!currentResults) {
        showAlert('No results to export', 'warning');
        return;
    }
    
    const headers = ['Frame', 'Timestamp', 'Confidence', 'Body_Parts_Count', 'VR_Format', 'Eye'];
    const rows = currentResults.map(result => [
        result.frame_idx,
        result.timestamp ? result.timestamp.toFixed(2) : 0,
        result.confidence.toFixed(3),
        Object.keys(result.body_parts || {}).length,
        result.vr_format || 'unknown',
        result.eye || 'unknown'
    ]);
    
    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const dataBlob = new Blob([csvContent], {type: 'text/csv'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `vr_body_analysis_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
}

// UI Helper Functions
function showUploadProgress() {
    document.getElementById('uploadProgress').style.display = 'block';
    updateUploadStatus('Uploading...');
}

function hideUploadProgress() {
    document.getElementById('uploadProgress').style.display = 'none';
}

function updateProgressBar(percentage) {
    const progressBar = document.getElementById('progressBar');
    progressBar.style.width = percentage + '%';
    progressBar.setAttribute('aria-valuenow', percentage);
}

function updateUploadStatus(status) {
    document.getElementById('uploadStatus').textContent = status;
}

function showProcessingArea() {
    hideUploadProgress();
    document.getElementById('processingArea').style.display = 'block';
}

function hideProcessingArea() {
    document.getElementById('processingArea').style.display = 'none';
}

function updateProcessingStatus(status) {
    document.getElementById('processingStatus').textContent = status;
}

function showResultsCard() {
    document.getElementById('resultsCard').style.display = 'block';
    // Smooth scroll to results
    document.getElementById('resultsCard').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

function showAlert(message, type = 'info') {
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // Insert alert at the top of the container
    const container = document.querySelector('.container');
    container.insertAdjacentHTML('afterbegin', alertHtml);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = container.querySelector('.alert');
        if (alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }
    }, 5000);
}