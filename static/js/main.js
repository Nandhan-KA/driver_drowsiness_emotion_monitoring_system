// Main JavaScript for Drowsiness Detection System

document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on a page with monitoring controls
    const startLiveBtn = document.getElementById('start-live-btn');
    const startUploadBtn = document.getElementById('start-upload-btn');
    const stopBtn = document.getElementById('stop-btn');
    const fileInput = document.getElementById('video-upload');
    
    if (startLiveBtn && startUploadBtn && stopBtn) {
        // Start live monitoring
        startLiveBtn.addEventListener('click', function() {
            fetch('/api/start', { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ mode: 'live' })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    updateStatus();
                } else if (data.status === 'warning') {
                    showNotification(data.message, 'warning');
                    updateStatus();
                } else {
                    showNotification(data.message, 'error');
                }
            })
            .catch(error => {
                console.error('Error starting live monitoring:', error);
                showNotification('Error starting live monitoring. Please try again.', 'error');
            });
        });
        
        // Start upload monitoring
        startUploadBtn.addEventListener('click', function() {
            if (!fileInput.files.length) {
                showNotification('Please select a video file first.', 'warning');
                return;
            }
            
            const formData = new FormData();
            formData.append('video', fileInput.files[0]);
            
            fetch('/api/start', { 
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    updateStatus();
                } else if (data.status === 'warning') {
                    showNotification(data.message, 'warning');
                    updateStatus();
                } else {
                    showNotification(data.message, 'error');
                }
            })
            .catch(error => {
                console.error('Error starting upload monitoring:', error);
                showNotification('Error starting upload monitoring. Please try again.', 'error');
            });
        });
        
        // Stop monitoring
        stopBtn.addEventListener('click', function() {
            fetch('/api/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        updateStatus();
                    } else {
                        showNotification(data.message, 'error');
                    }
                })
                .catch(error => {
                    console.error('Error stopping monitoring:', error);
                    showNotification('Error stopping monitoring. Please try again.', 'error');
                });
        });
        
        // Function to show notification
        function showNotification(message, type) {
            // Create notification element
            const notification = document.createElement('div');
            notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
            notification.role = 'alert';
            notification.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            
            // Find or create notification container
            let container = document.getElementById('notification-container');
            if (!container) {
                container = document.createElement('div');
                container.id = 'notification-container';
                container.style.position = 'fixed';
                container.style.top = '20px';
                container.style.right = '20px';
                container.style.zIndex = '1050';
                container.style.maxWidth = '350px';
                document.body.appendChild(container);
            }
            
            // Add notification to container
            container.appendChild(notification);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => {
                    notification.remove();
                }, 300);
            }, 5000);
        }
        
        // Update status function
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update monitoring status
                    const monitoringStatus = document.getElementById('monitoring-status');
                    if (monitoringStatus) {
                        monitoringStatus.textContent = data.is_monitoring ? 'Running' : 'Stopped';
                        monitoringStatus.className = data.is_monitoring ? 'badge bg-success' : 'badge bg-secondary';
                    }
                    
                    // Update heart rate
                    const heartRate = document.getElementById('heart-rate');
                    if (heartRate) {
                        heartRate.textContent = data.heart_rate + ' BPM';
                    }
                    
                    // Update emotion
                    const emotion = document.getElementById('emotion');
                    if (emotion) {
                        emotion.textContent = data.emotion;
                    }
                    
                    // Update EAR value
                    const earValue = document.getElementById('ear-value');
                    if (earValue) {
                        earValue.textContent = data.ear.toFixed(2);
                    }
                    
                    // Update current song
                    const currentSong = document.getElementById('current-song');
                    if (currentSong) {
                        currentSong.textContent = data.current_song || 'None';
                    }
                    
                    // Update button states
                    if (startLiveBtn && startUploadBtn && stopBtn) {
                        startLiveBtn.disabled = data.is_monitoring;
                        startUploadBtn.disabled = data.is_monitoring;
                        stopBtn.disabled = !data.is_monitoring;
                    }
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        }
        
        // Update status every 2 seconds
        setInterval(updateStatus, 2000);
        
        // Initial status update
        updateStatus();
    }
}); 