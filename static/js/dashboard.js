// Dashboard JavaScript for Drowsiness Detection System

$(document).ready(function() {
    // Load alerts
    function loadAlerts() {
        $.getJSON('/api/alerts', function(data) {
            var alertsHtml = '';
            
            if (data.length === 0) {
                alertsHtml = '<tr><td colspan="3" class="text-center">No alerts found</td></tr>';
            } else {
                $.each(data, function(index, alert) {
                    var alertType = alert.alert_type.charAt(0).toUpperCase() + alert.alert_type.slice(1);
                    var timestamp = new Date(alert.timestamp).toLocaleString();
                    
                    alertsHtml += '<tr>';
                    alertsHtml += '<td>' + timestamp + '</td>';
                    alertsHtml += '<td>' + alertType + '</td>';
                    alertsHtml += '<td>' + alert.details + '</td>';
                    alertsHtml += '</tr>';
                });
            }
            
            $('#alerts-table').html(alertsHtml);
        });
    }
    
    // Load emotions
    function loadEmotions() {
        $.getJSON('/api/emotions', function(data) {
            var emotionsHtml = '';
            
            if (data.length === 0) {
                emotionsHtml = '<tr><td colspan="3" class="text-center">No emotion data found</td></tr>';
            } else {
                $.each(data, function(index, emotion) {
                    var emotionName = emotion.emotion.charAt(0).toUpperCase() + emotion.emotion.slice(1);
                    var timestamp = new Date(emotion.timestamp).toLocaleString();
                    var confidence = (emotion.confidence * 100).toFixed(2) + '%';
                    
                    emotionsHtml += '<tr>';
                    emotionsHtml += '<td>' + timestamp + '</td>';
                    emotionsHtml += '<td>' + emotionName + '</td>';
                    emotionsHtml += '<td>' + confidence + '</td>';
                    emotionsHtml += '</tr>';
                });
            }
            
            $('#emotions-table').html(emotionsHtml);
        });
    }
    
    // Update status
    function updateStatus() {
        $.getJSON('/api/status', function(data) {
            $('#heart-rate').text(data.heart_rate + ' BPM');
            $('#emotion').text(data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1));
            $('#current-song').text(data.current_song);
            
            if (data.is_monitoring) {
                $('#monitoring-status').text('Running').removeClass('badge-secondary').addClass('badge-success');
            } else {
                $('#monitoring-status').text('Stopped').removeClass('badge-success').addClass('badge-secondary');
            }
            
            if (data.is_drowsy) {
                $('#drowsiness-status').text('Drowsy').removeClass('badge-success').addClass('badge-danger');
            } else {
                $('#drowsiness-status').text('Alert').removeClass('badge-danger').addClass('badge-success');
            }
            
            // Update heart rate color based on value
            var heartRate = data.heart_rate;
            if (heartRate < 50 || heartRate > 120) {
                $('#heart-rate').addClass('text-danger').removeClass('text-success');
            } else {
                $('#heart-rate').addClass('text-success').removeClass('text-danger');
            }
        });
    }
    
    // Handle SOS form submission
    $('#sos-form').submit(function(e) {
        e.preventDefault();
        
        $.post('/api/trigger_sos', $(this).serialize(), function(data) {
            alert('Test SOS alert sent!');
            loadAlerts();
        });
    });
    
    // Load data periodically
    setInterval(loadAlerts, 10000);
    setInterval(loadEmotions, 10000);
    setInterval(updateStatus, 2000);
    
    // Initial data load
    loadAlerts();
    loadEmotions();
    updateStatus();

    // Initialize variables
    let monitoring = false;
    let statsChart = null;

    // Initialize the statistics chart
    function initializeStatsChart() {
        const ctx = document.getElementById('statsChart').getContext('2d');
        statsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Eye Aspect Ratio',
                        data: [],
                        borderColor: 'rgba(54, 162, 235, 1)',
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Heart Rate',
                        data: [],
                        borderColor: 'rgba(255, 99, 132, 1)',
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Alert Level',
                        data: [],
                        borderColor: 'rgba(75, 192, 192, 1)',
                        fill: false,
                        tension: 0.4,
                        stepped: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'second',
                            displayFormats: {
                                second: 'HH:mm:ss'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Value'
                        }
                    }
                },
                animation: {
                    duration: 0
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }

    // Update monitoring statistics
    function updateMonitoringStats(data) {
        if (!statsChart) return;

        // Update labels and data
        statsChart.data.labels = data.timestamps;
        statsChart.data.datasets[0].data = data.ear_values;
        statsChart.data.datasets[1].data = data.heart_rate_values;
        statsChart.data.datasets[2].data = data.alert_levels;

        // Update chart
        statsChart.update('none'); // Use 'none' mode for better performance
    }

    // Update emotion distribution
    function updateEmotionDistribution(data) {
        if (!statsChart) return;

        // Update emotion distribution data
        statsChart.data.datasets[0].data = data.emotion_distribution.map((value, index) => ({
            x: data.timestamps[index],
            y: value
        }));
        statsChart.data.datasets[1].data = data.heart_rate_values.map((value, index) => ({
            x: data.timestamps[index],
            y: value
        }));
        statsChart.data.datasets[2].data = data.alert_levels.map((value, index) => ({
            x: data.timestamps[index],
            y: value
        }));

        // Update chart
        statsChart.update('none'); // Use 'none' mode for better performance
    }

    // Update system status
    function updateSystemStatus(data) {
        // Update metrics
        document.getElementById('ear-value').textContent = data.current_ear?.toFixed(2) || '0.00';
        document.getElementById('emotion-value').textContent = data.current_emotion || 'Neutral';
        document.getElementById('heart-rate').textContent = data.heart_rate || '0';

        // Update drowsiness status
        const statusIndicator = document.querySelector('.status-indicator');
        if (statusIndicator) {
            if (data.is_drowsy) {
                statusIndicator.classList.remove('status-inactive');
                statusIndicator.classList.add('status-active');
                document.getElementById('drowsiness-status').innerHTML = `
                    <span class="status-indicator status-active"></span>
                    Drowsy
                `;
            } else {
                statusIndicator.classList.remove('status-active');
                statusIndicator.classList.add('status-inactive');
                document.getElementById('drowsiness-status').innerHTML = `
                    <span class="status-indicator status-inactive"></span>
                    Alert
                `;
            }
        }

        // Update system status badges
        const monitoringStatus = document.getElementById('monitoring-status');
        const cameraStatus = document.getElementById('camera-status');
        const alertStatus = document.getElementById('alert-status');

        if (monitoringStatus) {
            monitoringStatus.textContent = data.is_monitoring ? 'Active' : 'Inactive';
            monitoringStatus.className = `status-badge ${data.is_monitoring ? 'status-active' : 'status-inactive'}`;
        }

        if (cameraStatus) {
            cameraStatus.textContent = data.camera_active ? 'Active' : 'Inactive';
            cameraStatus.className = `status-badge ${data.camera_active ? 'status-active' : 'status-inactive'}`;
        }

        if (alertStatus) {
            alertStatus.textContent = data.alert_system_active ? 'Active' : 'Inactive';
            alertStatus.className = `status-badge ${data.alert_system_active ? 'status-active' : 'status-inactive'}`;
        }
    }

    // Add alert to history
    function addAlertToHistory(message) {
        const alertHistory = document.getElementById('alert-history');
        const alertItem = document.createElement('div');
        alertItem.className = 'alert-item';
        alertItem.innerHTML = `
            <div class="alert-message">${message}</div>
            <div class="alert-timestamp">${new Date().toLocaleTimeString()}</div>
        `;
        alertHistory.insertBefore(alertItem, alertHistory.firstChild);

        // Keep only last 10 alerts
        while (alertHistory.children.length > 10) {
            alertHistory.removeChild(alertHistory.lastChild);
        }
    }

    // Update current time
    function updateTime() {
        const timeElement = document.getElementById('current-time');
        if (timeElement) {
            const now = new Date();
            const options = {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            };
            timeElement.textContent = now.toLocaleDateString('en-US', options);
        }
    }

    // Fetch and update status
    async function updateStatus() {
        try {
            const response = await fetch('/status');
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            // Update monitoring statistics
            updateMonitoringStats(data);

            // Update emotion distribution
            updateEmotionDistribution(data);

            // Update system status
            updateSystemStatus(data);

            // Add to alert history if drowsy
            if (data.is_drowsy) {
                addAlertToHistory('Drowsiness detected');
            }

        } catch (error) {
            console.error('Error updating status:', error);
            // Don't show alerts for every error to avoid spamming the user
            // Only update monitoring status to show connection issue
            const monitoringStatus = document.getElementById('monitoring-status');
            if (monitoringStatus) {
                monitoringStatus.textContent = 'Connection Error';
                monitoringStatus.className = 'status-badge status-inactive';
            }
        }
    }

    // Initialize dashboard
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize chart
        initializeStatsChart();

        // Live Monitoring Button
        document.getElementById('liveMonitoringBtn').addEventListener('click', async () => {
            try {
                const response = await fetch('/start_monitoring', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ mode: 'live' })
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    monitoring = true;
                    updateButtonStates();
                } else {
                    alert('Error starting monitoring: ' + data.message);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error starting monitoring');
            }
        });

        // Upload Video Button
        document.getElementById('uploadVideoBtn').addEventListener('click', () => {
            document.getElementById('videoFileInput').click();
        });

        // File Input Change
        document.getElementById('videoFileInput').addEventListener('change', async (event) => {
            const file = event.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('video', file);

            try {
                const uploadResponse = await fetch('/upload_video', {
                    method: 'POST',
                    body: formData
                });
                const uploadData = await uploadResponse.json();

                if (uploadData.status === 'success') {
                    const monitorResponse = await fetch('/start_monitoring', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            mode: 'upload',
                            video_file: uploadData.filename
                        })
                    });
                    const monitorData = await monitorResponse.json();

                    if (monitorData.status === 'success') {
                        monitoring = true;
                        updateButtonStates();
                    } else {
                        alert('Error starting monitoring: ' + monitorData.message);
                    }
                } else {
                    alert('Error uploading video: ' + uploadData.message);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing video');
            }
        });

        // Stop Monitoring Button
        document.getElementById('stopMonitoringBtn').addEventListener('click', async () => {
            try {
                const response = await fetch('/stop_monitoring', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'success') {
                    monitoring = false;
                    updateButtonStates();
                } else {
                    alert('Error stopping monitoring: ' + data.message);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error stopping monitoring');
            }
        });

        // Start stats update interval
        setInterval(updateStatus, 1000);

        // Update time immediately and set interval
        updateTime();
        setInterval(updateTime, 1000);
    });

    // Initialize button states
    function updateButtonStates() {
        const liveBtn = document.getElementById('liveMonitoringBtn');
        const uploadBtn = document.getElementById('uploadVideoBtn');
        const stopBtn = document.getElementById('stopMonitoringBtn');
        const fileInput = document.getElementById('videoFileInput');

        if (monitoring) {
            liveBtn.disabled = true;
            uploadBtn.disabled = true;
            stopBtn.disabled = false;
        } else {
            liveBtn.disabled = false;
            uploadBtn.disabled = false;
            stopBtn.disabled = true;
        }
    }
}); 