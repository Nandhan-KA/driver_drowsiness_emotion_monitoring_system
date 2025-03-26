"""
Drowsiness Detection System - All-in-One Application
This script handles setup, database initialization, and runs the application
with optimized video streaming to reduce latency.
"""

import os
import sys
import sqlite3
import threading
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from datetime import datetime
import logging
import subprocess
import platform
from werkzeug.utils import secure_filename
from phone_detection import PhoneDetector
from drowsiness_detection import DrowsinessDetector
from emotion_recognition import EmotionRecognizer
from heart_rate_monitor import HeartRateMonitor
from music_player import MusicPlayer
from sos_alert import SOSAlert
from database import db
import keyboard  # Add keyboard module for key detection
import asyncio
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for video streaming
camera = None
output_frame = None
frame_lock = threading.Lock()
is_monitoring = False
monitoring_thread = None
current_emotion = "unknown"
current_emotion_confidence = 0.0
current_ear = 0
is_drowsy = False
heart_rate = 75  # Initial heart rate
monitoring_mode = None  # 'live' or 'upload'
video_file = None  # Path to uploaded video file

# Upload folder configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize detection modules
drowsiness_detector = DrowsinessDetector()
phone_detector = PhoneDetector()
emotion_recognizer = EmotionRecognizer()
heart_rate_monitor = HeartRateMonitor()
music_player = MusicPlayer()
sos_alert = SOSAlert()

# Update the monitoring_stats structure
monitoring_stats = {
    'ear_values': [],
    'heart_rate_values': [],
    'alert_levels': [],
    'timestamps': []
}
MAX_STATS_POINTS = 60  # Store 1 minute of data at 1 sample per second

# Global variables for real-time updates
ear_values = []
heart_rate_values = []
alert_levels = []
timestamps = []
emotion_distribution = {
    'happy': 0,
    'sad': 0,
    'angry': 0,
    'neutral': 0,
    'surprised': 0,
    'fear': 0,
    'disgust': 0
}

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'cv2', 'dlib', 'numpy', 'tensorflow', 'keras', 'pygame', 
        'flask', 'twilio', 'PIL', 'matplotlib', 'sklearn', 
        'imutils', 'playsound'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install all required packages using: pip install -r requirements.txt")
        return False
    
    logger.info("All required packages are installed.")
    return True

def check_shape_predictor():
    """Check if the shape predictor file exists and download if needed"""
    shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(shape_predictor_path):
        logger.info(f"Shape predictor file not found: {shape_predictor_path}")
        logger.info("Downloading shape predictor file...")
        
        try:
            from download_shape_predictor import download_shape_predictor
            download_shape_predictor()
            return True
        except Exception as e:
            logger.error(f"Error downloading shape predictor: {e}")
            return False
    else:
        logger.info(f"Shape predictor file found: {shape_predictor_path}")
        return True

def check_directories():
    """Check if all required directories exist and create them if needed"""
    directories = [
        'models',
        'static',
        'static/css',
        'static/js',
        'static/music',
        'static/music/happy',
        'static/music/sad',
        'static/music/neutral',
        'templates'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    # Check if template files exist, if not create a basic one
    if not os.path.exists('templates/layout.html') or not os.path.exists('templates/index.html'):
        logger.warning("Template files missing, creating basic templates")
        create_basic_templates()
    
    logger.info("All required directories exist.")
    return True

def create_basic_templates():
    """Create basic template files if they don't exist"""
    # Create layout.html
    if not os.path.exists('templates/layout.html'):
        with open('templates/layout.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% if page_title %}{{ page_title }} - {% endif %}Drowsiness Detection System</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">Drowsiness Detection System</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link {% if request.path == '/' %}active{% endif %}" href="/">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link {% if request.path == '/dashboard' %}active{% endif %}" href="/dashboard">Dashboard</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    {% block content %}{% endblock %}

    <footer class="bg-dark text-white mt-5 py-3">
        <div class="container text-center">
            <p class="mb-0">Drowsiness Detection System &copy; {{ now.year }}</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block scripts %}{% endblock %}
</body>
</html>""")
    
    # Create index.html
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write("""{% extends "layout.html" %}

{% block content %}
<div class="container mt-4">
    <div class="jumbotron">
        <h1 class="display-4">Drowsiness Detection System</h1>
        <p class="lead">A system to detect driver drowsiness, recognize emotions, and provide safety features.</p>
        <hr class="my-4">
        <p>This system uses computer vision to monitor driver alertness and emotional state, providing timely alerts and music recommendations.</p>
        <a class="btn btn-primary btn-lg" href="/dashboard" role="button">Go to Dashboard</a>
    </div>
</div>
{% endblock %}""")
    
    # Create a basic CSS file
    if not os.path.exists('static/css/style.css'):
        os.makedirs('static/css', exist_ok=True)
        with open('static/css/style.css', 'w') as f:
            f.write("""/* Custom styles for Drowsiness Detection System */
body {
    background-color: #f8f9fa;
}
.jumbotron {
    background-color: #e9ecef;
    padding: 2rem 1rem;
    margin-bottom: 2rem;
    border-radius: 0.3rem;
}""")

def fix_database():
    """Fix database issues or create a new database if needed"""
    db_path = os.path.join(os.getcwd(), 'drowsiness_detection.db')
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                alert_type TEXT NOT NULL,
                heart_rate INTEGER,
                location TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Create emotion_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emotion_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                emotion TEXT NOT NULL,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Insert default user if not exists
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        
        if count == 0:
            cursor.execute("""
                INSERT INTO users (name, contact, email)
                VALUES ('Default User', '+1234567890', 'default@example.com')
            """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Database initialized successfully at {db_path}")
        return True
    except Exception as e:
        logger.error(f"Error fixing database: {e}")
        return False

def generate_frames():
    """Generate frames for video streaming with reduced latency"""
    global output_frame
    
    # Create a placeholder frame with a message when no camera feed is available
    placeholder_height, placeholder_width = 480, 640
    placeholder = np.zeros((placeholder_height, placeholder_width, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Camera not available", (int(placeholder_width/4), int(placeholder_height/2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(placeholder, "Please check your webcam", (int(placeholder_width/4), int(placeholder_height/2) + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    while True:
        try:
            # Check if output_frame is available
            if output_frame is None:
                # Use placeholder if no frame is available
                frame_to_show = placeholder.copy()
            else:
                # Acquire the lock
                with frame_lock:
                    if output_frame is None:
                        frame_to_show = placeholder.copy()
                    else:
                        # Make a copy to avoid modifying the frame while it's being processed
                        frame_to_show = output_frame.copy()
            
            # Compress the frame with lower quality to reduce latency
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # Lower quality (0-100)
            _, buffer = cv2.imencode('.jpg', frame_to_show, encode_param)
            
            # Yield the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Add a small delay to control frame rate
            time.sleep(0.03)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            # Use placeholder frame in case of error
            _, buffer = cv2.imencode('.jpg', placeholder, encode_param)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)

def monitoring_loop():
    """Main monitoring loop with optimized processing"""
    global output_frame, is_monitoring, current_emotion, current_emotion_confidence
    global current_ear, is_drowsy, heart_rate, monitoring_mode, video_file, camera
    global monitoring_stats
    
    logger.info(f"Starting monitoring loop in {monitoring_mode} mode")
    
    try:
        # Initialize camera
        if monitoring_mode == 'live':
            camera = cv2.VideoCapture(0)
        else:
            camera = cv2.VideoCapture(video_file)
        
        if not camera.isOpened():
            raise Exception("Failed to open camera/video source")
        
        frame_count = 0
        last_stats_update = time.time()
        last_emotion_check = time.time()
        emotion_check_interval = 1.0  # Check emotion every second
        
        logger.info(f"Monitoring started in {monitoring_mode} mode")
        
        while is_monitoring:
            success, frame = camera.read()
            
            if not success:
                if monitoring_mode == 'upload':
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.error("Failed to capture frame from camera")
                    break
            
            current_time = time.time()
            
            # Process frame for drowsiness detection
            frame, drowsy, ear = drowsiness_detector.detect_drowsiness(frame)
            
            # Update drowsiness status
            is_drowsy = drowsy
            current_ear = ear
            
            # Check emotion periodically
            if current_time - last_emotion_check >= emotion_check_interval:
                # Detect emotion
                emotion, confidence, frame = emotion_recognizer.detect_emotion(frame)
                current_emotion = emotion
                current_emotion_confidence = confidence
                
                # Play appropriate music based on emotion
                if emotion in ['happy', 'sad', 'neutral']:
                    music_player.play_for_emotion(emotion)
                
                last_emotion_check = current_time
            
            # Update statistics once per second
            if current_time - last_stats_update >= 1.0:
                # Update monitoring statistics
                monitoring_stats['timestamps'].append(current_time)
                monitoring_stats['ear_values'].append(ear)
                monitoring_stats['heart_rate_values'].append(heart_rate)
                monitoring_stats['alert_levels'].append(1 if drowsy else 0)
                
                # Keep only the last MAX_STATS_POINTS points
                if len(monitoring_stats['timestamps']) > MAX_STATS_POINTS:
                    monitoring_stats['timestamps'] = monitoring_stats['timestamps'][-MAX_STATS_POINTS:]
                    monitoring_stats['ear_values'] = monitoring_stats['ear_values'][-MAX_STATS_POINTS:]
                    monitoring_stats['heart_rate_values'] = monitoring_stats['heart_rate_values'][-MAX_STATS_POINTS:]
                    monitoring_stats['alert_levels'] = monitoring_stats['alert_levels'][-MAX_STATS_POINTS:]
                
                last_stats_update = current_time
            
            # Send SOS alert if drowsy
            if is_drowsy:
                sos_alert.send_alert(
                    reason="Driver appears drowsy",
                    location=f"Emotion: {current_emotion}, Heart Rate: {heart_rate} BPM"
                )
            
            # Update the output frame
            with frame_lock:
                output_frame = frame.copy()
            
            frame_count += 1
            time.sleep(0.01)  # Small delay to control frame rate
            
    except Exception as e:
        logger.error(f"Error in monitoring loop: {e}")
    finally:
        if camera:
            camera.release()
        heart_rate_monitor.stop()
        music_player.stop()
        emotion_recognizer.cleanup()
        is_monitoring = False
        logger.info("Monitoring stopped")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return jsonify({
                "status": "success",
                "message": "File uploaded successfully",
                "filename": filepath
            })
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            return jsonify({
                "status": "error",
                "message": f"Error saving file: {str(e)}"
            }), 500
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid file type. Allowed types: " + ", ".join(ALLOWED_EXTENSIONS)
        }), 400

@app.route('/status')
def get_status():
    """Get current monitoring status"""
    return jsonify({
        'is_monitoring': is_monitoring,
        'is_drowsy': is_drowsy,
        'current_ear': current_ear,
        'heart_rate': heart_rate,
        'current_emotion': current_emotion,
        'emotion_confidence': current_emotion_confidence,
        'monitoring_mode': monitoring_mode
    })

@app.context_processor
def inject_now():
    """Inject current datetime into templates"""
    return {'now': datetime.now()}

@app.route('/')
def index():
    """Home page route"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index template: {e}")
        return f"Error loading page: {str(e)}", 500

@app.route('/dashboard')
def dashboard():
    """Dashboard page route"""
    try:
        from database import db
        alerts = db.get_recent_alerts()
        emotions = db.get_recent_emotions()
        return render_template('dashboard.html', alerts=alerts, emotions=emotions, page_title="Dashboard")
    except Exception as e:
        logger.error(f"Error rendering dashboard template: {e}")
        return f"Error loading dashboard: {str(e)}", 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route with optimized settings"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    """Get current monitoring statistics with proper data formatting"""
    global monitoring_stats
    
    try:
        # Ensure all arrays have the same length
        min_length = min(len(monitoring_stats['timestamps']),
                        len(monitoring_stats['ear_values']),
                        len(monitoring_stats['heart_rate_values']),
                        len(monitoring_stats['alert_levels']))
        
        return jsonify({
            'timestamps': monitoring_stats['timestamps'][-min_length:],
            'ear_values': monitoring_stats['ear_values'][-min_length:],
            'heart_rate_values': monitoring_stats['heart_rate_values'][-min_length:],
            'alert_levels': monitoring_stats['alert_levels'][-min_length:]
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            'timestamps': [],
            'ear_values': [],
            'heart_rate_values': [],
            'alert_levels': []
        })

def check_webcam_availability():
    """Check if a webcam is available and working"""
    # Try different camera indices and backends
    backends = [
        cv2.CAP_ANY,      # Auto-detect
        cv2.CAP_DSHOW,    # DirectShow (Windows)
        cv2.CAP_MSMF,     # Media Foundation (Windows)
        cv2.CAP_V4L2      # Video for Linux
    ]
    
    for backend in backends:
        for idx in range(3):  # Try first 3 camera indices
            try:
                logger.info(f"Testing camera with index {idx} and backend {backend}")
                camera = cv2.VideoCapture(idx, backend)
                if camera.isOpened():
                    # Try to read a frame
                    success, frame = camera.read()
                    camera.release()
                    
                    if success:
                        logger.info(f"Successfully tested camera with index {idx} and backend {backend}")
                        return True, idx, backend
            except Exception as e:
                logger.warning(f"Failed to test camera with index {idx} and backend {backend}: {e}")
                continue
    
    logger.warning("No working webcam found. The application will run with limited functionality.")
    return False, -1, -1

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start monitoring with specified mode"""
    global is_monitoring, monitoring_mode, video_file
    
    data = request.get_json()
    monitoring_mode = data.get('mode', 'live')
    video_file = data.get('video_file')
    
    if not is_monitoring:
        is_monitoring = True
        thread = threading.Thread(target=monitoring_loop)
        thread.daemon = True
        thread.start()
        return jsonify({"status": "success", "message": "Monitoring started"})
    
    return jsonify({"status": "error", "message": "Monitoring already running"})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop monitoring"""
    global is_monitoring
    is_monitoring = False
    return jsonify({"status": "success", "message": "Monitoring stopped"})

@app.route('/api/alerts')
def api_alerts():
    """Get recent alerts API route"""
    from database import db
    alerts = db.get_recent_alerts()
    return jsonify(alerts)

@app.route('/api/emotions')
def api_emotions():
    """Get recent emotions API route"""
    from database import db
    emotions = db.get_recent_emotions()
    return jsonify(emotions)

@app.route('/api/trigger_sos', methods=['POST'])
def trigger_sos():
    """Trigger SOS alert API route"""
    try:
        from sos_alert import SOSAlert
        from database import db
        
        data = request.get_json() or {}
        alert_type = data.get('alert_type', 'manual')
        details = data.get('details', 'Manually triggered alert')
        location = data.get('location', 'Unknown location')
        
        # Log alert to database
        db.log_alert(
            alert_type=alert_type,
            details=details,
            location=location
        )
        
        # Send SOS alert
        sos = SOSAlert()
        sos.send_sos(
            alert_type=alert_type,
            details=details,
            location=location
        )
        
        return jsonify({"status": "success", "message": "SOS alert sent successfully"})
    except Exception as e:
        logger.error(f"Error triggering SOS alert: {e}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

def run_app():
    """Run the Flask application with optimized settings"""
    
    # Get configuration from config.py
    try:
        import config
        host = config.FLASK_CONFIG.get('HOST', '0.0.0.0')
        port = 5050  # Use port 5050 instead of 5000 to avoid conflicts
        debug = config.FLASK_CONFIG.get('DEBUG', False)
    except Exception as e:
        logger.warning(f"Error loading config: {e}. Using default values.")
        host = '0.0.0.0'
        port = 5050  # Use port 5050 instead of 5000
        debug = False
    
    logger.info(f"Starting Flask server on {host}:{port}")
    # Run the app with threaded=True for better performance
    app.run(host=host, port=port, debug=debug, threaded=True)

# Add a simple test route to verify the server is working
@app.route('/test')
def test():
    """Test route to verify the server is working"""
    return "Drowsiness Detection System is running!"

@app.route('/test-page')
def test_page():
    """Test page route to verify template rendering"""
    try:
        return render_template('test.html')
    except Exception as e:
        logger.error(f"Error rendering test template: {e}")
        return f"Error loading test page: {str(e)}", 500

def kill_process_on_port(port):
    """Kill any process running on the specified port"""
    try:
        system = platform.system()
        
        if system == 'Windows':
            # Windows command to find and kill process on port
            find_cmd = f'netstat -ano | findstr :{port}'
            result = subprocess.check_output(find_cmd, shell=True).decode()
            
            if result:
                # Extract PID from the result
                for line in result.strip().split('\n'):
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.strip().split()
                        pid = parts[-1]
                        logger.info(f"Found process with PID {pid} on port {port}, killing it")
                        
                        # Kill the process
                        kill_cmd = f'taskkill /F /PID {pid}'
                        subprocess.check_output(kill_cmd, shell=True)
                        logger.info(f"Process with PID {pid} killed")
        
        elif system == 'Linux' or system == 'Darwin':  # Linux or macOS
            # Unix command to find and kill process on port
            find_cmd = f"lsof -i :{port} | grep LISTEN | awk '{{print $2}}'"
            result = subprocess.check_output(find_cmd, shell=True).decode()
            
            if result:
                pid = result.strip()
                logger.info(f"Found process with PID {pid} on port {port}, killing it")
                
                # Kill the process
                kill_cmd = f'kill -9 {pid}'
                subprocess.check_output(kill_cmd, shell=True)
                logger.info(f"Process with PID {pid} killed")
        
        logger.info(f"Port {port} is now free")
        return True
    except Exception as e:
        logger.warning(f"Error killing process on port {port}: {e}")
        return False

def cleanup():
    """Cleanup function to properly close resources"""
    global is_monitoring, camera, output_frame
    
    logger.info("Cleaning up resources...")
    
    # Stop monitoring
    is_monitoring = False
    
    # Release camera
    if camera is not None:
        camera.release()
        camera = None
    
    # Clear output frame
    output_frame = None
    
    # Stop other components
    heart_rate_monitor.stop()
    music_player.stop()
    
    logger.info("Cleanup completed")

def main():
    """Main function to run the application"""
    logger.info("Starting Drowsiness Detection System...")
    
    # Kill any existing process on port 5050
    kill_process_on_port(5050)
    
    # Check requirements
    if not check_requirements():
        logger.error("Missing required packages. Please install them and try again.")
        return False
    
    # Check shape predictor
    if not check_shape_predictor():
        logger.error("Failed to set up shape predictor. Please download it manually.")
        return False
    
    # Check directories and templates
    check_directories()
    
    # Fix database
    if not fix_database():
        logger.error("Failed to set up database. Please check the error messages.")
        return False
    
    try:
        # Register Ctrl+Q handler
        keyboard.on_press_key('q', lambda _: handle_ctrl_q())
        
        # Run the application
        logger.info("All checks passed. Starting application...")
        run_app()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Cleaning up...")
        cleanup()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        cleanup()
        sys.exit(1)
    finally:
        cleanup()
    
    return True

def handle_ctrl_q():
    """Handle Ctrl+Q key press"""
    if keyboard.is_pressed('ctrl'):
        logger.info("Ctrl+Q pressed. Stopping application...")
        cleanup()
        sys.exit(0)

if __name__ == "__main__":
    main() 