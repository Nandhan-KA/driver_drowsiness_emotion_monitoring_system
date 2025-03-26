"""
Configuration settings for the Drowsiness Detection System
"""

# Database Configuration (SQLite)
DB_CONFIG = {
    'database': 'drowsiness_detection.db'
}

# Twilio Configuration for SMS alerts
TWILIO_CONFIG = {
    'account_sid': 'your_account_sid',
    'auth_token': 'your_auth_token',
    'from_number': '+1234567890'
}

# Email Configuration for email alerts
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': 'your_email@gmail.com',
    'password': 'your_app_password'
}

# Emergency Contact Information
EMERGENCY_CONTACTS = [
    {
        'name': 'Emergency Contact 1',
        'phone': '+1234567890',
        'email': 'emergency1@example.com'
    },
    {
        'name': 'Emergency Contact 2',
        'phone': '+0987654321',
        'email': 'emergency2@example.com'
    }
]

# Drowsiness Detection Parameters
DROWSINESS_PARAMS = {
    'EAR_THRESHOLD': 0.25,  # Eye Aspect Ratio threshold
    'CONSECUTIVE_FRAMES': 20,  # Number of consecutive frames to trigger alert
    'BLINK_RATIO_THRESHOLD': 0.8  # Threshold for blink ratio
}

# Heart Rate Parameters
HEART_RATE_PARAMS = {
    'LOW_THRESHOLD': 50,  # BPM - below this is considered low
    'HIGH_THRESHOLD': 120,  # BPM - above this is considered high
    'CHECK_INTERVAL': 10  # Check heart rate every X seconds
}

# Paths
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
EMOTION_MODEL_PATH = 'models/emotion_model.h5'

# Music Folders
MUSIC_FOLDERS = {
    'happy': 'static/music/happy',
    'sad': 'static/music/sad',
    'neutral': 'static/music/neutral'
}

# Flask App Settings
FLASK_CONFIG = {
    'SECRET_KEY': 'your_secret_key_here',
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000
} 