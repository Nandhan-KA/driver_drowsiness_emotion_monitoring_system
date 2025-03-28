# Drowsiness Detection and Emotion-Based Music Recommendation System with SOS Alert

## Overview
This project aims to enhance driver safety by:
- Detecting drowsiness using eye tracking
- Identifying driver emotions through facial expressions
- Recommending music based on the driver's mood
- Triggering SOS alerts in case of health emergencies

## Features
- Real-time drowsiness detection using Eye Aspect Ratio (EAR)
- Emotion recognition using CNN model
- Heart rate monitoring (simulated) with emergency alerts
- Emotion-based music recommendation
- SOS alerts via SMS & Email

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Webcam
- Internet connection (for SOS alerts)

### Installation
1. Clone this repository
```
git clone https://github.com/Nandhan-KA/driver_drowsiness_emotion_monitoring_system.git
cd driver_drowsiness_emotion_monitoring_system
```

2. Install required packages
```
pip install -r requirements.txt
```

3. Run the application
```
python drowsiness_detection.py
```

The script will automatically:
- Check if all required packages are installed
- Download the shape predictor file if it doesn't exist
- Set up the SQLite database
- Start the Flask web server


### Accessing the Application
Access the web interface at: http://127.0.0.1:5050

## Project Structure
- `app.py`: Main Flask application
- `drowsiness_detection.py`: Drowsiness detection module
- `emotion_recognition.py`: Emotion recognition module
- `heart_rate_monitor.py`: Simulated heart rate monitoring
- `music_player.py`: Music recommendation and playback
- `sos_alert.py`: Emergency alert system
- `database.py`: Database operations (SQLite)
- `config.py`: Configuration settings
- `download_shape_predictor.py`: Script to download the shape predictor file
- `run.py`: Script to run the application with proper setup
- `static/`: CSS, JS, and music files
- `templates/`: HTML templates for web interface
- `models/`: Pre-trained emotion recognition model

