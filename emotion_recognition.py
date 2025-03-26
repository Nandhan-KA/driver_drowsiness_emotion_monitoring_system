"""
Emotion Recognition Module for the Drowsiness Detection System
Uses a CNN model to detect emotions from facial expressions
"""

import cv2
import numpy as np
import tensorflow as tf
import threading
import queue
import logging
import os
import time
import config
from database import db
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionRecognizer:
    """Class for recognizing emotions from facial expressions"""
    
    def __init__(self):
        """Initialize emotion recognition system"""
        self.model = None
        self.emotions = ['happy', 'sad', 'angry']  # Focus on three emotions
        self.current_emotion = 'neutral'
        self.confidence = 0.0
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.frame_queue = queue.Queue(maxsize=1)  # Reduced queue size
        self.result_queue = queue.Queue(maxsize=1)  # Reduced queue size
        self.is_running = False
        self.processing_thread = None
        self.last_emotion_time = time.time()
        self.emotion_cooldown = 0.5  # Minimum time between emotion updates
        self._load_emotion_model()
        self._start_processing_thread()
        logger.info("EmotionRecognizer initialized")

    def _load_emotion_model(self):
        """Load the pre-trained emotion recognition model"""
        try:
            model_path = 'models/emotion_model.h5'
            if not os.path.exists(model_path):
                logger.info("Downloading emotion recognition model...")
                # Download model from URL
                import urllib.request
                url = "https://github.com/atulapra/Emotion-detection/raw/master/model.h5"
                urllib.request.urlretrieve(url, model_path)
            
            self.model = tf.keras.models.load_model(model_path)
            logger.info("Emotion recognition model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading emotion model: {str(e)}")
            raise
    
    def _start_processing_thread(self):
        """Start the emotion processing thread"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_frames(self):
        """Process frames in a separate thread"""
        while self.is_running:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=0.1)  # Reduced timeout
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # Process each face
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (48, 48))
                    face = face.astype('float32') / 255.0
                    face = np.expand_dims(face, axis=[0, -1])
                    
                    # Predict emotion
                    predictions = self.model.predict(face, verbose=0)  # Disable verbose output
                    emotion_idx = np.argmax(predictions[0])
                    confidence = predictions[0][emotion_idx]
                    
                    # Map the model's output to our three emotions
                    if emotion_idx == 3:  # Happy
                        mapped_idx = 0
                    elif emotion_idx == 4:  # Sad
                        mapped_idx = 1
                    elif emotion_idx == 0:  # Angry
                        mapped_idx = 2
                    else:
                        continue  # Skip other emotions
                    
                    # Update current emotion and confidence
                    self.current_emotion = self.emotions[mapped_idx]
                    self.confidence = float(confidence)
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    label = f"{self.current_emotion} ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                # Put processed frame in result queue
                if not self.result_queue.full():
                    self.result_queue.put(frame)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                continue
    
    def detect_emotion(self, frame):
        """Detect emotion in the given frame"""
        try:
            current_time = time.time()
            
            # Check if enough time has passed since last emotion update
            if current_time - self.last_emotion_time < self.emotion_cooldown:
                return self.current_emotion, self.confidence, frame
            
            # Put frame in processing queue if not full
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            
            # Get processed frame from result queue
            try:
                processed_frame = self.result_queue.get_nowait()
                self.last_emotion_time = current_time
                return self.current_emotion, self.confidence, processed_frame
            except queue.Empty:
                return self.current_emotion, self.confidence, frame
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return self.current_emotion, self.confidence, frame
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Emotion recognizer cleaned up")

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize emotion recognizer
    recognizer = EmotionRecognizer()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            emotion, confidence, processed_frame = recognizer.detect_emotion(frame)
            print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
            
            cv2.imshow('Emotion Detection Test', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.cleanup() 