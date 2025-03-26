import torch
import cv2
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO
from typing import Tuple, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhoneDetector:
    def __init__(self):
        """Initialize phone detection with improved parameters"""
        try:
            # Load YOLOv5 model with better parameters
            self.model = YOLO('yolov5s.pt')
            
            # Improved confidence thresholds
            self.conf_threshold = 0.5  # Higher confidence threshold for better accuracy
            self.iou_threshold = 0.45  # Adjusted IOU threshold
            
            # Phone detection parameters
            self.phone_classes = [67]  # YOLO class index for mobile phone
            self.min_phone_size = 50   # Minimum phone size in pixels
            
            # Performance optimization
            self.last_detection_time = 0
            self.detection_cooldown = 0.1  # Seconds between detections
            self.frame_buffer = None
            self.last_detection = None
            
            # Tracking parameters
            self.tracker = None
            self.tracking_failed = False
            self.tracking_failures = 0
            self.max_tracking_failures = 3
            
            logger.info("PhoneDetector initialized with improved parameters")
            
        except Exception as e:
            logger.error(f"Error initializing phone detector: {e}")
            self.model = None
    
    def detect_phone(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        Enhanced phone detection with tracking and performance optimization
        Returns: (processed_frame, phone_detected, confidence)
        """
        if frame is None:
            logger.error("Received empty frame")
            return None, False, 0.0

        try:
            current_time = time.time()
            
            # Try tracking first if we have a tracker
            if self.tracker is not None and not self.tracking_failed:
                phone_detected, confidence, bbox = self._track_phone(frame)
                if phone_detected:
                    return self._draw_detection(frame, bbox, confidence), True, confidence
            
            # If tracking failed or not initialized, run detection
            if current_time - self.last_detection_time >= self.detection_cooldown:
                phone_detected, confidence, bbox = self._detect_phone(frame)
                if phone_detected:
                    # Initialize tracker with detected phone
                    self.tracker = cv2.TrackerCSRT_create()
                    self.tracker.init(frame, bbox)
                    self.tracking_failed = False
                    self.tracking_failures = 0
                    self.last_detection_time = current_time
                    return self._draw_detection(frame, bbox, confidence), True, confidence
            
            return frame, False, 0.0
            
        except Exception as e:
            logger.error(f"Error in phone detection: {e}")
            return frame, False, 0.0

    def _detect_phone(self, frame: np.ndarray) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
        """Run YOLO detection for phones"""
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Check if it's a phone
                    if int(box.cls) in self.phone_classes:
                        # Get confidence and bounding box
                        confidence = float(box.conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Check minimum size
                        if (x2 - x1) >= self.min_phone_size and (y2 - y1) >= self.min_phone_size:
                            return True, confidence, (x1, y1, x2, y2)
            
            return False, 0.0, None
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return False, 0.0, None

    def _track_phone(self, frame: np.ndarray) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
        """Track previously detected phone"""
        try:
            # Update tracker
            success, bbox = self.tracker.update(frame)
            
            if success:
                # Convert bbox to tuple
                x, y, w, h = map(int, bbox)
                return True, 0.8, (x, y, x + w, y + h)
            else:
                self.tracking_failures += 1
                if self.tracking_failures >= self.max_tracking_failures:
                    self.tracking_failed = True
                    self.tracker = None
                return False, 0.0, None
                
        except Exception as e:
            logger.error(f"Error in phone tracking: {e}")
            self.tracking_failed = True
            self.tracker = None
            return False, 0.0, None

    def _draw_detection(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], confidence: float) -> np.ndarray:
        """Draw detection results on frame"""
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw confidence
        label = f"Phone: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw warning text
        warning = "WARNING: Phone Usage Detected!"
        cv2.putText(frame, warning, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def start(self, video_source=0):
        """
        Start phone detection
        
        Args:
            video_source: The video source (0 for webcam)
        """
        cap = cv2.VideoCapture(video_source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect phone
            frame, phone_detected, confidence = self.detect_phone(frame)
            
            # Display the frame
            cv2.imshow("Phone Detection", frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

# For testing
if __name__ == "__main__":
    detector = PhoneDetector()
    detector.start() 