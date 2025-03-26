"""
Drowsiness Detection Module for the Drowsiness Detection System
Uses Eye Aspect Ratio (EAR) to detect drowsiness
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import threading
import config
from database import db
from playsound import playsound
import os
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DrowsinessDetector:
    """Class for detecting drowsiness using eye aspect ratio"""
    
    def __init__(self):
        """Initialize drowsiness detection system"""
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.EAR_THRESHOLD = 0.25  # Increased threshold to be less sensitive
        self.EAR_FRAMES = 30  # Number of consecutive frames to check
        self.ear_history = []
        self.last_blink_time = time.time()
        self.blink_cooldown = 1.0  # Minimum time between blinks (seconds)
        self.drowsy_start_time = None
        self.DROWSY_THRESHOLD = 2.0  # Seconds of continuous low EAR to trigger drowsiness
        logger.info("DrowsinessDetector initialized")

    def calculate_ear(self, eye):
        """Calculate the eye aspect ratio"""
        # Vertical distances
        v1 = np.linalg.norm(eye[1] - eye[5])
        v2 = np.linalg.norm(eye[2] - eye[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye[0] - eye[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_drowsiness(self, frame):
        """Detect drowsiness in the given frame"""
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray, 0)
            
            if len(faces) == 0:
                return frame, False, 0.0
            
            # Get the first face
            face = faces[0]
            
            # Get facial landmarks
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            # Get eye landmarks
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            
            # Average EAR
            ear = (left_ear + right_ear) / 2.0
            
            # Update EAR history
            self.ear_history.append(ear)
            if len(self.ear_history) > self.EAR_FRAMES:
                self.ear_history.pop(0)
            
            # Check for drowsiness
            current_time = time.time()
            
            # If EAR is below threshold
            if ear < self.EAR_THRESHOLD:
                # Check if enough time has passed since last blink
                if current_time - self.last_blink_time > self.blink_cooldown:
                    if self.drowsy_start_time is None:
                        self.drowsy_start_time = current_time
                    
                    # Check if drowsy for long enough
                    if current_time - self.drowsy_start_time >= self.DROWSY_THRESHOLD:
                        is_drowsy = True
                    else:
                        is_drowsy = False
                else:
                    is_drowsy = False
                    self.drowsy_start_time = None
            else:
                is_drowsy = False
                self.drowsy_start_time = None
                self.last_blink_time = current_time
            
            # Draw face rectangle and landmarks
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw eye landmarks
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
            # Add EAR value and status text
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            status = "Drowsy" if is_drowsy else "Alert"
            color = (0, 0, 255) if is_drowsy else (0, 255, 0)
            cv2.putText(frame, f"Status: {status}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return frame, is_drowsy, ear
            
        except Exception as e:
            logger.error(f"Error in drowsiness detection: {str(e)}")
            return frame, False, 0.0

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'predictor'):
                del self.predictor
            if hasattr(self, 'detector'):
                del self.detector
            logger.info("DrowsinessDetector cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up DrowsinessDetector: {str(e)}")

    def calculate_mar(self, mouth_points):
        """
        Calculate the Mouth Aspect Ratio (MAR)
        
        Args:
            mouth_points: Points of the mouth landmarks
            
        Returns:
            float: The MAR value
        """
        # Calculate vertical distances
        v1 = dist.euclidean(mouth_points[2], mouth_points[10])
        v2 = dist.euclidean(mouth_points[4], mouth_points[8])
        v3 = dist.euclidean(mouth_points[0], mouth_points[6])
        
        # Calculate MAR
        mar = (v1 + v2) / (2.0 * v3)
        
        return mar
    
    def get_landmarks(self, frame, face):
        """
        Get facial landmarks for a detected face
        
        Args:
            frame: The frame containing the face
            face: The detected face rectangle (x, y, w, h)
            
        Returns:
            dlib.full_object_detection: Facial landmarks
        """
        try:
            # Convert face rectangle to dlib rectangle
            x, y, w, h = face
            dlib_rect = dlib.rectangle(x, y, x + w, y + h)
            
            # Get landmarks
            landmarks = self.predictor(frame, dlib_rect)
            return landmarks
            
        except Exception as e:
            logger.error(f"Error getting landmarks: {e}")
            return None
    
    def get_eye_landmarks(self, landmarks, start, end):
        """Get eye landmarks as numpy array"""
        points = []
        for n in range(start, end):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append([x, y])
        return np.array(points, dtype=np.int32)

    def _get_face_location(self, gray: np.ndarray) -> Optional[dlib.rectangle]:
        """
        Get face location using improved detection parameters
        
        Args:
            gray: Grayscale image
            
        Returns:
            Optional[dlib.rectangle]: Detected face rectangle or None
        """
        try:
            # Detect faces with improved parameters
            faces = self.detector(gray, 1)  # 1 means upscale the image once
            
            if len(faces) == 0:
                return None
            
            # Get the largest face (usually the closest one)
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Convert dlib rectangle to (x, y, w, h) format
            x = face.left()
            y = face.top()
            w = face.width()
            h = face.height()
            
            # Add some padding around the face
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2 * padding)
            h = min(gray.shape[0] - y, h + 2 * padding)
            
            return (x, y, w, h)
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return None

    def _calculate_ear(self, landmarks: dlib.full_object_detection) -> float:
        """Calculate Eye Aspect Ratio with improved accuracy"""
        try:
            # Left eye points (36-41)
            left_eye = np.array([(landmarks.part(36 + i).x, landmarks.part(36 + i).y) for i in range(6)])
            # Right eye points (42-47)
            right_eye = np.array([(landmarks.part(42 + i).x, landmarks.part(42 + i).y) for i in range(6)])
            
            # Calculate EAR for both eyes
            left_ear = self._calculate_ear_for_eye(left_eye)
            right_ear = self._calculate_ear_for_eye(right_eye)
            
            # Return average EAR
            return (left_ear + right_ear) / 2.0
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.0

    def _calculate_ear_for_eye(self, eye_points: np.ndarray) -> float:
        """Calculate EAR for a single eye with improved accuracy"""
        # Calculate vertical distances
        v1 = dist.euclidean(eye_points[1], eye_points[5])
        v2 = dist.euclidean(eye_points[2], eye_points[4])
        
        # Calculate horizontal distance
        h = dist.euclidean(eye_points[0], eye_points[3])
        
        # Calculate EAR with improved formula
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def _calculate_mar(self, landmarks: dlib.full_object_detection) -> float:
        """Calculate Mouth Aspect Ratio for yawn detection"""
        try:
            # Mouth points (48-68)
            mouth_points = np.array([(landmarks.part(48 + i).x, landmarks.part(48 + i).y) for i in range(20)])
            
            # Calculate vertical distances
            v1 = dist.euclidean(mouth_points[2], mouth_points[10])
            v2 = dist.euclidean(mouth_points[4], mouth_points[8])
            
            # Calculate horizontal distance
            h = dist.euclidean(mouth_points[0], mouth_points[6])
            
            # Calculate MAR
            mar = (v1 + v2) / (2.0 * h)
            return mar
        except Exception as e:
            logger.error(f"Error calculating MAR: {e}")
            return 0.0

    def _estimate_head_pose(self, landmarks: dlib.full_object_detection) -> Optional[Tuple[float, float, float]]:
        """Estimate head pose using facial landmarks"""
        current_time = time.time()
        if current_time - self.last_head_pose_time < self.head_pose_cooldown:
            return None
        
        # Get 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # Get 2D image points
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),       # Chin
            (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),     # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
        ], dtype="double")
        
        # Camera parameters
        size = (640, 480)
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        # Distortion coefficients
        dist_coeffs = np.zeros((4,1))
        
        # Solve PnP
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        
        if success:
            # Convert rotation vector to angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pose_matrix = np.hstack((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                np.hstack((pose_matrix, np.array([[0], [0], [0], [1]])))
            )
            
            self.last_head_pose_time = current_time
            return tuple(euler_angles.flatten())
        
        return None

    def _check_drowsiness(self, ear: float, mar: float, head_pose: Optional[Tuple[float, float, float]]) -> bool:
        """Check for drowsiness using multiple indicators"""
        current_time = time.time()
        
        # Check eye closure
        if ear < self.EAR_THRESHOLD:
            self.FRAME_COUNTER += 1
            if self.FRAME_COUNTER >= self.CONSECUTIVE_FRAMES:
                return True
        else:
            self.FRAME_COUNTER = 0
        
        # Check yawning
        if mar > self.YAWN_THRESHOLD:
            if current_time - self.last_yawn_time >= self.yawn_cooldown:
                self.last_yawn_time = current_time
                return True
        
        # Check head pose if available
        if head_pose is not None:
            pitch, yaw, roll = head_pose
            if abs(pitch) > 20 or abs(yaw) > 20 or abs(roll) > 20:
                return True
        
        # Check blink rate
        if current_time - self.last_blink_time >= 1.0:
            if self.blink_counter > self.blink_rate_threshold:
                return True
            self.blink_counter = 0
            self.last_blink_time = current_time
        
        return False

    def _draw_visual_feedback(self, frame: np.ndarray, landmarks: dlib.full_object_detection,
                            ear: float, mar: float, head_pose: Optional[Tuple[float, float, float]],
                            is_drowsy: bool) -> None:
        """Draw visual feedback on the frame with improved visibility"""
        # Draw face rectangle
        face_rect = landmarks.rect
        cv2.rectangle(frame, 
                     (face_rect.left(), face_rect.top()),
                     (face_rect.right(), face_rect.bottom()),
                     (0, 255, 0), 2)
        
        # Draw facial landmarks with improved visibility
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # Draw larger dots for better visibility
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            # Add white border for better contrast
            cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)
        
        # Draw metrics with improved visibility
        metrics = [
            (f"EAR: {ear:.2f}", 30),
            (f"MAR: {mar:.2f}", 60),
            (f"Status: {'DROWSY!' if is_drowsy else 'ALERT'}", 90)
        ]
        
        for text, y_pos in metrics:
            # Add black background for better readability
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (10, y_pos - 25), (10 + text_width, y_pos + 5), (0, 0, 0), -1)
            # Draw text
            color = (0, 0, 255) if is_drowsy and "Status" in text else (0, 255, 0)
            cv2.putText(frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw head pose if available
        if head_pose is not None:
            pitch, yaw, roll = head_pose
            pose_text = [
                (f"Pitch: {pitch:.1f}", 120),
                (f"Yaw: {yaw:.1f}", 150),
                (f"Roll: {roll:.1f}", 180)
            ]
            
            for text, y_pos in pose_text:
                # Add black background
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (10, y_pos - 25), (10 + text_width, y_pos + 5), (0, 0, 0), -1)
                # Draw text
                cv2.putText(frame, text, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _draw_no_face_warning(self, frame: np.ndarray) -> None:
        """Draw warning when no face is detected"""
        height, width = frame.shape[:2]
        cv2.putText(frame, "No face detected!", (width//4, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Please position your face in front of the camera",
                    (width//4, height//2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def trigger_alert(self):
        """Trigger drowsiness alert"""
        # Log alert to database
        db.log_alert(alert_type='drowsiness', details='Driver drowsiness detected')
        
        # Play alert sound in a separate thread
        if self.alert_thread is None or not self.alert_thread.is_alive():
            self.alert_thread = threading.Thread(target=self.play_alert_sound)
            self.alert_thread.daemon = True
            self.alert_thread.start()
    
    def play_alert_sound(self):
        """Play alert sound"""
        try:
            playsound(self.alert_sound)
        except Exception as e:
            print(f"Error playing alert sound: {e}")
    
    def start(self, video_source=0):
        """
        Start drowsiness detection
        
        Args:
            video_source: The video source (0 for webcam)
        """
        self.is_running = True
        cap = cv2.VideoCapture(video_source)
        
        while self.is_running:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect drowsiness
            frame, drowsy, ear = self.detect_drowsiness(frame)
            
            # Display the frame
            cv2.imshow("Drowsiness Detection", frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
    
    def stop(self):
        """Stop drowsiness detection"""
        self.is_running = False

# For testing
if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.start() 