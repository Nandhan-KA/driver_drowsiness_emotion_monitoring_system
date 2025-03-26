"""
Heart Rate Monitor Module for the Drowsiness Detection System
Simulates heart rate monitoring and triggers alerts for abnormal values
"""

import random
import time
import threading
from database import db
import config

class HeartRateMonitor:
    """Class for simulating heart rate monitoring"""
    
    def __init__(self):
        """Initialize the heart rate monitor"""
        self.heart_rate = 75  # Initial heart rate (normal)
        self.is_running = False
        self.monitor_thread = None
        
        # Heart rate thresholds
        self.low_threshold = config.HEART_RATE_PARAMS['LOW_THRESHOLD']
        self.high_threshold = config.HEART_RATE_PARAMS['HIGH_THRESHOLD']
        self.check_interval = config.HEART_RATE_PARAMS['CHECK_INTERVAL']
        
        print("Heart rate monitor initialized")
    
    def generate_heart_rate(self):
        """
        Generate a simulated heart rate
        
        Returns:
            int: Simulated heart rate
        """
        # Simulate heart rate with some random variation
        variation = random.randint(-5, 5)
        
        # Add variation to current heart rate
        self.heart_rate += variation
        
        # Ensure heart rate stays within realistic bounds
        if self.heart_rate < 40:
            self.heart_rate = 40
        elif self.heart_rate > 180:
            self.heart_rate = 180
        
        # Occasionally simulate abnormal heart rates for testing
        if random.random() < 0.05:  # 5% chance of abnormal heart rate
            if random.random() < 0.5:
                # Simulate low heart rate
                self.heart_rate = random.randint(30, self.low_threshold - 1)
            else:
                # Simulate high heart rate
                self.heart_rate = random.randint(self.high_threshold + 1, 180)
        
        return self.heart_rate
    
    def check_heart_rate(self):
        """
        Check if heart rate is abnormal and trigger alert if necessary
        
        Returns:
            tuple: (is_abnormal, status)
        """
        if self.heart_rate < self.low_threshold:
            return True, "low"
        elif self.heart_rate > self.high_threshold:
            return True, "high"
        else:
            return False, "normal"
    
    def trigger_alert(self, status):
        """
        Trigger heart rate alert
        
        Args:
            status: The heart rate status (low/high)
        """
        # Log alert to database
        details = f"Abnormal heart rate detected: {self.heart_rate} BPM ({status})"
        db.log_alert(
            alert_type='heart_rate',
            heart_rate=self.heart_rate,
            details=details
        )
        
        print(f"ALERT: {details}")
    
    def monitor(self):
        """Monitor heart rate continuously"""
        while self.is_running:
            # Generate heart rate
            self.generate_heart_rate()
            
            # Check if heart rate is abnormal
            is_abnormal, status = self.check_heart_rate()
            
            if is_abnormal:
                self.trigger_alert(status)
            
            # Sleep for check interval
            time.sleep(self.check_interval)
    
    def start(self):
        """Start heart rate monitoring"""
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self.monitor)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("Heart rate monitoring started")
    
    def stop(self):
        """Stop heart rate monitoring"""
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        print("Heart rate monitoring stopped")
    
    def get_heart_rate(self):
        """
        Get current heart rate
        
        Returns:
            int: Current heart rate
        """
        return self.heart_rate

# For testing
if __name__ == "__main__":
    monitor = HeartRateMonitor()
    monitor.start()
    
    try:
        while True:
            hr = monitor.get_heart_rate()
            is_abnormal, status = monitor.check_heart_rate()
            print(f"Heart Rate: {hr} BPM - Status: {status}")
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("Monitoring stopped") 