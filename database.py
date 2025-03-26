"""
Database operations for the Drowsiness Detection System
Uses SQLite3 instead of MySQL for easier setup
"""

import sqlite3
from datetime import datetime
import os
import config

class Database:
    """Database class for handling all database operations using SQLite3"""
    
    def __init__(self):
        """Initialize database connection"""
        self.connection = None
        # Use absolute path for database file
        self.db_path = os.path.join(os.getcwd(), 'drowsiness_detection.db')
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            # Connect to SQLite database
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Return rows as dictionaries
            
            print(f"Connected to SQLite database at {self.db_path}")
        except Exception as e:
            print(f"Error connecting to SQLite: {e}")
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.connection:
            self.connect()
            
        if not self.connection:
            print("Failed to create database connection. Tables cannot be created.")
            return
        
        try:
            cursor = self.connection.cursor()
            
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
            
            self.connection.commit()
            print("Tables created successfully")
            
            # Insert default user if not exists
            cursor.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            
            if count == 0:
                cursor.execute("""
                    INSERT INTO users (name, contact, email)
                    VALUES ('Default User', '+1234567890', 'default@example.com')
                """)
                self.connection.commit()
                print("Default user created")
                
            cursor.close()
            
        except Exception as e:
            print(f"Error creating tables: {e}")
    
    def log_alert(self, user_id=1, alert_type='drowsiness', heart_rate=None, location=None, details=None):
        """Log an alert in the database"""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO alerts (user_id, alert_type, heart_rate, location, details)
                VALUES (?, ?, ?, ?, ?)
            """
            cursor.execute(query, (user_id, alert_type, heart_rate, location, details))
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error logging alert: {e}")
            return False
    
    def log_emotion(self, user_id=1, emotion='neutral', confidence=0.0):
        """Log detected emotion in the database"""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO emotion_logs (user_id, emotion, confidence)
                VALUES (?, ?, ?)
            """
            cursor.execute(query, (user_id, emotion, confidence))
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error logging emotion: {e}")
            return False
    
    def get_recent_alerts(self, limit=10):
        """Get recent alerts from the database"""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            query = """
                SELECT a.*, u.name 
                FROM alerts a
                JOIN users u ON a.user_id = u.user_id
                ORDER BY a.timestamp DESC
                LIMIT ?
            """
            cursor.execute(query, (limit,))
            alerts = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            return alerts
        except Exception as e:
            print(f"Error getting alerts: {e}")
            return []
    
    def get_recent_emotions(self, limit=10):
        """Get recent emotion logs from the database"""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            query = """
                SELECT e.*, u.name 
                FROM emotion_logs e
                JOIN users u ON e.user_id = u.user_id
                ORDER BY e.timestamp DESC
                LIMIT ?
            """
            cursor.execute(query, (limit,))
            emotions = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            return emotions
        except Exception as e:
            print(f"Error getting emotions: {e}")
            return []
    
    def close(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()
            print("SQLite connection closed")

# Singleton instance
db = Database() 