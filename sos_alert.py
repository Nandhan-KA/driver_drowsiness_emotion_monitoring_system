"""
SOS Alert Module for the Drowsiness Detection System
Sends emergency alerts via SMS and email
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
import config
import threading
import logging
import time
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SOSAlert:
    """Class for sending SOS alerts"""
    
    def __init__(self):
        """Initialize the SOS Alert system"""
        self.config_file = 'config/sos_config.json'
        self.contacts = []
        self.twilio_client = None
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 minutes between alerts
        self._load_config()
        self._setup_twilio()
        
        print("SOS alert system initialized")
    
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            os.makedirs('config', exist_ok=True)
            if not os.path.exists(self.config_file):
                default_config = {
                    'emergency_contacts': [],
                    'twilio': {
                        'account_sid': '',
                        'auth_token': '',
                        'from_number': ''
                    }
                }
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=4)
                logger.info("Created default SOS config file")
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.contacts = config.get('emergency_contacts', [])
                self.twilio_config = config.get('twilio', {})
        except Exception as e:
            logger.error(f"Error loading SOS config: {e}")
            self.contacts = []
            self.twilio_config = {}
    
    def _setup_twilio(self):
        """Setup Twilio client for SMS alerts"""
        try:
            account_sid = self.twilio_config.get('account_sid')
            auth_token = self.twilio_config.get('auth_token')
            
            if account_sid and auth_token:
                self.twilio_client = Client(account_sid, auth_token)
                logger.info("Twilio client initialized")
            else:
                logger.warning("Twilio credentials not configured")
        except Exception as e:
            logger.error(f"Error setting up Twilio: {e}")
            self.twilio_client = None
    
    def add_contact(self, name, phone):
        """Add an emergency contact"""
        try:
            contact = {'name': name, 'phone': phone}
            if contact not in self.contacts:
                self.contacts.append(contact)
                self._save_config()
                logger.info(f"Added emergency contact: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding contact: {e}")
            return False
    
    def remove_contact(self, phone):
        """Remove an emergency contact"""
        try:
            self.contacts = [c for c in self.contacts if c['phone'] != phone]
            self._save_config()
            logger.info(f"Removed emergency contact: {phone}")
            return True
        except Exception as e:
            logger.error(f"Error removing contact: {e}")
            return False
    
    def _save_config(self):
        """Save configuration to JSON file"""
        try:
            config = {
                'emergency_contacts': self.contacts,
                'twilio': self.twilio_config
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info("Saved SOS config")
        except Exception as e:
            logger.error(f"Error saving SOS config: {e}")
    
    def send_alert(self, location="Unknown", reason="Drowsiness detected"):
        """Send SOS alert to all emergency contacts"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            logger.info("Alert cooldown active, skipping alert")
            return False

        if not self.twilio_client or not self.contacts:
            logger.warning("Cannot send alert: Twilio not configured or no contacts")
            return False

        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = (
                f"⚠️ DRIVER ALERT ⚠️\n"
                f"Time: {timestamp}\n"
                f"Reason: {reason}\n"
                f"Location: {location}\n"
                f"Please check on the driver immediately!"
            )

            for contact in self.contacts:
                try:
                    self.twilio_client.messages.create(
                        body=message,
                        from_=self.twilio_config['from_number'],
                        to=contact['phone']
                    )
                    logger.info(f"Alert sent to {contact['name']}")
                except Exception as e:
                    logger.error(f"Error sending alert to {contact['name']}: {e}")

            self.last_alert_time = current_time
            return True

        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
            return False
    
    def configure_twilio(self, account_sid, auth_token, from_number):
        """Configure Twilio credentials"""
        try:
            self.twilio_config = {
                'account_sid': account_sid,
                'auth_token': auth_token,
                'from_number': from_number
            }
            self._save_config()
            self._setup_twilio()
            return True
        except Exception as e:
            logger.error(f"Error configuring Twilio: {e}")
            return False
    
    def send_email(self, to_email, subject, message):
        """
        Send email alert
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            message: Email message
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['email']
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            )
            server.starttls()
            
            # Login to email account
            server.login(
                self.email_config['email'],
                self.email_config['password']
            )
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            print(f"Email sent to {to_email}")
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def send_sos(self, alert_type, details, location="Unknown"):
        """
        Send SOS alerts to all emergency contacts
        
        Args:
            alert_type: Type of alert (drowsiness, heart_rate)
            details: Alert details
            location: Location information
            
        Returns:
            bool: True if at least one alert was sent successfully
        """
        # Create alert message
        if alert_type == 'drowsiness':
            subject = "EMERGENCY: Driver Drowsiness Detected"
            message = f"EMERGENCY ALERT: Driver drowsiness detected.\n\nDetails: {details}\nLocation: {location}\n\nPlease take immediate action."
        elif alert_type == 'heart_rate':
            subject = "EMERGENCY: Abnormal Heart Rate Detected"
            message = f"EMERGENCY ALERT: Abnormal heart rate detected.\n\nDetails: {details}\nLocation: {location}\n\nPlease take immediate action."
        else:
            subject = "EMERGENCY ALERT"
            message = f"EMERGENCY ALERT: {alert_type}.\n\nDetails: {details}\nLocation: {location}\n\nPlease take immediate action."
        
        # Send alerts in separate threads
        sms_thread = threading.Thread(
            target=self._send_all_sms,
            args=(message,)
        )
        email_thread = threading.Thread(
            target=self._send_all_emails,
            args=(subject, message)
        )
        
        sms_thread.daemon = True
        email_thread.daemon = True
        
        sms_thread.start()
        email_thread.start()
        
        return True
    
    def _send_all_sms(self, message):
        """
        Send SMS to all emergency contacts
        
        Args:
            message: Alert message
        """
        for contact in self.contacts:
            if 'phone' in contact and contact['phone']:
                self.send_alert(location=contact['name'], reason=message)
    
    def _send_all_emails(self, subject, message):
        """
        Send emails to all emergency contacts
        
        Args:
            subject: Email subject
            message: Email message
        """
        for contact in self.contacts:
            if 'email' in contact and contact['email']:
                self.send_email(contact['email'], subject, message)

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize SOS Alert system
    sos = SOSAlert()
    
    # Test configuration
    sos.configure_twilio(
        account_sid="your_account_sid",
        auth_token="your_auth_token",
        from_number="your_twilio_number"
    )
    
    # Add test contact
    sos.add_contact("Test Contact", "+1234567890")
    
    # Test alert
    sos.send_alert(
        location="Test Location",
        reason="Test Alert"
    ) 