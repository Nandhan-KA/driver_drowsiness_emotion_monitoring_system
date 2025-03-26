"""
Music Player Module for the Drowsiness Detection System
Recommends and plays music based on detected emotions
"""

import os
import random
import pygame
import threading
import time
import config
import logging
from threading import Thread, Event
import requests

logger = logging.getLogger(__name__)

class MusicPlayer:
    """Class for playing music based on emotions"""
    
    def __init__(self):
        """Initialize the music player with emotion-based playlists"""
        pygame.mixer.init()
        self.current_song = None
        self.is_playing = False
        self.stop_event = Event()
        self.player_thread = None
        
        # Create music directories if they don't exist
        self.music_dir = "music"
        self.playlists = {
            "happy": os.path.join(self.music_dir, "happy"),
            "sad": os.path.join(self.music_dir, "sad"),
            "neutral": os.path.join(self.music_dir, "neutral")
        }
        
        self._create_music_directories()
        self._download_sample_music()
        
        print("Music player initialized")
    
    def _create_music_directories(self):
        """Create directories for different emotion-based playlists"""
        for directory in self.playlists.values():
            os.makedirs(directory, exist_ok=True)

    def _download_sample_music(self):
        """Download sample music for each emotion if directories are empty"""
        # Sample royalty-free music URLs (you should replace these with actual music files)
        sample_music = {
            "happy": [
                ("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", "happy_1.mp3"),
                ("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3", "happy_2.mp3")
            ],
            "sad": [
                ("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3", "sad_1.mp3"),
                ("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3", "sad_2.mp3")
            ],
            "neutral": [
                ("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3", "neutral_1.mp3"),
                ("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3", "neutral_2.mp3")
            ]
        }

        for emotion, songs in sample_music.items():
            directory = self.playlists[emotion]
            if not os.listdir(directory):  # Only download if directory is empty
                for url, filename in songs:
                    try:
                        response = requests.get(url)
                        if response.status_code == 200:
                            with open(os.path.join(directory, filename), 'wb') as f:
                                f.write(response.content)
                            logger.info(f"Downloaded {filename} for {emotion} emotion")
                    except Exception as e:
                        logger.error(f"Error downloading {filename}: {e}")

    def _get_random_song(self, emotion):
        """Get a random song from the specified emotion playlist"""
        try:
            directory = self.playlists.get(emotion.lower(), self.playlists["neutral"])
            songs = [f for f in os.listdir(directory) if f.endswith(('.mp3', '.wav'))]
            if songs:
                return os.path.join(directory, random.choice(songs))
            return None
        except Exception as e:
            logger.error(f"Error getting random song: {e}")
            return None

    def play_for_emotion(self, emotion):
        """Play music based on the detected emotion"""
        try:
            if self.is_playing:
                self.stop()

            song_path = self._get_random_song(emotion)
            if song_path and os.path.exists(song_path):
                self.current_song = song_path
                self.is_playing = True
                self.stop_event.clear()
                self.player_thread = Thread(target=self._play_thread)
                self.player_thread.daemon = True
                self.player_thread.start()
                logger.info(f"Playing music for {emotion} emotion: {os.path.basename(song_path)}")
            else:
                logger.warning(f"No music found for {emotion} emotion")
        except Exception as e:
            logger.error(f"Error playing music: {e}")

    def _play_thread(self):
        """Thread function for playing music"""
        try:
            pygame.mixer.music.load(self.current_song)
            pygame.mixer.music.play(-1)  # -1 means loop indefinitely
            
            while not self.stop_event.is_set() and pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            pygame.mixer.music.stop()
            self.is_playing = False
        except Exception as e:
            logger.error(f"Error in play thread: {e}")
            self.is_playing = False

    def stop(self):
        """Stop the currently playing music"""
        try:
            self.stop_event.set()
            if self.is_playing:
                pygame.mixer.music.stop()
            self.is_playing = False
            if self.player_thread:
                self.player_thread.join(timeout=1.0)
        except Exception as e:
            logger.error(f"Error stopping music: {e}")

    def cleanup(self):
        """Clean up resources"""
        self.stop()
        try:
            pygame.mixer.quit()
        except Exception as e:
            logger.error(f"Error cleaning up music player: {e}")

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    player = MusicPlayer()
    
    # Test each emotion
    for emotion in ["happy", "sad", "neutral"]:
        print(f"Testing {emotion} music...")
        player.play_for_emotion(emotion)
        time.sleep(5)  # Play for 5 seconds
        player.stop()
        time.sleep(1)  # Wait before next emotion
    
    player.cleanup() 