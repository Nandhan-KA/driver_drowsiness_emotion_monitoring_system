import urllib.request
import os
import sys

def download_shape_predictor():
    """Download the shape predictor file if it doesn't exist"""
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    
    if not os.path.exists(predictor_path):
        print(f"Downloading shape predictor file...")
        try:
            urllib.request.urlretrieve(url, predictor_path + '.bz2')
            import bz2
            with bz2.open(predictor_path + '.bz2', 'rb') as source, open(predictor_path, 'wb') as dest:
                dest.write(source.read())
            os.remove(predictor_path + '.bz2')
            print("Download completed successfully.")
        except Exception as e:
            print(f"Error downloading shape predictor: {e}")
            sys.exit(1) 