#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download dlib shape predictor model
"""
import urllib.request
import bz2
import os

def download_dlib_model():
    """Download and extract dlib shape predictor"""
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    url = 'https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2'
    compressed_path = os.path.join(model_dir, 'shape_predictor_68_face_landmarks.dat.bz2')
    extracted_path = os.path.join(model_dir, 'shape_predictor_68_face_landmarks.dat')
    
    # Check if already exists
    if os.path.exists(extracted_path):
        print(f"Model already exists: {extracted_path}")
        return True
    
    try:
        print(f"Downloading dlib shape predictor (99MB)...")
        urllib.request.urlretrieve(url, compressed_path)
        print(f"✅ Downloaded: {compressed_path}")
        
        print(f"Extracting...")
        with bz2.BZ2File(compressed_path) as f_in:
            with open(extracted_path, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"Extracted: {extracted_path}")
        
        # Remove compressed file
        os.remove(compressed_path)
        print(f"Cleaned up temporary file")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = download_dlib_model()
    if success:
        print("\ndlib model ready!")
    else:
        print("\nFailed to download dlib model")

