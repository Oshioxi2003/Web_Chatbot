#!/usr/bin/env python3
"""
Script to download necessary dependencies for Heroku deployment
- NLTK corpora
- Model files if needed
"""

import os
import nltk
import requests
import sys

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    
    # Create nltk_data directory
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    # Download required corpora
    corpora = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    
    for corpus in corpora:
        try:
            print(f"Downloading {corpus}...")
            nltk.download(corpus, download_dir=nltk_data_dir, quiet=True)
            print(f"✅ {corpus} downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download {corpus}: {e}")
    
    print("NLTK data download completed!")

def download_model_file():
    """Download model file if it doesn't exist"""
    model_file = 'data.pth'
    
    if os.path.exists(model_file):
        print(f"✅ Model file {model_file} already exists")
        return
    
    # If you have a model file hosted somewhere, download it here
    # Example:
    # model_url = "https://your-storage/data.pth"
    # try:
    #     print(f"Downloading model file from {model_url}...")
    #     response = requests.get(model_url)
    #     response.raise_for_status()
    #     with open(model_file, 'wb') as f:
    #         f.write(response.content)
    #     print(f"✅ Model file downloaded successfully")
    # except Exception as e:
    #     print(f"❌ Failed to download model file: {e}")
    
    print(f"⚠️  Model file {model_file} not found. Please ensure it's included in your deployment.")

def main():
    """Main function to download all dependencies"""
    print("Starting dependency download for Heroku deployment...")
    
    try:
        download_nltk_data()
        download_model_file()
        print("✅ All dependencies downloaded successfully!")
        
    except Exception as e:
        print(f"❌ Error during dependency download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
