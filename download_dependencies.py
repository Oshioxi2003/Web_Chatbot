#!/usr/bin/env python3
"""
Script to download necessary dependencies for Heroku deployment
- NLTK corpora
- Model files if needed
"""

import os
import sys
# import nltk - removed to reduce dependencies
# import requests - removed since we don't download files

def download_nltk_data():
    """Download required NLTK data - DISABLED for Heroku deployment"""
    print("NLTK data download is disabled - app uses only Gemini API")
    # Commented out to reduce dependencies
    # try:
    #     import nltk
    #     nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    #     os.makedirs(nltk_data_dir, exist_ok=True)
    #     nltk.data.path.append(nltk_data_dir)
    #     corpora = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    #     for corpus in corpora:
    #         nltk.download(corpus, download_dir=nltk_data_dir, quiet=True)
    # except Exception as e:
    #     print(f"NLTK download skipped: {e}")
    pass

def download_model_file():
    """Download model file - DISABLED for Heroku deployment"""
    print("Local ML model is disabled - app uses only Gemini API")
    # Local PyTorch model removed to reduce slug size
    pass

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
