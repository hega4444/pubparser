# config.py

# This file contains the configuration for the document processing workflow
# It contains the API keys, the settings for the document processing,
# and the output directory

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Title Processing Settings
MAX_TITLE_WORDS = int(os.getenv("MAX_TITLE_WORDS", "10"))

# Document Completion Threshold
COMPLETION_THRESHOLD = float(os.getenv("COMPLETION_THRESHOLD", "0.7"))

# Field Analysis Settings
ANALYZABLE_PREFIX = "EXTRACT:"

# Output Settings
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validate required environment variables
def validate_env():
    """Validate that all required environment variables are set"""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    if not MAX_TITLE_WORDS:
        raise ValueError("MAX_TITLE_WORDS environment variable is not set") 