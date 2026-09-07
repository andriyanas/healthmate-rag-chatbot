"""
Configuration settings for the Healthmate RAG application.

This module consolidates system paths, vector database parameters, 
embedding model configurations, and external API setups.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the root .env file
load_dotenv()

# Base directory definitions
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_CSV_PATH = DATA_DIR / "medquad_full.csv"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "medquad_health_kb"

# Dataset repository configuration
HF_DATASET_NAME = "lavita/MedQuAD"

# Vector embedding parameter settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 256

# LLM service settings
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# RAG search and context memory thresholds
TOP_K = 4                 
MAX_HISTORY_TURNS = 6      
