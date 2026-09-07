"""
Script to build and update the ChromaDB vector store index from raw CSV data.

This script checks for the existence of the processed dataset and triggers 
the embedding generation and indexing process for the RAG pipeline.
"""
import sys
from pathlib import Path

# Add the project root directory to sys.path to resolve module imports from 'src'
# Path(__file__).resolve().parent.parent gets the absolute path of the root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import required path configurations and vector indexing logic from the 'src' module
from src.config import RAW_CSV_PATH
from src.vector_store import build_index


def main():
    """
    Main execution function to validate dataset presence and build the vector index.
    """
    
    # Verify whether the target raw CSV dataset file exists before proceeding
    if not RAW_CSV_PATH.exists():
        print(f"File {RAW_CSV_PATH} tidak ditemukan.")
        print("Jalankan dulu: python scripts/download_data.py")
        return

    # Execute the vector store ingestion and embedding generation process
    build_index(RAW_CSV_PATH)


if __name__ == "__main__":
    # Ensure main() only executes when running this file directly from the terminal
    main()
