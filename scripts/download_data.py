"""
Script to download, clean, and cache the MedQuAD medical dataset locally.

This script fetches the specified dataset from Hugging Face, removes any missing 
or empty question-answer entries to ensure data quality, and saves the cleaned 
data as a CSV file for ingestion into the vector store.
"""

import sys
from pathlib import Path

# Add the project root directory to sys.path to enable absolute imports from the 'src' package
# Path(__file__).resolve().parent.parent resolves to the root folder of the project
sys.path.append(str(Path(__file__).resolve().parent.parent))


from datasets import load_dataset
from src.config import HF_DATASET_NAME, RAW_CSV_PATH, DATA_DIR


def main():
    """
    Main function to execute the dataset retrieval, cleaning, and storage workflow.
    """

    # Ensure the target directory for storing raw data exists; create parents if necessary
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Mengunduh dataset '{HF_DATASET_NAME}' dari Hugging Face...")
    
    # Fetch the specified dataset split ('train') directly from Hugging Face Hub
    ds = load_dataset(HF_DATASET_NAME, split="train")
    # Convert the Hugging Face Dataset object into a pandas DataFrame for easier manipulation
    df = ds.to_pandas()

    # Track initial row count prior to data cleaning
    before = len(df)
    # Remove rows where either the 'question' or 'answer' field is null/NaN
    df = df.dropna(subset=["question", "answer"])
    # Filter out rows that contain only whitespace characters in 'question' or 'answer'
    df = df[(df["question"].str.strip() != "") & (df["answer"].str.strip() != "")]
    # Track final row count after applying data quality filters
    after = len(df)
    # Persist the cleaned DataFrame locally as a CSV file without the pandas default index column
    df.to_csv(RAW_CSV_PATH, index=False)

    print(f"Selesai. {after}/{before} baris valid disimpan ke: {RAW_CSV_PATH}")
    print("Kolom yang tersedia:", list(df.columns))


if __name__ == "__main__":
    # Ensure script runs only when explicitly called from the CLI, not upon import
    main()
