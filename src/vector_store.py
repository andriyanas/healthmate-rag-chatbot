"""
Local vector store module using ChromaDB and Sentence Transformers.

Handles vector indexing, collection management, and semantic retrieval 
for the MedQuAD dataset without relying on external API calls.
"""

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from tqdm import tqdm

from src.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
)


def _get_embedding_function():
    """
    Initializes and returns the Sentence Transformer embedding function.

    Returns:
        SentenceTransformerEmbeddingFunction: Local embedding model instance.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )


def get_client():
    """
    Creates a persistent ChromaDB client pointing to the configured local directory.

    Returns:
        chromadb.PersistentClient: Persistent client instance.
    """
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def collection_exists() -> bool:
    """
    Checks if the target collection already exists in the local ChromaDB database.

    Returns:
        bool: True if collection exists, False otherwise.
    """
    client = get_client()
    existing = [c.name for c in client.list_collections()]
    return COLLECTION_NAME in existing


def get_collection():
    """
    Retrieves the existing vector collection with the configured embedding function.

    Returns:
        chromadb.Collection: Active ChromaDB collection object.
    """
    client = get_client()
    return client.get_collection(
        name=COLLECTION_NAME, embedding_function=_get_embedding_function()
    )


def build_index(csv_path):
    """
    Builds the vector store index from a MedQuAD CSV file in batches.

    Args:
        csv_path (str): File path to the raw dataset CSV.

    Returns:
        chromadb.Collection: Newly populated ChromaDB collection instance.
    """

    df = pd.read_csv(csv_path)

    client = get_client()

    # Drop existing collection to ensure clean state during re-indexing
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=_get_embedding_function()
    )

    total = len(df)
    print(f"Mengindeks {total} baris MedQuAD ke ChromaDB (lokal, gratis)...")

    # Ingest records in batches to optimize memory usage
    for start in tqdm(range(0, total, EMBEDDING_BATCH_SIZE)):
        batch = df.iloc[start : start + EMBEDDING_BATCH_SIZE]

        ids = [f"doc_{i}" for i in batch.index.tolist()]
        
        documents = batch["question"].astype(str).tolist()
        metadatas = [
            {
                "answer": str(row.get("answer", "")),
                "question": str(row.get("question", "")),
                "focus": str(row.get("question_focus", "")),
                "qtype": str(row.get("question_type", "")),
            }
            for _, row in batch.iterrows()
        ]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Selesai. Total dokumen ter-index: {collection.count()}")
    return collection


def retrieve(query: str, top_k: int):
    """
    Executes a semantic similarity search against the local vector collection.

    Args:
        query (str): The search prompt or user question.
        top_k (int): Number of top relevant document matches to return.

    Returns:
        list: List of dictionaries containing document metadata and relevance scores.
    """
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    docs = []
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Map retrieved raw metadata and compute similarity score
    for meta, dist in zip(metadatas, distances):
        docs.append(
            {
                "question": meta.get("question", ""),
                "answer": meta.get("answer", ""),
                "focus": meta.get("focus", ""),
                "qtype": meta.get("qtype", ""),
                "score": 1 - dist,  # semakin tinggi semakin relevan
            }
        )
    return docs
