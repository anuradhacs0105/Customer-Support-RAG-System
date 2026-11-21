
from pathlib import Path
import pickle
from functools import lru_cache

import faiss
from sentence_transformers import SentenceTransformer


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vector store files
INDEX_FILE = PROJECT_ROOT / "vector_store" / "index.faiss"
CHUNKS_FILE = PROJECT_ROOT / "vector_store" / "chunks.pkl"

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load the embedding model only once."""
    print("Loading embedding model...")
    return SentenceTransformer(MODEL_NAME)


def load_vector_store():
    """Load FAISS index and stored document chunks."""

    print("Loading FAISS vector store...")

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}"
        )

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "rb") as file:
        chunks = pickle.load(file)

    print(f"Loaded {index.ntotal} vectors.")

    return index, chunks


def retrieve_documents(query, index, chunks, top_k=1):
    """Retrieve the closest matching document chunk."""

    print(f"\nQuery: {query}")

    model = load_embedding_model()

    # Convert customer question into an embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # Search FAISS for the closest document
    distances, indices = index.search(
        query_embedding.astype("float32"),
        min(top_k, index.ntotal)
    )

    results = []

    for distance, index_number in zip(
        distances[0], indices[0]
    ):

        if index_number == -1:
            continue

        results.append({
            "chunk": chunks[index_number],
            "distance": float(distance)
        })

    return results


if __name__ == "__main__":

    index, chunks = load_vector_store()

    query = input("\nEnter your customer-support question: ")

    results = retrieve_documents(
        query,
        index,
        chunks,
        top_k=1
    )

    print("\nRetrieved Information")
    print("=====================")

    for number, result in enumerate(results, start=1):

        print(f"\nResult {number}")
        print(f"Distance: {result['distance']:.4f}")
        print(result["chunk"])
