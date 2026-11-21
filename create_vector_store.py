from pathlib import Path
import pickle

import faiss

from create_embeddings import create_embeddings
from load_documents import load_document, create_chunks


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vector store location
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_FILE = VECTOR_STORE_DIR / "chunks.pkl"


def create_vector_store():

    print("Loading customer support document...")

    document = load_document()

    chunks = create_chunks(document)

    print(f"Number of chunks: {len(chunks)}")

    print("Creating embeddings...")

    embeddings = create_embeddings(chunks)

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # Get embedding dimension
    dimension = embeddings.shape[1]

    print(
        f"Embedding dimension: {dimension}"
    )

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to FAISS
    index.add(
        embeddings.astype("float32")
    )

    print(
        f"Number of vectors stored: {index.ntotal}"
    )

    # Create vector_store directory
    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # Save chunks
    with open(
        CHUNKS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print()
    print("Vector store created successfully.")

    print(
        f"FAISS index saved to: {INDEX_FILE}"
    )

    print(
        f"Chunks saved to: {CHUNKS_FILE}"
    )


if __name__ == "__main__":
    create_vector_store()