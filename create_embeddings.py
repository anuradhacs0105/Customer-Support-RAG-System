from sentence_transformers import SentenceTransformer

from load_documents import load_document, create_chunks


MODEL_NAME = "all-MiniLM-L6-v2"


def create_embeddings(chunks):
    """Convert text chunks into numerical embeddings."""

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    print("Creating embeddings...")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    print("Embeddings created successfully.")

    return embeddings


if __name__ == "__main__":

    print("Loading customer support document...")

    document = load_document()

    chunks = create_chunks(document)

    print(
        f"Number of chunks: {len(chunks)}"
    )

    embeddings = create_embeddings(chunks)

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        "Embedding generation completed."
    )