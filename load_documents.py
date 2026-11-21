from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of the customer support knowledge base
DOCUMENT_PATH = PROJECT_ROOT / "data" / "customer_support.txt"


def load_document():
    """Load the customer support knowledge base."""

    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Document not found: {DOCUMENT_PATH}"
        )

    with open(
        DOCUMENT_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    return text


def create_chunks(text):
    """Create one chunk for each customer-support topic."""

    sections = text.split("\n\n")

    chunks = []

    current_chunk = []

    headings = {
        "ORDER TRACKING",
        "ORDER CANCELLATION",
        "REFUND POLICY",
        "RETURN POLICY",
        "PAYMENT METHODS",
        "DELIVERY",
        "ACCOUNT SUPPORT",
        "DAMAGED PRODUCT"
    }

    for section in sections:

        section = section.strip()

        if not section:
            continue

        if section in headings:

            if current_chunk:
                chunks.append(
                    "\n\n".join(current_chunk)
                )

            current_chunk = [section]

        else:

            current_chunk.append(section)

    if current_chunk:
        chunks.append(
            "\n\n".join(current_chunk)
        )

    return chunks


if __name__ == "__main__":

    print(
        "Loading customer support knowledge base..."
    )

    document = load_document()

    print(
        "Document loaded successfully."
    )

    chunks = create_chunks(document)

    print(
        "Number of chunks:",
        len(chunks)
    )

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"\n--- Chunk {i} ---"
        )

        print(chunk)