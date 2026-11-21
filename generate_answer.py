
from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai

# Load API key from the project's .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini model
MODEL_NAME = "gemini-3.5-flash-lite"


def generate_answer(question, retrieved_results):

    if not API_KEY:
        raise ValueError(
            "Google API key not found. Check your .env file."
        )

    if not retrieved_results:
        return "Sorry, I could not find relevant information in the customer support documents."

    # Collect information retrieved by FAISS
    context = "\n\n".join(
        result["chunk"] for result in retrieved_results
    )

    prompt = f"""
You are a helpful customer support assistant.

Answer the customer's question using only the
information provided in the context below.

If the answer is not available in the context,
clearly say that you do not have that information
and suggest contacting customer support.

Do not invent information.

Context:
{context}

Customer Question:
{question}

Answer:
"""

    # Connect to Gemini API
    client = genai.Client(api_key=API_KEY)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    if not response.text:
        return "Sorry, I could not generate an answer. Please try again."

    return response.text
