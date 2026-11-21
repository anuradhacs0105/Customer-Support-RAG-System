
from retrieve import load_vector_store, retrieve_documents
from generate_answer import generate_answer

# Load the existing FAISS database
index, chunks = load_vector_store()

# Get customer question
question = input("\nEnter your customer support question: ")

# Retrieve relevant information
results = retrieve_documents(
    question, index, chunks, top_k=3
)

# Display retrieved information
print("\nRetrieved Sources")
print("=================")

for number, result in enumerate(results, start=1):
    print(f"\nSource {number}:")
    print(result["chunk"])

# Generate answer using Gemini
print("\nGemini Generated Answer")
print("=======================")

answer = generate_answer(question, results)
print(answer)
