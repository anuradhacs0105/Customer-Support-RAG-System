
import streamlit as st

from src.retrieve import load_vector_store, retrieve_documents
from src.generate_answer import generate_answer


# Page configuration
st.set_page_config(
    page_title="Customer Support Assistant",
    page_icon="💬",
    layout="centered"
)


# Load FAISS database only once
@st.cache_resource
def get_vector_store():
    return load_vector_store()


# Application heading
st.title("Customer Support Assistant")

st.write(
    "Ask questions about orders, payments, refunds, delivery and more."
)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Load vector database
try:
    index, chunks = get_vector_store()

except Exception as error:
    st.error("Unable to load the document database.")
    st.exception(error)
    st.stop()


# Clear conversation button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# Display previous conversation
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Retrieved Sources"):
                for source in message["sources"]:
                    st.write(source["chunk"])


# Customer question input
user_input = st.chat_input("Enter your question here...")


if user_input:

    # Display customer question
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save customer question
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    try:

        # Retrieve only the closest matching document
        results = retrieve_documents(
            user_input,
            index,
            chunks,
            top_k=1
        )

        # Generate answer using Gemini
        answer = generate_answer(
            user_input,
            results
        )

        # Save generated answer and retrieved sources
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": results
        })

        # Display generated answer
        with st.chat_message("assistant"):
            st.markdown(answer)

            if results:
                with st.expander("Retrieved Sources"):
                    for source in results:
                        st.write(source["chunk"])

    except Exception:
        st.error(
            "Sorry, something went wrong. Please try again."
        )
