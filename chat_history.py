
def initialize_chat_history(session_state):
    """Initialize chat history if it does not exist."""
    if "messages" not in session_state:
        session_state["messages"] = []


def add_message(session_state, role, content, sources=None):
    """Add a user or assistant message to chat history."""
    message = {
        "role": role,
        "content": content,
        "sources": sources or []
    }

    session_state["messages"].append(message)


def clear_chat_history(session_state):
    """Clear all previous chat messages."""
    session_state["messages"] = []
