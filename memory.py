from typing import Dict

from langchain_core.chat_history import InMemoryChatMessageHistory

# In memory store for chat messages
# TODO: replace with a persistent store later
store: Dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
