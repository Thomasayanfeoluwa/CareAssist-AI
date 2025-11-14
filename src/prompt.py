def build_prompt(user_query: str, context_str: str) -> str:
    return (
        f"User asked: {user_query}\n"
        f"Context:\n{context_str}\n"
        "Answer:"
    )
