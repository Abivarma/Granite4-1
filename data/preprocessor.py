"""Data loading and preprocessing utilities for BFCL v3 benchmark."""


def bfcl_to_openai_tools(bfcl_functions: list[dict]) -> list[dict]:
    """Convert BFCL function definitions to OpenAI tools schema for Ollama."""
    return [
        {
            "type": "function",
            "function": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for fn in bfcl_functions
    ]
