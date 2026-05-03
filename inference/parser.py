"""Ollama inference runner and response parser."""


def extract_tool_calls(response) -> list[dict]:
    """Extract tool calls from an Ollama chat response.

    Returns list of {"name": str, "arguments": dict}.
    Returns [] if model produced no tool calls.
    """
    tool_calls = getattr(response.message, "tool_calls", None)
    if not tool_calls:
        return []
    return [
        {
            "name": tc.function.name,
            "arguments": tc.function.arguments if isinstance(tc.function.arguments, dict) else {},
        }
        for tc in tool_calls
    ]
