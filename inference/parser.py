"""Ollama inference runner and response parser."""

import json


def extract_tool_calls(response) -> list[dict]:
    """Extract tool calls from an Ollama chat response.

    Returns list of {"name": str, "arguments": dict}.
    Returns [] if model produced no tool calls.
    """
    tool_calls = getattr(response.message, "tool_calls", None)
    if not tool_calls:
        return []
    result = []
    for tc in tool_calls:
        args = tc.function.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        elif not isinstance(args, dict):
            args = {}
        result.append({"name": tc.function.name, "arguments": args})
    return result
