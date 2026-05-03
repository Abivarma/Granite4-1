"""Data loading and preprocessing utilities for BFCL v3 benchmark."""
import copy


def _normalize_parameters(params: dict) -> dict:
    """Normalize BFCL parameter schema to OpenAI/JSON Schema format.

    BFCL uses "type": "dict" for object parameters; OpenAI/Ollama requires
    "type": "object". This function recursively fixes those occurrences.
    """
    params = copy.deepcopy(params)
    if params.get("type") == "dict":
        params["type"] = "object"
    # Recurse into properties
    for prop in params.get("properties", {}).values():
        if isinstance(prop, dict):
            if prop.get("type") == "dict":
                prop["type"] = "object"
            if "properties" in prop:
                prop.update(_normalize_parameters(prop))
    return params


def bfcl_to_openai_tools(bfcl_functions: list[dict]) -> list[dict]:
    """Convert BFCL function definitions to OpenAI tools schema for Ollama."""
    return [
        {
            "type": "function",
            "function": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": _normalize_parameters(
                    fn.get("parameters", {"type": "object", "properties": {}})
                ),
            },
        }
        for fn in bfcl_functions
    ]
