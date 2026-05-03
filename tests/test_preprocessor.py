from data.preprocessor import bfcl_to_openai_tools

def test_single_function_wrapped_correctly():
    bfcl_fn = {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
    result = bfcl_to_openai_tools([bfcl_fn])
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["description"] == "Get weather for a city"
    assert "location" in result[0]["function"]["parameters"]["properties"]

def test_multiple_functions_all_converted():
    fns = [
        {"name": "fn_a", "description": "A", "parameters": {"type": "object", "properties": {}}},
        {"name": "fn_b", "description": "B", "parameters": {"type": "object", "properties": {}}},
    ]
    result = bfcl_to_openai_tools(fns)
    assert len(result) == 2
    assert result[0]["function"]["name"] == "fn_a"
    assert result[1]["function"]["name"] == "fn_b"

def test_empty_list_returns_empty():
    assert bfcl_to_openai_tools([]) == []

def test_missing_description_uses_empty_string():
    fn = {"name": "do_thing", "parameters": {"type": "object", "properties": {}}}
    result = bfcl_to_openai_tools([fn])
    assert result[0]["function"]["description"] == ""
