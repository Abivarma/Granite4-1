from unittest.mock import MagicMock
from inference.parser import extract_tool_calls

def _make_response(tool_calls_data):
    response = MagicMock()
    if tool_calls_data is None:
        response.message.tool_calls = None
        return response
    calls = []
    for name, args in tool_calls_data:
        tc = MagicMock()
        tc.function.name = name
        tc.function.arguments = args
        calls.append(tc)
    response.message.tool_calls = calls
    return response

def test_single_tool_call_extracted():
    resp = _make_response([("get_weather", {"location": "Boston", "unit": "celsius"})])
    result = extract_tool_calls(resp)
    assert len(result) == 1
    assert result[0]["name"] == "get_weather"
    assert result[0]["arguments"]["location"] == "Boston"

def test_multiple_tool_calls_extracted():
    resp = _make_response([("fn_a", {"x": 1}), ("fn_b", {"y": 2})])
    result = extract_tool_calls(resp)
    assert len(result) == 2
    assert result[0]["name"] == "fn_a"
    assert result[1]["name"] == "fn_b"

def test_no_tool_calls_returns_empty_list():
    resp = _make_response(None)
    result = extract_tool_calls(resp)
    assert result == []

def test_empty_tool_calls_list_returns_empty():
    resp = _make_response([])
    result = extract_tool_calls(resp)
    assert result == []

def test_string_arguments_parsed_as_json():
    resp = _make_response([("get_weather", '{"location": "Boston", "unit": "celsius"}')])
    result = extract_tool_calls(resp)
    assert result[0]["arguments"]["location"] == "Boston"
    assert result[0]["arguments"]["unit"] == "celsius"
