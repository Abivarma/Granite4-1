"""Unit tests for AST matching logic."""
from evaluation.ast_matcher import match_single_call, match_parallel_calls, MatchResult


def test_correct_name_and_params():
    predicted = {"name": "get_weather", "arguments": {"location": "Boston", "unit": "celsius"}}
    ground_truth = {"get_weather": {"location": ["Boston"], "unit": ["celsius"]}}
    r = match_single_call(predicted, ground_truth)
    assert r.name_correct is True
    assert r.params_correct is True
    assert r.full_match is True


def test_wrong_function_name():
    predicted = {"name": "get_forecast", "arguments": {"location": "Boston"}}
    ground_truth = {"get_weather": {"location": ["Boston"]}}
    r = match_single_call(predicted, ground_truth)
    assert r.name_correct is False
    assert r.full_match is False


def test_correct_name_wrong_param_value():
    predicted = {"name": "get_weather", "arguments": {"location": "London", "unit": "celsius"}}
    ground_truth = {"get_weather": {"location": ["Boston"], "unit": ["celsius"]}}
    r = match_single_call(predicted, ground_truth)
    assert r.name_correct is True
    assert r.params_correct is False
    assert r.full_match is False


def test_string_matches_single_item_list():
    predicted = {"name": "fn", "arguments": {"x": "hello"}}
    ground_truth = {"fn": {"x": ["hello"]}}
    r = match_single_call(predicted, ground_truth)
    assert r.params_correct is True


def test_int_matches_float():
    predicted = {"name": "fn", "arguments": {"n": 42}}
    ground_truth = {"fn": {"n": [42.0]}}
    r = match_single_call(predicted, ground_truth)
    assert r.params_correct is True


def test_missing_required_param():
    predicted = {"name": "get_weather", "arguments": {}}
    ground_truth = {"get_weather": {"location": ["Boston"]}}
    r = match_single_call(predicted, ground_truth)
    assert r.params_correct is False


def test_null_predicted_is_full_miss():
    r = match_single_call(None, {"get_weather": {"location": ["Boston"]}})
    assert r.name_correct is False
    assert r.full_match is False


def test_parallel_all_matched():
    predicted = [
        {"name": "get_weather", "arguments": {"location": "Boston"}},
        {"name": "get_time", "arguments": {"timezone": "EST"}},
    ]
    gt_list = [
        {"get_weather": {"location": ["Boston"]}},
        {"get_time": {"timezone": ["EST"]}},
    ]
    results = match_parallel_calls(predicted, gt_list)
    assert all(r.full_match for r in results)
