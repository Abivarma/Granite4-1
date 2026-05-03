import pytest
from data.loader import load_bfcl_category, load_ground_truth

def test_load_simple_category_returns_list():
    samples = load_bfcl_category("simple")
    assert isinstance(samples, list)
    assert len(samples) > 0

def test_sample_has_required_fields():
    samples = load_bfcl_category("simple")
    s = samples[0]
    assert "id" in s
    assert "question" in s
    assert "function" in s

def test_question_is_list_of_messages():
    samples = load_bfcl_category("simple")
    msgs = samples[0]["question"]
    assert isinstance(msgs, list)
    assert "role" in msgs[0]
    assert "content" in msgs[0]

def test_ground_truth_keys_match_sample_ids():
    samples = load_bfcl_category("simple")
    gt = load_ground_truth("simple")
    sample_ids = {s["id"] for s in samples}
    gt_ids = set(gt.keys())
    overlap = len(sample_ids & gt_ids) / len(sample_ids)
    assert overlap >= 0.9
