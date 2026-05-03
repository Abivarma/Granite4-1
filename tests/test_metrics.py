"""Unit tests for metrics computation."""
import statistics
from evaluation.metrics import compute_category_metrics, evaluate_results, CategoryMetrics


def _make_eval(name_correct, full_match, completion_tokens, latency_ms, error=None):
    return {
        "name_correct": name_correct,
        "full_match": full_match,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "error": error,
    }


def test_perfect_accuracy():
    evals = [_make_eval(True, True, 20, 100), _make_eval(True, True, 30, 200)]
    m = compute_category_metrics(evals)
    assert m.name_acc == 1.0
    assert m.full_acc == 1.0


def test_zero_accuracy():
    evals = [_make_eval(False, False, 10, 150), _make_eval(False, False, 10, 150)]
    m = compute_category_metrics(evals)
    assert m.name_acc == 0.0
    assert m.full_acc == 0.0


def test_token_efficiency_only_on_correct():
    evals = [
        _make_eval(True, True, 20, 100),
        _make_eval(True, False, 50, 200),
        _make_eval(False, False, 80, 300),
    ]
    m = compute_category_metrics(evals)
    assert m.avg_tokens_correct == 20.0


def test_null_rate():
    evals = [
        _make_eval(False, False, 0, 0, error="no tool call"),
        _make_eval(True, True, 20, 100),
    ]
    m = compute_category_metrics(evals)
    assert m.null_rate == 0.5


def test_median_latency():
    evals = [
        _make_eval(True, True, 10, 100),
        _make_eval(True, True, 10, 200),
        _make_eval(True, True, 10, 300),
    ]
    m = compute_category_metrics(evals)
    assert m.median_latency_ms == 200.0
