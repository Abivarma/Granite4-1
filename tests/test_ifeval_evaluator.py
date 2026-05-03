"""Unit tests for IFEval evaluator."""
from evaluation.ifeval_evaluator import compute_ifeval_metrics, IFEvalMetrics


def _make_result(sample_id, response="", tokens=50, latency=1000.0, error=None):
    return {
        "id": sample_id,
        "response": response,
        "completion_tokens": tokens,
        "latency_ms": latency,
        "error": error,
    }

def _make_sample(sample_id, n_instructions=1):
    return {
        "id": sample_id,
        "prompt": "test prompt",
        "instruction_id_list": ["keywords:existence"] * n_instructions,
        "kwargs": [{"keywords": ["test"]}] * n_instructions,
    }


def test_all_errors_gives_zero_accuracy():
    results = [_make_result("ifeval_0", error="timeout")]
    samples = [_make_sample("ifeval_0")]
    m = compute_ifeval_metrics(results, samples)
    assert m.prompt_strict_acc == 0.0
    assert m.instruction_strict_acc == 0.0


def test_total_prompts_matches_results():
    results = [_make_result(f"ifeval_{i}") for i in range(3)]
    samples = [_make_sample(f"ifeval_{i}") for i in range(3)]
    m = compute_ifeval_metrics(results, samples)
    assert m.total_prompts == 3


def test_avg_tokens_excludes_errors():
    results = [
        _make_result("ifeval_0", tokens=100),
        _make_result("ifeval_1", error="fail", tokens=0),
    ]
    samples = [_make_sample("ifeval_0"), _make_sample("ifeval_1")]
    m = compute_ifeval_metrics(results, samples)
    assert m.avg_tokens == 100.0
