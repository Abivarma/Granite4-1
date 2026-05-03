"""AST matching and metrics computation for BFCL v3 evaluation."""
import statistics
from dataclasses import dataclass
from evaluation.ast_matcher import match_single_call, match_parallel_calls


@dataclass
class CategoryMetrics:
    total: int
    name_acc: float
    full_acc: float
    null_rate: float
    avg_tokens_all: float
    avg_tokens_correct: float
    median_latency_ms: float
    p95_latency_ms: float


def compute_category_metrics(evaluated: list[dict]) -> CategoryMetrics:
    """Compute metrics from a list of evaluated result dicts.

    Each dict must have: name_correct, full_match, completion_tokens, latency_ms, error.
    """
    n = len(evaluated)
    if n == 0:
        return CategoryMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    name_correct_count = sum(1 for e in evaluated if e["name_correct"])
    full_match_count = sum(1 for e in evaluated if e["full_match"])
    null_count = sum(
        1 for e in evaluated
        if e.get("error") or (not e.get("name_correct") and e.get("completion_tokens", 0) == 0)
    )

    all_tokens = [e["completion_tokens"] for e in evaluated]
    correct_tokens = [e["completion_tokens"] for e in evaluated if e["full_match"]]
    latencies = sorted(e["latency_ms"] for e in evaluated)
    p95_idx = min(int(0.95 * n), n - 1)

    return CategoryMetrics(
        total=n,
        name_acc=name_correct_count / n,
        full_acc=full_match_count / n,
        null_rate=null_count / n,
        avg_tokens_all=statistics.mean(all_tokens) if all_tokens else 0.0,
        avg_tokens_correct=statistics.mean(correct_tokens) if correct_tokens else 0.0,
        median_latency_ms=statistics.median(latencies) if latencies else 0.0,
        p95_latency_ms=latencies[p95_idx] if latencies else 0.0,
    )


def evaluate_results(results: list[dict], ground_truth: dict, category: str) -> list[dict]:
    """Join inference results with ground truth and return evaluated list.

    Returns list of dicts with: id, name_correct, full_match, completion_tokens, latency_ms, error.
    """
    is_parallel = category in ("parallel", "parallel_multiple")
    evaluated = []

    for r in results:
        sample_id = r["id"]
        gt = ground_truth.get(sample_id)
        predicted_calls = r.get("tool_calls", [])

        if gt is None or r.get("error"):
            evaluated.append({
                "id": sample_id,
                "name_correct": False,
                "full_match": False,
                "completion_tokens": r.get("completion_tokens", 0),
                "latency_ms": r.get("latency_ms", 0.0),
                "error": r.get("error", "no_ground_truth"),
            })
            continue

        if is_parallel:
            gt_list = gt if isinstance(gt, list) else [gt]
            match_results = match_parallel_calls(predicted_calls, gt_list)
            name_correct = all(m.name_correct for m in match_results)
            full_match = all(m.full_match for m in match_results)
        else:
            gt_single = gt[0] if isinstance(gt, list) and len(gt) > 0 else gt
            pred = predicted_calls[0] if predicted_calls else None
            match_result = match_single_call(pred, gt_single)
            name_correct = match_result.name_correct
            full_match = match_result.full_match

        evaluated.append({
            "id": sample_id,
            "name_correct": name_correct,
            "full_match": full_match,
            "completion_tokens": r.get("completion_tokens", 0),
            "latency_ms": r.get("latency_ms", 0.0),
            "error": None,
        })

    return evaluated
