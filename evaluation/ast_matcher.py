"""AST matching and metrics computation for BFCL v3 evaluation."""
from dataclasses import dataclass


@dataclass
class MatchResult:
    name_correct: bool
    params_correct: bool

    @property
    def full_match(self) -> bool:
        return self.name_correct and self.params_correct


def _normalize_scalar(v):
    """Normalize a single scalar for comparison: coerce int/float to float, else lowercase str."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return str(v).strip().lower()


def _normalize(value) -> set:
    """Normalize a predicted value to a set for lenient comparison."""
    if isinstance(value, list):
        return {_normalize_scalar(v) for v in value}
    return {_normalize_scalar(value)}


def _values_match(predicted_val, gt_list: list) -> bool:
    """Return True if predicted_val matches any value in gt_list (lenient)."""
    pred_set = _normalize(predicted_val)
    gt_set = _normalize(gt_list)
    return bool(pred_set & gt_set)


def match_single_call(predicted: dict | None, ground_truth: dict) -> MatchResult:
    """Compare one predicted tool call against one ground truth entry.

    ground_truth format: {"function_name": {"param": [acceptable_values], ...}}
    predicted format:    {"name": str, "arguments": {param: value, ...}}
    """
    if predicted is None:
        return MatchResult(name_correct=False, params_correct=False)

    gt_fn_name = next(iter(ground_truth))
    gt_params = ground_truth[gt_fn_name]

    name_correct = predicted.get("name", "") == gt_fn_name
    if not name_correct:
        return MatchResult(name_correct=False, params_correct=False)

    pred_args = predicted.get("arguments", {})
    params_correct = all(
        param in pred_args and _values_match(pred_args[param], acceptable)
        for param, acceptable in gt_params.items()
    )
    return MatchResult(name_correct=True, params_correct=params_correct)


def match_parallel_calls(predicted_calls: list[dict], ground_truth_list: list[dict]) -> list[MatchResult]:
    """Match predicted calls against GT list (unordered). Returns one MatchResult per GT entry."""
    results = []
    for gt in ground_truth_list:
        best = MatchResult(name_correct=False, params_correct=False)
        for pred in predicted_calls:
            r = match_single_call(pred, gt)
            if r.full_match:
                best = r
                break
            if r.name_correct:
                best = r
        results.append(best)
    return results
