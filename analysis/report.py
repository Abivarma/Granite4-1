"""Pandas report tables and matplotlib/seaborn charts."""
import pandas as pd
from evaluation.metrics import CategoryMetrics


def build_accuracy_table(results: dict[str, dict[str, CategoryMetrics]]) -> pd.DataFrame:
    """Build accuracy table. results: {model_label: {category: CategoryMetrics}}"""
    rows = []
    for model_label, cat_metrics in results.items():
        row = {"Model": model_label}
        total_correct = 0
        total_count = 0
        for cat, m in cat_metrics.items():
            row[cat.replace("_", " ").title()] = f"{m.full_acc * 100:.1f}%"
            total_correct += int(m.full_acc * m.total)
            total_count += m.total
        overall = total_correct / total_count if total_count > 0 else 0
        row["Overall"] = f"{overall * 100:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def build_token_efficiency_table(results: dict[str, dict[str, CategoryMetrics]]) -> pd.DataFrame:
    """Avg completion tokens on correct calls only (fewer = better)."""
    rows = []
    for model_label, cat_metrics in results.items():
        all_correct_tokens = []
        for m in cat_metrics.values():
            if m.avg_tokens_correct > 0:
                all_correct_tokens.extend([m.avg_tokens_correct] * max(1, int(m.full_acc * m.total)))
        avg = sum(all_correct_tokens) / len(all_correct_tokens) if all_correct_tokens else 0.0
        rows.append({"Model": model_label, "Avg Tokens (Correct Calls)": round(avg, 1)})
    return pd.DataFrame(rows).set_index("Model").sort_values("Avg Tokens (Correct Calls)")


def build_version_delta_table(
    results: dict[str, dict[str, CategoryMetrics]],
    model_new: str,
    model_old: str,
) -> pd.DataFrame:
    """Accuracy delta between two model versions per category."""
    rows = []
    new_metrics = results.get(model_new, {})
    old_metrics = results.get(model_old, {})
    for cat in new_metrics:
        if cat in old_metrics:
            delta = (new_metrics[cat].full_acc - old_metrics[cat].full_acc) * 100
            rows.append({
                "Category": cat.replace("_", " ").title(),
                f"{model_new} Acc %": f"{new_metrics[cat].full_acc * 100:.1f}%",
                f"{model_old} Acc %": f"{old_metrics[cat].full_acc * 100:.1f}%",
                "Delta (pp)": f"{delta:+.1f}",
            })
    return pd.DataFrame(rows).set_index("Category")
