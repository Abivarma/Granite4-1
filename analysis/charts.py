"""Pandas report tables and matplotlib/seaborn charts."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script/test use
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from evaluation.metrics import CategoryMetrics

PALETTE = sns.color_palette("colorblind", 5)


def _model_colors(model_labels: list[str]) -> dict:
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(model_labels)}


def plot_accuracy_by_category(
    results: dict[str, dict[str, CategoryMetrics]],
    save_path: str = "results/accuracy_by_category.png",
) -> None:
    categories = list(next(iter(results.values())).keys())
    cat_labels = [c.replace("_", " ").title() for c in categories]
    models = list(results.keys())
    colors = _model_colors(models)

    x = np.arange(len(categories))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        accs = [results[model][cat].full_acc * 100 for cat in categories]
        ax.bar(x + i * width, accs, width, label=model, color=colors[model])

    ax.set_xlabel("Category")
    ax.set_ylabel("Full AST Accuracy (%)")
    ax.set_title("Tool Calling Accuracy by Category — BFCL v3")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(cat_labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_token_efficiency(
    results: dict[str, dict[str, CategoryMetrics]],
    save_path: str = "results/token_efficiency.png",
) -> None:
    models = list(results.keys())
    colors = _model_colors(models)
    avg_tokens = []
    for model in models:
        all_t = [m.avg_tokens_correct for m in results[model].values() if m.avg_tokens_correct > 0]
        avg_tokens.append(sum(all_t) / len(all_t) if all_t else 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(models, avg_tokens, color=[colors[m] for m in models])
    ax.set_xlabel("Avg Completion Tokens (Correct Calls Only)")
    ax.set_title("Token Efficiency — Fewer is Better")
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_latency_comparison(
    raw_results: dict[str, list[dict]],
    save_path: str = "results/latency_comparison.png",
) -> None:
    """raw_results: {model_label: [inference result dicts]}"""
    data = [
        [r["latency_ms"] for r in calls if r.get("latency_ms", 0) > 0]
        for calls in raw_results.values()
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=list(raw_results.keys()), vert=False, patch_artist=True)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Response Latency Distribution per Model")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
