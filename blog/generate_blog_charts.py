"""Generate publication-quality charts for the Phase 1 blog post."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ── Data ──────────────────────────────────────────────────────────────────────
MODELS = ["Granite 4.1 8B", "Granite 4.0", "Llama 3.1 8B", "Qwen 2.5 7B", "Mistral 7B"]
CATEGORIES = ["Simple", "Multiple", "Parallel", "Parallel\nMultiple", "Overall"]

DATA = {
    "Granite 4.1 8B": [50.2, 46.5, 30.0, 47.0, 42.2],
    "Granite 4.0":    [43.0, 39.5, 24.2, 34.5, 34.8],
    "Llama 3.1 8B":   [49.5, 51.5, 24.0, 44.0, 40.4],
    "Qwen 2.5 7B":    [53.0, 48.0, 28.0, 47.0, 42.8],
    "Mistral 7B":     [48.2, 41.0, 21.2, 29.5, 34.9],
}

TOKENS = {
    "Llama 3.1 8B":   56.7,
    "Granite 4.0":    62.1,
    "Granite 4.1 8B": 63.5,
    "Qwen 2.5 7B":    64.3,
    "Mistral 7B":    104.8,
}

DELTA = {
    "Simple":            7.2,
    "Multiple":          7.0,
    "Parallel":          5.8,
    "Parallel Multiple": 12.5,
}

PALETTE = {
    "Granite 4.1 8B": "#0043CE",  # IBM Blue
    "Granite 4.0":    "#A8C4F0",  # Light IBM Blue
    "Llama 3.1 8B":   "#1A8754",  # Meta Green
    "Qwen 2.5 7B":    "#F08000",  # Qwen Orange
    "Mistral 7B":     "#9B59B6",  # Mistral Purple
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Chart 1: Main Accuracy Comparison (grouped bar) ──────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(CATEGORIES))
n = len(MODELS)
width = 0.15
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

for i, model in enumerate(MODELS):
    bars = ax.bar(
        x + offsets[i],
        DATA[model],
        width * 0.9,
        label=model,
        color=PALETTE[model],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8, color="#444")

ax.set_ylim(0, 68)
ax.set_xticks(x)
ax.set_xticklabels(CATEGORIES, fontsize=13)
ax.set_ylabel("Full AST Accuracy (%)", fontsize=13)
ax.set_title("Tool Calling Accuracy — BFCL v3 Full Benchmark\n5 Models · 1,000 Prompts Each · Mac M3 36GB · Ollama",
             fontsize=14, pad=14, fontweight="bold")
ax.axhline(68.27, color="red", linestyle="--", linewidth=1.2, alpha=0.6, zorder=2)
ax.text(len(CATEGORIES) - 0.5, 69.5, "IBM published 68.27%\n(official eval script)",
        ha="right", fontsize=9, color="red", alpha=0.8)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.set_facecolor("#FAFAFA")
plt.tight_layout()
plt.savefig("blog/chart1_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 1: Accuracy comparison saved")

# ── Chart 2: Heatmap ─────────────────────────────────────────────────────────
cats_heat = ["Simple", "Multiple", "Parallel", "Parallel Multiple", "Overall"]
matrix = np.array([DATA[m] for m in MODELS])

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(matrix, cmap="Blues", vmin=15, vmax=60, aspect="auto")

ax.set_xticks(range(len(cats_heat)))
ax.set_xticklabels(cats_heat, fontsize=12)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(MODELS, fontsize=12)

for i in range(len(MODELS)):
    for j in range(len(cats_heat)):
        val = matrix[i, j]
        color = "white" if val > 42 else "#222"
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
ax.set_title("Accuracy Heatmap — All Models × All Categories", fontsize=13, pad=12, fontweight="bold")
plt.tight_layout()
plt.savefig("blog/chart2_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 2: Heatmap saved")

# ── Chart 3: Granite 4.1 vs 4.0 Improvement ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
cats_delta = list(DELTA.keys())
deltas = list(DELTA.values())
g41 = [DATA["Granite 4.1 8B"][["Simple","Multiple","Parallel","Parallel\nMultiple"].index(
    c.replace(" ", "\n") if c == "Parallel Multiple" else c)] for c in cats_delta]
g40 = [DATA["Granite 4.0"][["Simple","Multiple","Parallel","Parallel\nMultiple"].index(
    c.replace(" ", "\n") if c == "Parallel Multiple" else c)] for c in cats_delta]

x = np.arange(len(cats_delta))
ax.bar(x - 0.2, g40, 0.35, label="Granite 4.0", color=PALETTE["Granite 4.0"],
       edgecolor="white", zorder=3)
ax.bar(x + 0.2, g41, 0.35, label="Granite 4.1 8B", color=PALETTE["Granite 4.1 8B"],
       edgecolor="white", zorder=3)

for i, (old, new, d) in enumerate(zip(g40, g41, deltas)):
    ax.annotate(f"+{d}pp", xy=(x[i] + 0.2, new), xytext=(x[i] + 0.2, new + 1.5),
                ha="center", fontsize=11, fontweight="bold", color="#0043CE",
                arrowprops=dict(arrowstyle="->", color="#0043CE", lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels(cats_delta, fontsize=13)
ax.set_ylim(0, 62)
ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("Granite 4.1 vs 4.0 — Version Improvement\n(IBM's Core Claim, Verified Locally)",
             fontsize=13, pad=12, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.set_facecolor("#FAFAFA")
plt.tight_layout()
plt.savefig("blog/chart3_version_delta.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 3: Version delta saved")

# ── Chart 4: Token Efficiency ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
tok_models = list(TOKENS.keys())
tok_vals   = list(TOKENS.values())
colors     = [PALETTE[m] for m in tok_models]

bars = ax.barh(tok_models, tok_vals, color=colors, edgecolor="white", height=0.55, zorder=3)
ax.bar_label(bars, fmt="%.1f tokens", padding=5, fontsize=11)
ax.set_xlim(0, 125)
ax.set_xlabel("Avg Completion Tokens on Correct Calls (fewer = cheaper)", fontsize=12)
ax.set_title("Token Efficiency — Which Model Is Most Economical?\n(Only counting calls the model actually got right)",
             fontsize=13, pad=12, fontweight="bold")
ax.axvline(56.7, color="#1A8754", linestyle="--", alpha=0.4)
ax.grid(axis="x", alpha=0.25, zorder=0)
ax.set_facecolor("#FAFAFA")
plt.tight_layout()
plt.savefig("blog/chart4_token_efficiency.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 4: Token efficiency saved")

# ── Chart 5: Radar Chart ──────────────────────────────────────────────────────
from matplotlib.patches import FancyArrowPatch
cats_radar = ["Simple", "Multiple", "Parallel", "Parallel\nMultiple"]
N = len(cats_radar)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_facecolor("#FAFAFA")

# Only show top 3 models for clarity
radar_models = ["Granite 4.1 8B", "Qwen 2.5 7B", "Llama 3.1 8B"]
for model in radar_models:
    vals = DATA[model][:4]
    vals += vals[:1]
    ax.plot(angles, vals, "o-", linewidth=2.5, label=model, color=PALETTE[model])
    ax.fill(angles, vals, alpha=0.08, color=PALETTE[model])

ax.set_thetagrids(np.degrees(angles[:-1]), cats_radar, fontsize=13)
ax.set_ylim(0, 60)
ax.set_yticks([15, 30, 45, 60])
ax.set_yticklabels(["15%", "30%", "45%", "60%"], fontsize=9, color="#888")
ax.set_title("Top 3 Models — Capability Radar\n(by category accuracy)",
             fontsize=13, pad=25, fontweight="bold")
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("blog/chart5_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 5: Radar chart saved")

print("\n✅ All blog charts saved to blog/")
print("   chart1_accuracy_comparison.png")
print("   chart2_heatmap.png")
print("   chart3_version_delta.png")
print("   chart4_token_efficiency.png")
print("   chart5_radar.png")
