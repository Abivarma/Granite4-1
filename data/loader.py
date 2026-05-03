"""Data loading and preprocessing utilities for BFCL v3 benchmark."""

from pathlib import Path
import json
from huggingface_hub import hf_hub_download

_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "bfcl"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

_CONFIG_MAP = {
    "simple":            "BFCL_v3_simple.json",
    "multiple":          "BFCL_v3_multiple.json",
    "parallel":          "BFCL_v3_parallel.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple.json",
}

_FILE_MAP = _CONFIG_MAP

_GT_FILE_MAP = {
    "simple":            "possible_answer/BFCL_v3_simple.json",
    "multiple":          "possible_answer/BFCL_v3_multiple.json",
    "parallel":          "possible_answer/BFCL_v3_parallel.json",
    "parallel_multiple": "possible_answer/BFCL_v3_parallel_multiple.json",
}


def _download_jsonl(filename: str) -> list[dict]:
    """Download a JSONL file from the BFCL HuggingFace dataset."""
    path = hf_hub_download(
        repo_id=_REPO_ID,
        filename=filename,
        repo_type="dataset",
    )
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_bfcl_category(category: str) -> list[dict]:
    """Load BFCL v3 prompts for a category. Returns list of {id, question, function}."""
    if category not in _CONFIG_MAP:
        raise ValueError(f"Unknown category {category!r}. Valid: {list(_CONFIG_MAP)}")
    cache_file = _CACHE_DIR / f"{category}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    filename = _FILE_MAP[category]
    rows = _download_jsonl(filename)

    samples = []
    for row in rows:
        # question is a list of turns; each turn is a list of messages.
        # Flatten to the first turn's messages for single-turn categories.
        question = row["question"]
        if question and isinstance(question[0], list):
            question = question[0]
        samples.append({
            "id": row["id"],
            "question": question,
            "function": row["function"],
        })

    cache_file.write_text(json.dumps(samples, indent=2))
    return samples


def load_ground_truth(category: str) -> dict[str, list]:
    """Load ground truth for a category. Returns {id: [gt_call, ...]}."""
    if category not in _CONFIG_MAP:
        raise ValueError(f"Unknown category {category!r}. Valid: {list(_CONFIG_MAP)}")
    cache_file = _CACHE_DIR / f"{category}_gt.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    filename = _GT_FILE_MAP[category]
    rows = _download_jsonl(filename)

    gt = {}
    for row in rows:
        row_id = row.get("id", "")
        if "ground_truth" in row and row["ground_truth"]:
            gt[row_id] = row["ground_truth"]
        elif "answer" in row and row["answer"]:
            gt[row_id] = row["answer"]

    cache_file.write_text(json.dumps(gt, indent=2))
    return gt


def load_all_categories() -> dict[str, tuple[list, dict]]:
    """Returns {category: (samples, ground_truth)} for all 4 categories."""
    from config import BFCL_CATEGORIES
    return {cat: (load_bfcl_category(cat), load_ground_truth(cat)) for cat in BFCL_CATEGORIES}
