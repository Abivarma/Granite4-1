"""IFEval dataset loader for instruction following benchmark."""
from datasets import load_dataset
from pathlib import Path
import json

_CACHE_FILE = Path(__file__).parent.parent / ".cache" / "ifeval.json"


def load_ifeval() -> list[dict]:
    """Load IFEval dataset. Returns list of {id, prompt, instruction_id_list, kwargs}."""
    if _CACHE_FILE.exists():
        return json.loads(_CACHE_FILE.read_text())

    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("google/IFEval", split="train")
    samples = []
    for i, row in enumerate(ds):
        samples.append({
            "id": f"ifeval_{i}",
            "prompt": row["prompt"],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": row["kwargs"],
        })
    _CACHE_FILE.write_text(json.dumps(samples, indent=2))
    return samples
