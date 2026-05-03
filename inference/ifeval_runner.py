"""IFEval inference runner — plain text generation, no tool calling."""
import json
import time
from pathlib import Path
import ollama
from config import TEMPERATURE

_RESULTS_BASE = Path(__file__).parent.parent / "results_ifeval"


def _result_path(model_tag: str, sample_id: str) -> Path:
    safe_tag = model_tag.replace(":", "_").replace("/", "_")
    path = _RESULTS_BASE / safe_tag / f"{sample_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_ifeval_single(model_tag: str, sample: dict) -> dict:
    """Run one IFEval prompt. Returns cached result if available.

    Result: {id, model, response, prompt_tokens, completion_tokens, latency_ms, error}
    """
    result_file = _result_path(model_tag, sample["id"])
    if result_file.exists():
        return json.loads(result_file.read_text())

    result = {
        "id": sample["id"],
        "model": model_tag,
        "response": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0.0,
        "error": None,
    }

    try:
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=model_tag,
            messages=[{"role": "user", "content": sample["prompt"]}],
            options={"temperature": TEMPERATURE},
        )
        result["response"] = resp.message.content or ""
        result["prompt_tokens"] = resp.prompt_eval_count or 0
        result["completion_tokens"] = resp.eval_count or 0
        result["latency_ms"] = (time.perf_counter() - t0) * 1000
    except Exception as e:
        result["error"] = str(e)

    result_file.write_text(json.dumps(result, indent=2))
    return result


def run_ifeval_category(model_tag: str, samples: list[dict], progress=None) -> list[dict]:
    results = []
    for sample in samples:
        r = run_ifeval_single(model_tag, sample)
        results.append(r)
        if progress:
            progress.update(1)
    return results
