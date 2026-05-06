"""IFEval inference runner — plain text generation, no tool calling."""
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
import ollama
from config import TEMPERATURE

_RESULTS_BASE = Path(__file__).parent.parent / "results_ifeval"
_TIMEOUT_SECONDS = 120  # per-sample wall-clock limit


def _result_path(model_tag: str, sample_id: str) -> Path:
    safe_tag = model_tag.replace(":", "_").replace("/", "_")
    path = _RESULTS_BASE / safe_tag / f"{sample_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _call_ollama(model_tag: str, prompt: str) -> tuple:
    """Returns (response_text, prompt_tokens, completion_tokens, latency_ms)."""
    t0 = time.perf_counter()
    resp = ollama.chat(
        model=model_tag,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": TEMPERATURE, "num_predict": 1024},
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return (
        resp.message.content or "",
        resp.prompt_eval_count or 0,
        resp.eval_count or 0,
        latency_ms,
    )


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
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_ollama, model_tag, sample["prompt"])
            text, pt, ct, lat = future.result(timeout=_TIMEOUT_SECONDS)
        result["response"] = text
        result["prompt_tokens"] = pt
        result["completion_tokens"] = ct
        result["latency_ms"] = lat
    except FuturesTimeoutError:
        result["error"] = f"timeout after {_TIMEOUT_SECONDS}s"
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
