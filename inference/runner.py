"""Ollama inference runner and response parser."""
import json
import time
from pathlib import Path
import ollama
from config import RESULTS_DIR, TEMPERATURE, MAX_RETRIES
from data.preprocessor import bfcl_to_openai_tools
from inference.parser import extract_tool_calls


def _result_path(model_tag: str, sample_id: str) -> Path:
    safe_tag = model_tag.replace(":", "_").replace("/", "_")
    path = RESULTS_DIR / safe_tag / f"{sample_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_single(model_tag: str, sample: dict) -> dict:
    """Run one BFCL sample against a model. Returns cached result if available.

    Result schema:
    {
        "id": str,
        "model": str,
        "tool_calls": [{"name": str, "arguments": dict}],
        "prompt_tokens": int,
        "completion_tokens": int,
        "latency_ms": float,
        "error": str | None,
    }
    """
    result_file = _result_path(model_tag, sample["id"])
    if result_file.exists():
        return json.loads(result_file.read_text())

    tools = bfcl_to_openai_tools(sample["function"])
    messages = sample["question"]

    result = {
        "id": sample["id"],
        "model": model_tag,
        "tool_calls": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0.0,
        "error": None,
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            response = ollama.chat(
                model=model_tag,
                messages=messages,
                tools=tools,
                options={"temperature": TEMPERATURE},
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            result["tool_calls"] = extract_tool_calls(response)
            result["prompt_tokens"] = response.prompt_eval_count or 0
            result["completion_tokens"] = response.eval_count or 0
            result["latency_ms"] = latency_ms
            result["error"] = None
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                result["error"] = str(e)

    result_file.write_text(json.dumps(result, indent=2))
    return result


def run_category(model_tag: str, samples: list[dict], progress=None) -> list[dict]:
    """Run all samples in a category for one model. Returns list of results."""
    results = []
    for sample in samples:
        r = run_single(model_tag, sample)
        results.append(r)
        if progress:
            progress.update(1)
    return results
