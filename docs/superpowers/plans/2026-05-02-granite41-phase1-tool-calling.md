# Granite 4.1 Phase 1 — Tool Calling Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully automated tool-calling benchmark that runs BFCL v3 (all categories) across 5 Ollama models and produces accuracy, token-efficiency, and latency comparisons — answering whether Granite 4.1 8B improves over Granite 4.0 and beats same-size competitors.

**Architecture:** Each module is independently unit-tested. The Jupyter notebook is the orchestration layer that calls modules in sequence: load → preprocess → infer (with caching) → evaluate → visualise. Results are cached per call to JSON so 5,700+ inference calls survive interruption and resume cleanly.

**Tech Stack:** Python 3.11+, ollama (SDK), datasets (HuggingFace), pandas, matplotlib, seaborn, pytest, jupyter, tqdm

---

## File Map

| File | Responsibility |
|---|---|
| `config.py` | Models list, Ollama URL, paths, constants |
| `data/loader.py` | Download + cache BFCL v3 from HuggingFace; load ground truth |
| `data/preprocessor.py` | Convert BFCL tool schema → OpenAI tools format |
| `inference/runner.py` | Ollama chat calls with per-call JSON caching + retry |
| `inference/parser.py` | Extract tool call (name + args) from Ollama response |
| `evaluation/ast_matcher.py` | Lenient AST comparison: name match + param match |
| `evaluation/metrics.py` | Aggregate accuracy %, token counts, latency per model/category |
| `analysis/report.py` | Build pandas summary DataFrames |
| `analysis/charts.py` | Matplotlib/seaborn charts (bar, box, stacked) |
| `tests/test_preprocessor.py` | Unit tests for schema conversion |
| `tests/test_parser.py` | Unit tests for response parsing |
| `tests/test_ast_matcher.py` | Unit tests for AST comparison logic |
| `tests/test_metrics.py` | Unit tests for metric calculations |
| `phase1_tool_calling.ipynb` | End-to-end notebook: smoke test → full run → charts |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `data/__init__.py`
- Create: `inference/__init__.py`
- Create: `evaluation/__init__.py`
- Create: `analysis/__init__.py`
- Create: `results/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```
ollama>=0.4.0
datasets>=2.19.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
tqdm>=4.66.0
pytest>=8.0.0
ipykernel>=6.29.0
```

- [ ] **Step 2: Install dependencies**

```bash
cd /Users/abivarma/Personal_projects/granite4_1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 3: Create package init files and results dir**

```bash
mkdir -p data inference evaluation analysis tests results
touch data/__init__.py inference/__init__.py evaluation/__init__.py analysis/__init__.py tests/__init__.py results/.gitkeep
```

- [ ] **Step 4: Verify Ollama is running and check available models**

```bash
ollama list
```

Expected output contains models. If granite4.1:8b is missing:
```bash
ollama pull granite4.1:8b
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

For Granite 4.0, search available tags:
```bash
ollama search granite4
```
Use the tag shown (likely `granite4:latest` or `ibm/granite4:latest`). Note the exact tag — you'll put it in config.py in Task 2.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt data/ inference/ evaluation/ analysis/ tests/ results/
git commit -m "chore: project scaffold for granite4.1 benchmark"
```

---

## Task 2: Config

**Files:**
- Create: `config.py`

- [ ] **Step 1: Write config.py**

Replace `"granite4:latest"` with whatever tag `ollama search granite4` returned for Granite 4.0.

```python
from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"

MODELS = [
    {"tag": "granite4.1:8b",  "label": "Granite 4.1 8B"},
    {"tag": "granite4:latest", "label": "Granite 4.0"},   # verify tag from: ollama search granite4
    {"tag": "llama3.1:8b",    "label": "Llama 3.1 8B"},
    {"tag": "qwen2.5:7b",     "label": "Qwen 2.5 7B"},
    {"tag": "mistral:7b",     "label": "Mistral 7B"},
]

BFCL_DATASET_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

BFCL_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
]

RESULTS_DIR = Path("results")
TEMPERATURE = 0
MAX_RETRIES = 1
```

- [ ] **Step 2: Verify config imports cleanly**

```bash
python -c "import config; print([m['tag'] for m in config.MODELS])"
```

Expected: `['granite4.1:8b', 'granite4:latest', 'llama3.1:8b', 'qwen2.5:7b', 'mistral:7b']`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "chore: add model config"
```

---

## Task 3: BFCL Data Loader

**Files:**
- Create: `data/loader.py`

- [ ] **Step 1: Write failing test for loader**

```python
# tests/test_loader.py
import pytest
from data.loader import load_bfcl_category, load_ground_truth

def test_load_simple_category_returns_list():
    samples = load_bfcl_category("simple")
    assert isinstance(samples, list)
    assert len(samples) > 0

def test_sample_has_required_fields():
    samples = load_bfcl_category("simple")
    s = samples[0]
    assert "id" in s
    assert "question" in s        # list of message dicts
    assert "function" in s        # list of function defs (BFCL format)

def test_question_is_list_of_messages():
    samples = load_bfcl_category("simple")
    msgs = samples[0]["question"]
    assert isinstance(msgs, list)
    assert "role" in msgs[0]
    assert "content" in msgs[0]

def test_ground_truth_keys_match_sample_ids():
    samples = load_bfcl_category("simple")
    gt = load_ground_truth("simple")
    sample_ids = {s["id"] for s in samples}
    gt_ids = set(gt.keys())
    # At least 90% of sample IDs should have ground truth
    overlap = len(sample_ids & gt_ids) / len(sample_ids)
    assert overlap >= 0.9
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_loader.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — loader doesn't exist yet.

- [ ] **Step 3: Implement data/loader.py**

```python
from datasets import load_dataset, get_dataset_config_names
from pathlib import Path
import json

_CACHE_DIR = Path(".cache/bfcl")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CONFIG_MAP = {
    "simple":           "BFCL_v3_simple",
    "multiple":         "BFCL_v3_multiple",
    "parallel":         "BFCL_v3_parallel",
    "parallel_multiple": "BFCL_v3_parallel_multiple",
}

def load_bfcl_category(category: str) -> list[dict]:
    """Load BFCL v3 prompts for a given category. Returns list of {id, question, function}."""
    config_name = _CONFIG_MAP[category]
    cache_file = _CACHE_DIR / f"{category}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    ds = load_dataset(
        "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        name=config_name,
        split="test",
        trust_remote_code=True,
    )
    samples = [
        {
            "id": row["id"],
            "question": row["question"],      # list of message dicts
            "function": row["function"],      # list of function defs
        }
        for row in ds
    ]
    cache_file.write_text(json.dumps(samples, indent=2))
    return samples


def load_ground_truth(category: str) -> dict[str, list]:
    """Load ground truth for a category. Returns {id: [gt_call, ...]}."""
    config_name = _CONFIG_MAP[category]
    cache_file = _CACHE_DIR / f"{category}_gt.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    # Ground truth is in the "answers" split of the same config
    ds = load_dataset(
        "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        name=config_name,
        split="test",
        trust_remote_code=True,
    )
    gt = {}
    for row in ds:
        if "ground_truth" in row:
            gt[row["id"]] = row["ground_truth"]
        elif "answer" in row:
            gt[row["id"]] = row["answer"]

    cache_file.write_text(json.dumps(gt, indent=2))
    return gt


def load_all_categories() -> dict[str, tuple[list, dict]]:
    """Returns {category: (samples, ground_truth)} for all 4 categories."""
    from config import BFCL_CATEGORIES
    return {
        cat: (load_bfcl_category(cat), load_ground_truth(cat))
        for cat in BFCL_CATEGORIES
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_loader.py -v
```

Expected: all 4 tests PASS. First run downloads dataset (~30s). Subsequent runs use cache.

**If ground_truth field names differ:** Print `ds[0].keys()` and `ds[0]` to inspect actual field names, then update `load_ground_truth` accordingly.

- [ ] **Step 5: Commit**

```bash
git add data/loader.py tests/test_loader.py
git commit -m "feat: BFCL v3 data loader with local cache"
```

---

## Task 4: Tool Schema Preprocessor

**Files:**
- Create: `data/preprocessor.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preprocessor.py
from data.preprocessor import bfcl_to_openai_tools

def test_single_function_wrapped_correctly():
    bfcl_fn = {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
    result = bfcl_to_openai_tools([bfcl_fn])
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["description"] == "Get weather for a city"
    assert "location" in result[0]["function"]["parameters"]["properties"]

def test_multiple_functions_all_converted():
    fns = [
        {"name": "fn_a", "description": "A", "parameters": {"type": "object", "properties": {}}},
        {"name": "fn_b", "description": "B", "parameters": {"type": "object", "properties": {}}},
    ]
    result = bfcl_to_openai_tools(fns)
    assert len(result) == 2
    assert result[0]["function"]["name"] == "fn_a"
    assert result[1]["function"]["name"] == "fn_b"

def test_empty_list_returns_empty():
    assert bfcl_to_openai_tools([]) == []

def test_missing_description_uses_empty_string():
    fn = {"name": "do_thing", "parameters": {"type": "object", "properties": {}}}
    result = bfcl_to_openai_tools([fn])
    assert result[0]["function"]["description"] == ""
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement data/preprocessor.py**

```python
def bfcl_to_openai_tools(bfcl_functions: list[dict]) -> list[dict]:
    """Convert BFCL function definitions to OpenAI tools schema for Ollama."""
    return [
        {
            "type": "function",
            "function": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for fn in bfcl_functions
    ]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add data/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: BFCL to OpenAI tool schema converter"
```

---

## Task 5: Response Parser

**Files:**
- Create: `inference/parser.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_parser.py
from unittest.mock import MagicMock
from inference.parser import extract_tool_calls

def _make_response(tool_calls_data):
    """Build a mock Ollama response with tool_calls."""
    response = MagicMock()
    if tool_calls_data is None:
        response.message.tool_calls = None
        return response
    calls = []
    for name, args in tool_calls_data:
        tc = MagicMock()
        tc.function.name = name
        tc.function.arguments = args
        calls.append(tc)
    response.message.tool_calls = calls
    return response

def test_single_tool_call_extracted():
    resp = _make_response([("get_weather", {"location": "Boston", "unit": "celsius"})])
    result = extract_tool_calls(resp)
    assert len(result) == 1
    assert result[0]["name"] == "get_weather"
    assert result[0]["arguments"]["location"] == "Boston"

def test_multiple_tool_calls_extracted():
    resp = _make_response([
        ("fn_a", {"x": 1}),
        ("fn_b", {"y": 2}),
    ])
    result = extract_tool_calls(resp)
    assert len(result) == 2
    assert result[0]["name"] == "fn_a"
    assert result[1]["name"] == "fn_b"

def test_no_tool_calls_returns_empty_list():
    resp = _make_response(None)
    result = extract_tool_calls(resp)
    assert result == []

def test_empty_tool_calls_list_returns_empty():
    resp = _make_response([])
    result = extract_tool_calls(resp)
    assert result == []
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_parser.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement inference/parser.py**

```python
def extract_tool_calls(response) -> list[dict]:
    """Extract tool calls from an Ollama chat response.
    
    Returns list of {"name": str, "arguments": dict}.
    Returns [] if model produced no tool calls.
    """
    tool_calls = getattr(response.message, "tool_calls", None)
    if not tool_calls:
        return []
    return [
        {
            "name": tc.function.name,
            "arguments": tc.function.arguments if isinstance(tc.function.arguments, dict) else {},
        }
        for tc in tool_calls
    ]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_parser.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add inference/parser.py tests/test_parser.py
git commit -m "feat: Ollama response tool call parser"
```

---

## Task 6: Inference Runner with Caching

**Files:**
- Create: `inference/runner.py`

- [ ] **Step 1: Write inference/runner.py**

```python
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
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                result["error"] = str(e)

    result_file.write_text(json.dumps(result, indent=2))
    return result


def run_category(model_tag: str, samples: list[dict], progress=None) -> list[dict]:
    """Run all samples in a category. Returns list of results."""
    results = []
    for sample in samples:
        r = run_single(model_tag, sample)
        results.append(r)
        if progress:
            progress.update(1)
    return results
```

- [ ] **Step 2: Smoke test — run 2 prompts against granite4.1:8b only**

In a Python shell or notebook cell:
```python
from data.loader import load_bfcl_category
from inference.runner import run_single

samples = load_bfcl_category("simple")[:2]
for s in samples:
    r = run_single("granite4.1:8b", s)
    print(r["id"], r["tool_calls"], r["completion_tokens"], r["latency_ms"])
```

Expected: two results printed with non-empty `tool_calls`, positive token counts, and latency > 0. Files appear in `results/granite4_1_8b/`.

- [ ] **Step 3: Commit**

```bash
git add inference/runner.py
git commit -m "feat: inference runner with per-call JSON caching"
```

---

## Task 7: AST Matcher

**Files:**
- Create: `evaluation/ast_matcher.py`
- Create: `tests/test_ast_matcher.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ast_matcher.py
from evaluation.ast_matcher import match_single_call, MatchResult

# --- Function name ---

def test_correct_name_and_params():
    predicted = {"name": "get_weather", "arguments": {"location": "Boston", "unit": "celsius"}}
    ground_truth = {"get_weather": {"location": ["Boston"], "unit": ["celsius"]}}
    r = match_single_call(predicted, ground_truth)
    assert r.name_correct is True
    assert r.params_correct is True
    assert r.full_match is True

def test_wrong_function_name():
    predicted = {"name": "get_forecast", "arguments": {"location": "Boston"}}
    ground_truth = {"get_weather": {"location": ["Boston"]}}
    r = match_single_call(predicted, ground_truth)
    assert r.name_correct is False
    assert r.full_match is False

def test_correct_name_wrong_param_value():
    predicted = {"name": "get_weather", "arguments": {"location": "London", "unit": "celsius"}}
    ground_truth = {"get_weather": {"location": ["Boston"], "unit": ["celsius"]}}
    r = match_single_call(predicted, ground_truth)
    assert r.name_correct is True
    assert r.params_correct is False
    assert r.full_match is False

# --- Lenient type matching ---

def test_string_matches_single_item_list():
    predicted = {"name": "fn", "arguments": {"x": "hello"}}
    ground_truth = {"fn": {"x": ["hello"]}}
    r = match_single_call(predicted, ground_truth)
    assert r.params_correct is True

def test_int_matches_float():
    predicted = {"name": "fn", "arguments": {"n": 42}}
    ground_truth = {"fn": {"n": [42.0]}}
    r = match_single_call(predicted, ground_truth)
    assert r.params_correct is True

def test_missing_required_param():
    predicted = {"name": "get_weather", "arguments": {}}
    ground_truth = {"get_weather": {"location": ["Boston"]}}
    r = match_single_call(predicted, ground_truth)
    assert r.params_correct is False

def test_null_predicted_is_full_miss():
    r = match_single_call(None, {"get_weather": {"location": ["Boston"]}})
    assert r.name_correct is False
    assert r.full_match is False
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_ast_matcher.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement evaluation/ast_matcher.py**

```python
from dataclasses import dataclass

@dataclass
class MatchResult:
    name_correct: bool
    params_correct: bool

    @property
    def full_match(self) -> bool:
        return self.name_correct and self.params_correct


def _normalize(value) -> set:
    """Normalize a predicted value for lenient comparison against GT list."""
    if isinstance(value, list):
        return {str(v).strip().lower() for v in value}
    return {str(value).strip().lower()}


def _values_match(predicted_val, gt_list: list) -> bool:
    """Lenient match: predicted value against ground truth list of acceptable values."""
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
    """Match a list of predicted calls against a list of GT calls (unordered).
    
    Each GT call must be matched by at least one predicted call.
    Returns one MatchResult per GT entry.
    """
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_ast_matcher.py -v
```

Expected: all 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/ast_matcher.py tests/test_ast_matcher.py
git commit -m "feat: lenient AST matcher for BFCL tool call evaluation"
```

---

## Task 8: Metrics Computation

**Files:**
- Create: `evaluation/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
from evaluation.metrics import compute_category_metrics, CategoryMetrics

def _make_result(name_correct, full_match, completion_tokens, latency_ms, error=None):
    return {
        "name_correct": name_correct,
        "full_match": full_match,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "error": error,
    }

def test_perfect_accuracy():
    evals = [
        _make_result(True, True, 20, 100),
        _make_result(True, True, 30, 200),
    ]
    m = compute_category_metrics(evals)
    assert m.name_acc == 1.0
    assert m.full_acc == 1.0

def test_zero_accuracy():
    evals = [
        _make_result(False, False, 10, 150),
        _make_result(False, False, 10, 150),
    ]
    m = compute_category_metrics(evals)
    assert m.name_acc == 0.0
    assert m.full_acc == 0.0

def test_token_efficiency_only_on_correct():
    evals = [
        _make_result(True, True, 20, 100),   # correct, 20 tokens
        _make_result(True, False, 50, 200),  # wrong params, 50 tokens — excluded
        _make_result(False, False, 80, 300), # wrong name — excluded
    ]
    m = compute_category_metrics(evals)
    assert m.avg_tokens_correct == 20.0      # only the 20-token correct call

def test_null_rate():
    evals = [
        _make_result(False, False, 0, 0, error="no tool call"),
        _make_result(True, True, 20, 100),
    ]
    m = compute_category_metrics(evals)
    assert m.null_rate == 0.5

def test_median_latency():
    evals = [
        _make_result(True, True, 10, 100),
        _make_result(True, True, 10, 200),
        _make_result(True, True, 10, 300),
    ]
    m = compute_category_metrics(evals)
    assert m.median_latency_ms == 200.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement evaluation/metrics.py**

```python
from dataclasses import dataclass
import statistics
from evaluation.ast_matcher import match_single_call, match_parallel_calls, MatchResult


@dataclass
class CategoryMetrics:
    total: int
    name_acc: float          # fraction with correct function name
    full_acc: float          # fraction with full AST match
    null_rate: float         # fraction with no tool call / error
    avg_tokens_all: float    # avg completion tokens across all calls
    avg_tokens_correct: float  # avg completion tokens only on full-match calls
    median_latency_ms: float
    p95_latency_ms: float


def compute_category_metrics(evaluated: list[dict]) -> CategoryMetrics:
    """Compute metrics from a list of evaluated result dicts.
    
    Each dict must have: name_correct, full_match, completion_tokens, latency_ms, error.
    """
    n = len(evaluated)
    if n == 0:
        return CategoryMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    name_correct = sum(1 for e in evaluated if e["name_correct"])
    full_match   = sum(1 for e in evaluated if e["full_match"])
    nulls        = sum(1 for e in evaluated if e.get("error") or not e.get("name_correct") and e["completion_tokens"] == 0)

    all_tokens     = [e["completion_tokens"] for e in evaluated]
    correct_tokens = [e["completion_tokens"] for e in evaluated if e["full_match"]]
    latencies      = sorted(e["latency_ms"] for e in evaluated)

    p95_idx = int(0.95 * n)

    return CategoryMetrics(
        total=n,
        name_acc=name_correct / n,
        full_acc=full_match / n,
        null_rate=nulls / n,
        avg_tokens_all=statistics.mean(all_tokens) if all_tokens else 0,
        avg_tokens_correct=statistics.mean(correct_tokens) if correct_tokens else 0,
        median_latency_ms=statistics.median(latencies) if latencies else 0,
        p95_latency_ms=latencies[p95_idx] if latencies else 0,
    )


def evaluate_results(results: list[dict], ground_truth: dict, category: str) -> list[dict]:
    """Join inference results with ground truth and produce evaluated list.
    
    Returns list of dicts with all fields needed by compute_category_metrics.
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
                "latency_ms": r.get("latency_ms", 0),
                "error": r.get("error", "no_ground_truth"),
            })
            continue

        if is_parallel:
            # gt is a list of expected calls
            gt_list = gt if isinstance(gt, list) else [gt]
            match_results = match_parallel_calls(predicted_calls, gt_list)
            name_correct = all(m.name_correct for m in match_results)
            full_match   = all(m.full_match for m in match_results)
        else:
            # gt is a single expected call (list of length 1 or a dict)
            gt_single = gt[0] if isinstance(gt, list) and len(gt) > 0 else gt
            pred = predicted_calls[0] if predicted_calls else None
            match_result = match_single_call(pred, gt_single)
            name_correct = match_result.name_correct
            full_match   = match_result.full_match

        evaluated.append({
            "id": sample_id,
            "name_correct": name_correct,
            "full_match": full_match,
            "completion_tokens": r.get("completion_tokens", 0),
            "latency_ms": r.get("latency_ms", 0),
            "error": None,
        })

    return evaluated
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Run all tests to confirm nothing broken**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: metrics computation and result evaluation"
```

---

## Task 9: Analysis & Charts

**Files:**
- Create: `analysis/report.py`
- Create: `analysis/charts.py`

- [ ] **Step 1: Implement analysis/report.py**

```python
import pandas as pd
from evaluation.metrics import CategoryMetrics


def build_accuracy_table(
    results: dict[str, dict[str, CategoryMetrics]]
) -> pd.DataFrame:
    """Build accuracy table.
    
    results: {model_label: {category: CategoryMetrics}}
    Returns DataFrame indexed by model, columns = categories + Overall.
    """
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


def build_token_efficiency_table(
    results: dict[str, dict[str, CategoryMetrics]]
) -> pd.DataFrame:
    """Build token efficiency table (avg completion tokens on correct calls)."""
    rows = []
    for model_label, cat_metrics in results.items():
        all_correct_tokens = []
        for m in cat_metrics.values():
            if m.avg_tokens_correct > 0:
                all_correct_tokens.extend([m.avg_tokens_correct] * int(m.full_acc * m.total))
        avg = sum(all_correct_tokens) / len(all_correct_tokens) if all_correct_tokens else 0
        rows.append({"Model": model_label, "Avg Tokens (Correct Calls)": round(avg, 1)})
    return pd.DataFrame(rows).set_index("Model").sort_values("Avg Tokens (Correct Calls)")


def build_version_delta_table(
    results: dict[str, dict[str, CategoryMetrics]],
    model_new: str,
    model_old: str,
) -> pd.DataFrame:
    """Show accuracy delta between two model versions per category."""
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
```

- [ ] **Step 2: Implement analysis/charts.py**

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from evaluation.metrics import CategoryMetrics

PALETTE = sns.color_palette("colorblind", 5)


def _model_colors(model_labels: list[str]) -> dict:
    return {label: PALETTE[i] for i, label in enumerate(model_labels)}


def plot_accuracy_by_category(
    results: dict[str, dict[str, CategoryMetrics]],
    save_path: str = "results/accuracy_by_category.png",
):
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
    plt.show()
    print(f"Saved: {save_path}")


def plot_token_efficiency(
    results: dict[str, dict[str, CategoryMetrics]],
    save_path: str = "results/token_efficiency.png",
):
    models = list(results.keys())
    colors = _model_colors(models)
    avg_tokens = []
    for model in models:
        all_t = [m.avg_tokens_correct for m in results[model].values() if m.avg_tokens_correct > 0]
        avg_tokens.append(sum(all_t) / len(all_t) if all_t else 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(models, avg_tokens, color=[colors[m] for m in models])
    ax.set_xlabel("Avg Completion Tokens (Correct Calls Only)")
    ax.set_title("Token Efficiency — Fewer is Better")
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def plot_latency_comparison(
    raw_results: dict[str, list[dict]],
    save_path: str = "results/latency_comparison.png",
):
    """raw_results: {model_label: [inference result dicts]}"""
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [
        [r["latency_ms"] for r in calls if r["latency_ms"] > 0]
        for calls in raw_results.values()
    ]
    ax.boxplot(data, labels=list(raw_results.keys()), vert=False, patch_artist=True)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Response Latency Distribution per Model")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")
```

- [ ] **Step 3: Commit**

```bash
git add analysis/report.py analysis/charts.py
git commit -m "feat: analysis tables and charts"
```

---

## Task 10: Phase 1 Notebook

**Files:**
- Create: `phase1_tool_calling.ipynb`

- [ ] **Step 1: Create notebook with these cells in order**

Open Jupyter:
```bash
source .venv/bin/activate
jupyter notebook
```

Create `phase1_tool_calling.ipynb` with the following cells:

**Cell 1 — Imports & Config**
```python
import sys
sys.path.insert(0, ".")

from config import MODELS, BFCL_CATEGORIES
from data.loader import load_bfcl_category, load_ground_truth
from data.preprocessor import bfcl_to_openai_tools
from inference.runner import run_single, run_category
from evaluation.metrics import evaluate_results, compute_category_metrics, CategoryMetrics
from analysis.report import build_accuracy_table, build_token_efficiency_table, build_version_delta_table
from analysis.charts import plot_accuracy_by_category, plot_token_efficiency, plot_latency_comparison
from tqdm.notebook import tqdm
import pandas as pd

pd.set_option("display.max_colwidth", None)
print("All imports OK")
```

**Cell 2 — Smoke Test (2 samples, 1 model)**
```python
# Verify the full pipeline works before 5-hour run
samples = load_bfcl_category("simple")[:2]
gt = load_ground_truth("simple")

test_model = MODELS[0]["tag"]  # granite4.1:8b
for s in samples:
    r = run_single(test_model, s)
    evaluated = evaluate_results([r], gt, "simple")
    print(f"ID: {r['id']} | tool_calls: {r['tool_calls']} | tokens: {r['completion_tokens']} | latency: {r['latency_ms']:.0f}ms")
    print(f"  name_correct={evaluated[0]['name_correct']} full_match={evaluated[0]['full_match']}")
```

**Cell 3 — Full Benchmark Run (run once, ~3–7 hrs)**
```python
# Skips already-cached results automatically
all_raw_results = {}   # {model_label: {category: [result dicts]}}

for model in MODELS:
    tag, label = model["tag"], model["label"]
    print(f"\n{'='*50}\nRunning: {label} ({tag})\n{'='*50}")
    all_raw_results[label] = {}

    for cat in BFCL_CATEGORIES:
        samples = load_bfcl_category(cat)
        with tqdm(total=len(samples), desc=f"  {cat}") as pbar:
            results = run_category(tag, samples, progress=pbar)
        all_raw_results[label][cat] = results
        print(f"  {cat}: {len(results)} calls done")

print("\nAll inference complete.")
```

**Cell 4 — Evaluate Results**
```python
all_metrics = {}   # {model_label: {category: CategoryMetrics}}

for model in MODELS:
    label = model["label"]
    all_metrics[label] = {}
    for cat in BFCL_CATEGORIES:
        gt = load_ground_truth(cat)
        raw = all_raw_results[label][cat]
        evaluated = evaluate_results(raw, gt, cat)
        all_metrics[label][cat] = compute_category_metrics(evaluated)

print("Evaluation complete.")
```

**Cell 5 — Accuracy Table**
```python
acc_table = build_accuracy_table(all_metrics)
print("=== Full AST Accuracy by Category ===")
display(acc_table)
```

**Cell 6 — Token Efficiency Table**
```python
tok_table = build_token_efficiency_table(all_metrics)
print("=== Token Efficiency (fewer = better) ===")
display(tok_table)
```

**Cell 7 — Version Delta (Granite 4.1 vs 4.0)**
```python
delta_table = build_version_delta_table(
    all_metrics,
    model_new="Granite 4.1 8B",
    model_old="Granite 4.0",
)
print("=== Granite 4.1 vs 4.0 — Accuracy Delta (percentage points) ===")
display(delta_table)
```

**Cell 8 — Charts**
```python
plot_accuracy_by_category(all_metrics)
plot_token_efficiency(all_metrics)

# Flatten raw results per model for latency box plot
flat_raw = {
    label: [r for cat_results in all_raw_results[label].values() for r in cat_results]
    for label in all_raw_results
}
plot_latency_comparison(flat_raw)
```

**Cell 9 — Summary Findings**
```python
# Compute overall accuracy per model
print("=== SUMMARY ===\n")
for model in MODELS:
    label = model["label"]
    total_c = sum(int(m.full_acc * m.total) for m in all_metrics[label].values())
    total_n = sum(m.total for m in all_metrics[label].values())
    overall_acc = total_c / total_n if total_n > 0 else 0
    
    all_correct_tokens = []
    for m in all_metrics[label].values():
        if m.avg_tokens_correct > 0:
            all_correct_tokens.extend([m.avg_tokens_correct] * int(m.full_acc * m.total))
    avg_tok = sum(all_correct_tokens) / len(all_correct_tokens) if all_correct_tokens else 0
    
    print(f"{label:20s}  Overall Acc: {overall_acc*100:5.1f}%  Avg Tokens (correct): {avg_tok:5.1f}")
```

- [ ] **Step 2: Run the smoke test cell (Cell 2) first**

Run only Cell 1 + Cell 2. Verify no errors and reasonable output before proceeding.

- [ ] **Step 3: Run Cells 3–9 for the full benchmark**

This takes 3–7 hours. The run is fully resumable — re-running Cell 3 skips cached results.

- [ ] **Step 4: Cross-validate calibration**

Check Granite 4.1 8B overall accuracy. IBM published BFCL v3 score: **68.27%**.
- If your result is within ±5pp of 68.27% → experiment is correctly calibrated.
- If significantly different: check that `granite4.1:8b` tag is the instruct model, temperature is 0, and tool schemas converted correctly.

- [ ] **Step 5: Final commit**

```bash
git add phase1_tool_calling.ipynb results/
git commit -m "feat: phase 1 tool calling benchmark complete — results and charts"
```

---

## Self-Review

**Spec coverage check:**
- ✅ All 5 models covered (config.py Task 2)
- ✅ All 4 BFCL categories loaded (loader.py Task 3)
- ✅ BFCL → OpenAI schema conversion (preprocessor.py Task 4)
- ✅ Ollama inference with caching (runner.py Task 6)
- ✅ Response parsing (parser.py Task 5)
- ✅ Lenient AST matching, string↔list, int↔float (ast_matcher.py Task 7)
- ✅ All metrics: name acc, full acc, null rate, token efficiency, latency p50/p95 (metrics.py Task 8)
- ✅ Accuracy table, token efficiency table, version delta table (report.py Task 9)
- ✅ 4 charts: accuracy by category, token efficiency, latency box plot (charts.py Task 9)
- ✅ Smoke test before full run (notebook Task 10)
- ✅ Calibration check against IBM's published score (notebook Task 10)
- ✅ All parallel categories handled separately (evaluate_results in metrics.py)

**Type consistency:**
- `match_single_call` takes `dict | None` + `dict` → returns `MatchResult` ✅
- `match_parallel_calls` takes `list[dict]` + `list[dict]` → returns `list[MatchResult]` ✅
- `compute_category_metrics` takes `list[dict]` → returns `CategoryMetrics` ✅
- `evaluate_results` takes `list[dict], dict, str` → returns `list[dict]` ✅
- `build_accuracy_table` takes `dict[str, dict[str, CategoryMetrics]]` → returns `DataFrame` ✅

**No placeholders:** All steps contain actual code. ✅
