# Granite 4.1 Benchmark Research — Design Spec
**Date:** 2026-05-02  
**Author:** Research experiment — local Mac M3 36GB  
**Status:** Approved for implementation

---

## Context

IBM released Granite 4.1 (April 2026) claiming that the 8B dense model matches or outperforms Granite 4.0 32B MoE on tool calling and instruction following — without long chain-of-thought reasoning. The core enterprise promise is predictable latency, lower token cost, and competitive accuracy vs same-size open models.

This project validates those claims experimentally using public benchmark datasets with ground truth, running entirely locally via Ollama on a Mac M3 (36GB RAM). No cloud APIs, no manual data collection.

---

## Research Questions

1. **Version progression:** Does Granite 4.1 8B measurably improve over Granite 4.0 on tool calling accuracy and token efficiency?
2. **Competitive standing:** Does Granite 4.1 8B beat or match Llama 3.1 8B, Qwen 2.5 7B, and Mistral 7B on tool calling?
3. **Token efficiency:** Does Granite 4.1 use fewer completion tokens per correct answer than competitors?

---

## Experiment Phases

| Phase | Capability | Dataset | Notebook |
|---|---|---|---|
| **Phase 1** | Tool Calling | BFCL v3 (full) | `phase1_tool_calling.ipynb` |
| **Phase 2** | Instruction Following | IFEval (Google) | `phase2_instruction_following.ipynb` |
| **Phase 3** | Code Generation | HumanEval | `phase3_code_generation.ipynb` |

This spec covers **Phase 1** in full detail.

---

## Phase 1: Tool Calling Benchmark on Full BFCL v3

### Dataset

**Source:** `gorilla-llm/Berkeley-Function-Calling-Leaderboard` (HuggingFace, Apache 2.0)

| Category | Count | What it tests |
|---|---|---|
| Simple | 258 | Single tool, single call |
| Multiple | 1,037 | Pick correct tool from several candidates |
| Parallel | 16 | Fire multiple tool calls simultaneously |
| Parallel-Multiple | 24 | Parallel + multiple candidates |
| **Total** | **~1,335** | Full real-world tool use coverage |

Ground truth format: JSONL matched by `id`, each entry is `{id, ground_truth: [{function_name: {param: [value]}}]}`.

---

### Models

| # | Model | Ollama Tag | Est. Size | Role |
|---|---|---|---|---|
| 1 | Granite 4.1 8B | `granite4.1:8b` | 5.3 GB | Primary subject |
| 2 | Granite 4.0 | `granite4:latest` | ~5–7 GB | Direct version predecessor |
| 3 | Llama 3.1 8B | `llama3.1:8b` | 4.7 GB | Meta baseline |
| 4 | Qwen 2.5 7B | `qwen2.5:7b` | 4.4 GB | Community favourite |
| 5 | Mistral 7B | `mistral:7b` | 4.1 GB | Established open-source baseline |

**Total memory:** ~24 GB — fits in 36 GB.  
**Note:** Granite 4.0 tag must be verified at run time via `ollama search granite4`. Community projects referenced `ibm/granite4:latest`.

---

### Project Structure

```
granite41_benchmark/
├── config.py                        # model tags, Ollama URL, settings
├── data/
│   ├── loader.py                    # download + cache BFCL v3 from HuggingFace
│   └── preprocessor.py             # convert BFCL tool schemas → OpenAI format
├── inference/
│   ├── runner.py                    # Ollama calls with per-call JSON caching
│   └── parser.py                   # extract tool calls from raw model output
├── evaluation/
│   ├── ast_matcher.py              # lenient AST comparison (handles type equivalence)
│   └── metrics.py                  # accuracy %, tokens, latency aggregation
├── analysis/
│   ├── report.py                   # pandas summary tables
│   └── charts.py                   # matplotlib/seaborn visualisations
├── results/
│   └── {model_name}/               # one JSON file per prompt, cached immediately
├── phase1_tool_calling.ipynb        # main notebook: orchestrate + visualise
├── phase2_instruction_following.ipynb
├── phase3_code_generation.ipynb
└── requirements.txt
```

---

### Inference Pipeline

**API:** Ollama Python SDK — `ollama.chat(model, messages, tools=[...], options={"temperature": 0})`

All four Ollama models accept OpenAI-compatible `tools` parameter. The preprocessor converts BFCL's native schema to OpenAI tool schema before each call.

**Per-call flow:**
```
1. Load prompt + tool definitions from BFCL dataset
2. Convert tool definitions → OpenAI tools schema (preprocessor)
3. Call ollama.chat() with tools=, temperature=0
4. Record: raw_response, tool_calls[], prompt_tokens, completion_tokens, latency_ms
5. Write to results/{model_name}/{id}.json  ← cache immediately after each call
6. On retry: if no tool call returned or malformed JSON → retry once → mark null if still failing
```

**Settings:**
- Temperature: 0 (deterministic, reproducible)
- Sequential execution per model (clean latency numbers)
- Resumable: skip calls where `results/{model}/{id}.json` already exists

**Estimated run time:** ~2–4 sec/call × 1,335 prompts × 5 models ≈ 3–7 hours total. Fully resumable.

---

### Evaluation Methodology

**Three-layer AST evaluation:**

| Layer | What's checked | Pass condition |
|---|---|---|
| Function Name | Did model call the right function? | Exact string match |
| Parameter Keys | Did model include all required params? | All required keys present |
| Parameter Values | Are values correct? | Lenient: string↔list, int↔float equivalence |
| **Full Match** | Name + all params correct | All layers pass |

**Lenient matching rationale:** BFCL ground truth stores values as lists `["Boston"]` but models often return strings `"Boston"` — both are semantically correct. Same for numeric type equivalence.

---

### Metrics Collected

| Metric | Formula | What it reveals |
|---|---|---|
| Function Name Accuracy % | correct_name / total | Does model understand the task? |
| Full AST Accuracy % | full_match / total | End-to-end correctness |
| Null Rate % | null_or_malformed / total | Model reliability / robustness |
| Avg Completion Tokens (correct calls) | sum(completion_tokens on correct) / correct | **Token efficiency — IBM's key claim** |
| Avg Completion Tokens (all calls) | sum(completion_tokens) / total | Overall verbosity |
| Median Latency (ms) | median of latency_ms | Speed per call |
| P95 Latency (ms) | 95th percentile of latency_ms | Worst-case response time |

All metrics broken down by: per model × per category (simple, multiple, parallel, parallel-multiple).

---

### Analysis & Output

The notebook produces:

1. **Master accuracy table** — Full AST accuracy % per model × category
2. **Grouped bar chart** — Accuracy by category, one bar per model
3. **Token efficiency bar chart** — Avg completion tokens per correct answer
4. **Latency box plot** — Response time distribution per model
5. **Error breakdown stacked bar** — Wrong function vs wrong params vs null, per model
6. **Version delta table** — Granite 4.1 minus Granite 4.0 accuracy, per category
7. **Summary verdict** — Structured findings answering the 3 research questions

---

### Dependencies

```
ollama              # Ollama Python SDK
datasets            # HuggingFace datasets
pandas
matplotlib
seaborn
jupyter
tqdm                # progress bars for long inference runs
```

---

### Verification Plan

1. Run `ollama list` to confirm all 5 models are pulled
2. Run smoke test: 5 prompts (one per model) before full run
3. Check `results/` directory populates correctly after smoke test
4. Run full benchmark, monitor via tqdm progress bar
5. Run evaluation notebook — verify accuracy numbers are in plausible range (50–90%)
6. Cross-check Granite 4.1 8B accuracy against IBM's published BFCL v3 score of 68.27%
7. If own result is within ±5% of 68.27 → experiment is correctly calibrated

---

## Future Phases (brief)

**Phase 2 — Instruction Following (IFEval)**  
Dataset: `google/IFEval` (541 prompts, Apache 2.0). Evaluation: Python heuristics, zero human judges. Same 5 models. Separate notebook.

**Phase 3 — Code Generation (HumanEval)**  
Dataset: `openai/openai_humaneval` (164 problems). Evaluation: execute unit tests, pass@1. Requires sandboxed code execution. Separate notebook.
