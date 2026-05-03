"""IFEval evaluation: check model responses against instruction constraints."""
from dataclasses import dataclass


@dataclass
class IFEvalResult:
    prompt_strict: bool   # ALL instructions in prompt satisfied (strict)
    prompt_loose: bool    # ALL instructions satisfied (loose — minor formatting)
    instruction_results: list[bool]  # per-instruction pass/fail


@dataclass
class IFEvalMetrics:
    total_prompts: int
    prompt_strict_acc: float
    prompt_loose_acc: float
    instruction_strict_acc: float
    instruction_loose_acc: float
    avg_tokens: float
    median_latency_ms: float


def evaluate_ifeval_response(response: str, instruction_id_list: list, kwargs: list) -> IFEvalResult:
    """Check a model response against all instructions for one prompt."""
    try:
        from evaluation import ifeval_instructions_registry
        checker = ifeval_instructions_registry.INSTRUCTION_DICT
    except ImportError:
        # Fallback: mark all as False if verifier unavailable
        n = len(instruction_id_list)
        return IFEvalResult(False, False, [False] * n)

    instruction_results = []
    for instr_id, kwarg in zip(instruction_id_list, kwargs):
        try:
            verifier_cls = checker[instr_id]
            verifier = verifier_cls(**kwarg)
            passed = verifier.check_following(response)
            instruction_results.append(bool(passed))
        except Exception:
            instruction_results.append(False)

    all_pass = all(instruction_results)
    return IFEvalResult(
        prompt_strict=all_pass,
        prompt_loose=all_pass,  # simplified: use same for loose
        instruction_results=instruction_results,
    )


def compute_ifeval_metrics(results: list[dict], samples: list[dict]) -> IFEvalMetrics:
    """Compute IFEval metrics from inference results and dataset samples."""
    import statistics
    sample_map = {s["id"]: s for s in samples}

    prompt_strict_passes = 0
    all_instruction_results = []
    tokens = []
    latencies = []

    for r in results:
        sample = sample_map.get(r["id"])
        if not sample or r.get("error"):
            n_instr = len(sample["instruction_id_list"]) if sample else 1
            all_instruction_results.extend([False] * n_instr)
            continue

        eval_result = evaluate_ifeval_response(
            r.get("response", ""),
            sample["instruction_id_list"],
            sample["kwargs"],
        )
        if eval_result.prompt_strict:
            prompt_strict_passes += 1
        all_instruction_results.extend(eval_result.instruction_results)

        if r.get("completion_tokens", 0) > 0:
            tokens.append(r["completion_tokens"])
        if r.get("latency_ms", 0) > 0:
            latencies.append(r["latency_ms"])

    n = len(results)
    n_instr = len(all_instruction_results)

    return IFEvalMetrics(
        total_prompts=n,
        prompt_strict_acc=prompt_strict_passes / n if n else 0,
        prompt_loose_acc=prompt_strict_passes / n if n else 0,
        instruction_strict_acc=sum(all_instruction_results) / n_instr if n_instr else 0,
        instruction_loose_acc=sum(all_instruction_results) / n_instr if n_instr else 0,
        avg_tokens=statistics.mean(tokens) if tokens else 0,
        median_latency_ms=statistics.median(latencies) if latencies else 0,
    )
