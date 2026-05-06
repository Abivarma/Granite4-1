# I Ran IBM's New Granite 4.1 Against 4 Other LLMs on My MacBook M3 — Here's the Honest Truth

**Testing IBM's biggest claim: does a small model really beat models 4x its size at tool calling?**

---

*Everything in this post was run locally on a MacBook M3 (36GB RAM) using Ollama. No cloud APIs. No sponsored benchmarks. Just real code and real results.*

---

## Why Am I Even Doing This?

On April 29, 2026, IBM quietly dropped Granite 4.1 — a new family of open-source language models in 3B, 8B, and 30B sizes. No flashy event. No viral demo. Just a technical blog post and a HuggingFace upload.

But inside that announcement was a claim that made me stop scrolling:

> *"The Granite 4.1 8B instruct model consistently matches or outperforms the Granite 4.0 32B MoE model."*

An 8 billion parameter model beating a 32 billion parameter model? On tool calling — the backbone of every modern AI agent?

That's a 4x size advantage claimed away. And IBM said they did it without *long chain-of-thought reasoning* — meaning the model gives you answers fast, without burning through thousands of tokens "thinking out loud" before responding.

I had to test this myself.

Not because I doubted IBM. But because **the only benchmarks that matter are the ones you run yourself** on your own hardware, with your own data.

---

## What Is Tool Calling? (And Why Does It Matter)

Before we get into numbers, let me explain what "tool calling" actually means — because it's the foundation of almost every useful AI application being built today.

Imagine you're building an AI assistant that can answer "What's the weather in Boston?" To do this, the model doesn't magically know the weather. Instead, you give it access to a `get_weather(location, unit)` function. The model reads your question and outputs a structured function call:

```json
{
  "name": "get_weather",
  "arguments": {"location": "Boston", "unit": "celsius"}
}
```

Your code then actually calls the API and returns the result. The model orchestrates the logic; your code does the execution.

This pattern — called **tool calling** or **function calling** — is how every production AI agent works. Customer service bots, code assistants, research agents, invoice processors — they all rely on the model's ability to reliably call the right function with the right parameters.

**If a model is bad at tool calling, it's bad at being an agent. Full stop.**

---

## The Benchmark: BFCL v3

To test tool calling, I used the **Berkeley Function Calling Leaderboard v3 (BFCL v3)** — the industry standard benchmark developed at UC Berkeley. It's publicly available, has ground truth answers, and evaluates models across four difficulty levels:

| Category | Prompts | What the Model Must Do |
|---|---|---|
| **Simple** | 400 | Given 1 tool, call it correctly |
| **Multiple** | 200 | Choose the right tool from 5–10 candidates |
| **Parallel** | 200 | Fire 2+ tool calls simultaneously |
| **Parallel Multiple** | 200 | Simultaneously choose + call multiple tools |
| **Total** | **1,000** | — |

The "Multiple" category is where real capability shows — it's not enough to know *how* to call a function, the model must figure out *which* function to call from a list of plausible options.

---

## The Setup

**Hardware:** MacBook M3, 36GB unified memory  
**Platform:** Ollama (local inference — no cloud)  
**Models tested:**

| Model | Size | Role |
|---|---|---|
| Granite 4.1 8B | 5.3 GB | IBM's latest — the subject |
| Granite 4.0 | 2.1 GB | IBM's previous version — direct comparison |
| Llama 3.1 8B | 4.9 GB | Meta's flagship 8B — industry baseline |
| Qwen 2.5 7B | 4.7 GB | Alibaba's model — community favourite |
| Mistral 7B | 4.4 GB | Classic open-source baseline |

I ran **1,000 prompts per model = 5,000 total inference calls**, each cached to disk so the run is resumable. Every result is a JSON file. No results were thrown away or cherry-picked.

Evaluation uses **AST matching** — the model's function call is parsed and compared against ground truth. Scoring is lenient for types (string `"42"` matches integer `42`) but strict on function names and required parameters.

---

## The Results

Let's start with the big picture.

> 📊 **[Insert chart1_accuracy_comparison.png here]**
> *Full AST Accuracy across all categories. IBM's published score of 68.27% shown as the red dashed line.*

### Overall Accuracy Rankings

| Model | Overall Accuracy |
|---|---|
| 🥇 Qwen 2.5 7B | **42.8%** |
| 🥈 Granite 4.1 8B | **42.2%** |
| 🥉 Llama 3.1 8B | **40.4%** |
| Mistral 7B | 34.9% |
| Granite 4.0 | 34.8% |

The gap between 1st and 2nd is **0.6 percentage points**. That's statistical noise. Granite 4.1 and Qwen 2.5 are essentially tied at the top.

---

## Finding #1: The Version Improvement Is Real

The most defensible finding from this experiment is the Granite 4.1 vs 4.0 comparison — because it's apples-to-apples from the same vendor.

> 📊 **[Insert chart3_version_delta.png here]**
> *Granite 4.1 8B vs Granite 4.0 accuracy improvement per category.*

| Category | Granite 4.0 | Granite 4.1 8B | Improvement |
|---|---|---|---|
| Simple | 43.0% | 50.2% | **+7.2pp** |
| Multiple | 39.5% | 46.5% | **+7.0pp** |
| Parallel | 24.2% | 30.0% | **+5.8pp** |
| Parallel Multiple | 34.5% | 47.0% | **+12.5pp** |

Granite 4.1 improved over Granite 4.0 in **every single category**. The biggest jump — +12.5 percentage points — is on Parallel Multiple, the hardest category. That's not random noise. That's architectural improvement.

IBM's claim that the new training pipeline (with their 4-stage RLHF process and improved SFT data curation) made a real difference? **Verified.**

---

## Finding #2: The Competitive Standing

> 📊 **[Insert chart5_radar.png here]**
> *Capability radar for top 3 models across all 4 difficulty categories.*

Looking at the radar chart, a few patterns emerge:

- **Granite 4.1 and Qwen 2.5** have very similar capability shapes — strong at Simple and Parallel Multiple, moderate on Parallel
- **Llama 3.1** is notably stronger on Multiple (51.5% — best of all 5 models) but drops sharply on Parallel
- **The Parallel category is where everyone suffers** — best score is Granite's 30%, meaning even the top model fails 70% of simultaneous multi-call tasks

> 📊 **[Insert chart2_heatmap.png here]**
> *Accuracy heatmap — darker = better. Read across rows to compare models.*

The heatmap makes the pattern immediately clear:
- The top half (Granite 4.1, Llama, Qwen) clusters around 44–53% on most categories
- Parallel is the universal weakness — the entire column is pale
- Granite 4.0 and Mistral form a clearly weaker tier, especially on the harder categories

---

## Finding #3: Token Efficiency — The Surprise Result

IBM made a specific claim: Granite 4.1 uses dramatically fewer tokens than competitors on typical enterprise workloads. They cited a case where Granite used 4M tokens vs Qwen's 78M on an intelligence index.

I measured this directly — **average completion tokens only on calls the model actually got right:**

> 📊 **[Insert chart4_token_efficiency.png here]**
> *Fewer tokens = cheaper in production. Only counting correct answers.*

| Model | Avg Tokens (Correct Calls) |
|---|---|
| Llama 3.1 8B | **56.7** ← Most efficient |
| Granite 4.0 | 62.1 |
| Granite 4.1 8B | 63.5 |
| Qwen 2.5 7B | 64.3 |
| Mistral 7B | **104.8** ← Least efficient |

**This is where IBM's narrative doesn't hold on this benchmark.** Llama is the most token-efficient. Granite 4.1 actually uses *more* tokens than Granite 4.0.

The good news: Granite 4.1, Granite 4.0, Llama, and Qwen are all tightly clustered between 56–64 tokens. The real outlier is Mistral at 104.8 — which is partly explained by Mistral generating verbose text without always producing clean tool calls.

IBM's token efficiency advantage may be more visible on instruction-following and chat tasks than on structured tool calling. Phase 2 will test that.

---

## Finding #4: Mistral 7B Fundamentally Can't Do Parallel Calling

This wasn't a marginal difference — Mistral would simply hang indefinitely on complex parallel tool-calling prompts. The model kept generating tokens without ever producing a structured function call. We had to mark 59 of 200 parallel_multiple samples as timeouts.

The data is honest about this: Mistral's 21.2% on Parallel and 29.5% on Parallel Multiple reflects a real capability gap, not a bad prompt. **Mistral 7B is not the right model if you need agents that can fire multiple tool calls in a single response.**

---

## The Calibration Gap — Being Honest About the Numbers

IBM published a BFCL v3 score of **68.27%** for Granite 4.1 8B. Our measurement gives **42.2%**.

That's a 26-point gap that deserves an explanation rather than being swept under the rug.

The gap comes from three factors:

1. **Different dataset split.** IBM tested on BFCL v3's official split: 258 Simple / 1,037 Multiple / 16 Parallel / 24 Parallel-Multiple. Our HuggingFace download gave us a different distribution: 400 / 200 / 200 / 200. More parallel samples — the hardest categories — drags down the average.

2. **Official evaluation script.** IBM used Berkeley's official BFCL evaluator which has additional normalisation rules. Our custom AST matcher is strict and correct, but will differ in edge cases.

3. **The relative comparisons are still valid.** All 5 models faced exactly the same prompts. Rankings, deltas, and category breakdowns are reliable and directly comparable.

Think of it this way: a rugby player might run 100m in 12 seconds on a wet grass field (our test) vs 11 seconds on a professional track (IBM's test). The conditions differ — but their *relative speed compared to other players tested on the same field* is real data.

---

## What These Results Actually Mean

**If you're building tool-calling agents and can only run an 8B model locally:**

- Granite 4.1 and Qwen 2.5 are your top options — virtually tied
- Llama 3.1 is strong on "pick the right tool" (Multiple category) but struggles with simultaneous calls
- Avoid Mistral 7B for anything that needs parallel tool calls
- Granite 4.1 is meaningfully better than Granite 4.0 — the upgrade is worth it

**If you care about IBM's specific claims:**

- ✅ "4.1 improves over 4.0" — true, proven across all 4 categories
- ✅ "Competitive with same-size open models" — true, top of the pack at 8B
- ⚠️ "Token efficient" — not clearly visible on tool calling; needs more testing on chat tasks
- ❌ "68.27% BFCL score" — reproducible only with the official eval script and dataset split

---

## What's Next

This is **Phase 1 of a larger experiment**. The three-phase research plan:

| Phase | Capability | Dataset | Status |
|---|---|---|---|
| Phase 1 | Tool Calling | BFCL v3 | ✅ Done |
| Phase 2 | Instruction Following | Google IFEval (541 prompts) | 🔄 Running |
| Phase 3 | Code Generation | OpenAI HumanEval (164 problems) | ⏳ Planned |

Phase 2 is where Granite 4.1 should shine brightest — IBM published an **87.06% IFEval score** (instruction following accuracy), which would put it well above the 42% we saw on tool calling. Testing whether that holds up locally is the next experiment.

---

## The Code

Everything — the dataset loader, the inference runner, the AST evaluator, the metrics pipeline, and the notebook — is open and replicable. If you have Ollama installed on a Mac with 16GB+ RAM, you can run this yourself.

**Requirements:**
- Mac M3/M2/M1 with 16GB+ RAM (36GB recommended for all 5 models)
- [Ollama](https://ollama.com) installed
- Python 3.11

```bash
git clone <repo>
cd granite4_1
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pull models (one time, ~20GB total)
ollama pull granite4.1:8b
ollama pull granite4:latest
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b

# Run Phase 1
jupyter lab phase1_tool_calling.ipynb
```

---

## Closing Thought

IBM's Granite 4.1 is a genuinely good model. It's not a marketing story — the improvement over Granite 4.0 is real, the competitive positioning against Llama and Qwen is fair, and the Apache 2.0 license means you can use it in production without legal headaches.

But the most interesting number from this experiment isn't any model's accuracy. It's the **Parallel category** — where the best model in the test scored only 30%.

When AI agents need to do multiple things at once — the most natural thing for an orchestration system to ask — every model tested struggles. That's the real frontier. Not "can an 8B model match a 32B model" but "can any model reliably coordinate complex multi-step tool use without falling apart."

Phase 2 will tell us if instruction following paints a different picture. Stay tuned.

---

*All code, data, and charts from this experiment are available in the project repository. Benchmark run: 5,000 inference calls, ~8 hours, MacBook M3 36GB, May 2026.*

---

**Tags:** `IBM` `Granite` `LLM` `Tool Calling` `AI Agents` `Open Source` `Ollama` `Benchmark` `Machine Learning`
