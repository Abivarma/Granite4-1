# I Ran IBM's Granite 4.1 Against 4 LLMs on My MacBook Again — This Time on Instruction Following

**Phase 2: Does the model that follows complex instructions best also call tools best? The answer will surprise you.**

---

*Everything in this post was run locally on a MacBook M3 (36GB RAM) using Ollama. No cloud APIs. No sponsored benchmarks. Just real code and real results. This is Part 2 of a 3-part series.*

---

## If You Missed Part 1

In [Part 1 of this series](https://github.com/Abivarma/Granite4-1/blob/main/blog/phase1_blog.md), I ran five 7–8B models — IBM Granite 4.1 8B, Granite 4.0, Llama 3.1 8B, Qwen 2.5 7B, and Mistral 7B — through 1,000 tool-calling prompts from the Berkeley Function Calling Leaderboard (BFCL v3).

The headline result: **Granite 4.1 and Qwen 2.5 tied at the top**, separated by just 0.6 percentage points (42.2% vs 42.8%). Mistral had a quiet meltdown on parallel tool calls — timing out on 59 out of 200 prompts.

But tool calling is only one dimension of what makes a model useful. A model that can call functions correctly is useless if it ignores your explicit instructions — *"answer in bullet points"*, *"don't use commas"*, *"write at least 300 words"*. That's a completely different capability, and it's what Phase 2 tests.

The question: does the Phase 1 ranking hold? Or does a different model take the crown when the task changes?

---

## What Is Instruction Following? (And Why Does It Matter)

Instruction following sounds obvious. You tell the model something. The model does it.

But language models don't "do" things. They predict the next token based on everything they've seen. Whether they consistently honour explicit formatting constraints — letter frequency, word counts, capitalisation rules, ending phrases — depends entirely on how they were fine-tuned.

This is different from tool calling in a subtle but important way:

- **Tool calling** tests whether the model can *reason* about function signatures and produce structured output
- **Instruction following** tests whether the model can *constrain its own generation* throughout an entire response

The second one is harder to fake. A model can get lucky on a tool call by generating plausible-looking JSON. But counting whether the word "sneaker" appears 10 or more times in a 300-word response in all caps — that requires sustained attention across every token generated.

This is the backbone of prompt engineering. Every system prompt, every output format requirement, every persona constraint — they all depend on instruction following. If a model fails here, no amount of clever prompting fixes it.

---

## The Benchmark: Google IFEval

For this phase I used **IFEval** (Instruction Following Evaluation), a benchmark developed by Google Research and published at NeurIPS 2023. It's the standard dataset for measuring verifiable instruction compliance.

IFEval is elegant in design: every prompt contains 1–3 explicit instructions that can be checked programmatically — no human judgment needed.

| Category | What's Being Tested | Example Instruction |
|---|---|---|
| **language** | Language detection | "Your entire response must be in French" |
| **detectable_format** | Formatting structure | "Include exactly 3 highlighted sections using markdown" |
| **detectable_content** | Content presence | "End your response with a postscript starting with P.S." |
| **startend** | Response framing | "Start your response with the word 'Certainly'" |
| **punctuation** | Punctuation rules | "Do not use any commas in your response" |
| **change_case** | Case formatting | "Write your entire response in all capital letters" |
| **keywords** | Keyword constraints | "Use the word 'algorithm' at least 5 times" |
| **length_constraints** | Length targets | "Write at least 300 words" |
| **combination** | Multiple constraints | Any combination of the above |

**541 prompts total.** Each prompt is scored at two levels:
- **Prompt accuracy**: did the model satisfy *all* instructions in the prompt?
- **Instruction accuracy**: of all individual instructions across all prompts, what fraction were satisfied?

Instruction accuracy is the more forgiving metric — a prompt with 2 instructions where 1 fails still contributes 1 passing instruction to the count.

---

## The Setup

Same hardware, same models, same approach as Phase 1.

**Hardware:** MacBook M3, 36GB unified memory  
**Platform:** Ollama (local inference — no cloud)  
**Models tested:**

| Model | Size | Role |
|---|---|---|
| Granite 4.1 8B | 5.3 GB | IBM's latest — the subject |
| Granite 4.0 | 2.1 GB | IBM's previous version — direct comparison |
| Llama 3.1 8B | 4.9 GB | Meta's flagship 8B — industry baseline |
| Qwen 2.5 7B | 4.7 GB | Alibaba's model — Phase 1 leader |
| Mistral 7B | 4.4 GB | Classic open-source baseline |

**541 prompts × 5 models = 2,705 total inference calls.** All results cached to disk as JSON.

One important note: I added a **1,024-token output cap** (`num_predict: 1024`) and a **120-second per-call timeout** after one prompt — "write the word sneaker 10+ times in ALL CAPS" — caused a model to generate indefinitely for 31 minutes. The cap was necessary to make the run tractable locally and almost certainly explains a small part of our gap with IBM's published scores.

---

## The Results

### Overall Rankings

| Model | Prompt Accuracy | Instruction Accuracy | Avg Tokens | Median Latency |
|---|---|---|---|---|
| 🥇 Granite 4.1 8B | **79.7%** | **85.3%** | 299 | 9,531ms |
| 🥈 Granite 4.0 | 77.3% | 83.7% | 277 | 4,009ms |
| 🥉 Llama 3.1 8B | 70.8% | 78.3% | 299 | 10,611ms |
| Qwen 2.5 7B | 70.1% | 78.4% | 271 | 8,022ms |
| Mistral 7B | 46.8% | 57.0% | 345 | 11,399ms |

> 📊 **[Insert ifeval_accuracy.png here]**
> *Prompt and Instruction accuracy for all 5 models. Red dashed line = IBM's published Granite 4.1 8B score of 87.06%.*

The picture is immediately cleaner than Phase 1. There's no statistical tie at the top — Granite 4.1 leads clearly. And the bottom is starker: Mistral 7B sits 24 points below the third-place model.

---

## Finding #1: The Phase 1 Leader Just Got Dethroned

The most striking result is what happened to Qwen 2.5 7B.

In Phase 1, Qwen *edged out* Granite 4.1 on tool calling — 42.8% vs 42.2%, the narrowest margin in the test. It was reasonable to ask: is Qwen actually the best 7–8B model overall?

Phase 2 answers that question:

| Model | Phase 1 Tool Calling | Phase 2 Instruction Following | Rank Change |
|---|---|---|---|
| Granite 4.1 8B | 42.2% | **79.7%** | Stays 1st |
| Granite 4.0 | 34.8% | 77.3% | +2 positions |
| Llama 3.1 8B | 40.4% | 70.8% | -1 position |
| Qwen 2.5 7B | **42.8%** | 70.1% | -2 positions |
| Mistral 7B | 34.9% | 46.8% | Stays 5th |

**Qwen 2.5 drops two places.** It led on tool calling but falls to 4th on instruction following — 9.6 percentage points behind Granite 4.1 and nearly tied with Llama 3.1.

This reveals something important: **tool calling ability and instruction following ability are not the same skill.** A model can be well-optimised for JSON-structure generation (tool calling) without being fine-tuned to honour formatting constraints throughout an open-ended response (instruction following). Qwen seems to fall into this camp.

---

## Finding #2: Granite 4.0 Is a Dark Horse

Here's the result I didn't expect: **Granite 4.0 finishes second**.

In Phase 1, Granite 4.0 was near the bottom — 34.8%, trailing Llama, Qwen, and Granite 4.1 by meaningful margins. It looked like a model that had been clearly superseded.

Phase 2 tells a different story:

| | Phase 1 | Phase 2 | Delta |
|---|---|---|---|
| Granite 4.0 | 34.8% | 77.3% | **+42.5pp** |
| Granite 4.1 8B | 42.2% | 79.7% | +37.5pp |
| Qwen 2.5 7B | 42.8% | 70.1% | +27.3pp |

Granite 4.0's jump from Phase 1 to Phase 2 is the largest of any model. It closes most of the gap with Granite 4.1, and crucially, it does this while generating the fewest tokens (277 average) and running at the fastest latency (4,009ms median — more than 2x faster than Granite 4.1).

The interpretation: IBM's training pipeline — even in the 4.0 generation — was clearly oriented around instruction compliance. What changed in 4.1 was tool calling and agentic capability. The fundamental "do what you're told" behaviour was already well-baked.

**For production use cases that don't need tool calling, Granite 4.0 is a compelling choice**: nearly as accurate as 4.1, twice as fast, and about a third of the model size.

---

## Finding #3: The Category Breakdown Reveals Strengths and Landmines

Overall accuracy hides what's actually happening. Let me break it down by instruction category.

| Category | Granite 4.1 | Granite 4.0 | Llama 3.1 | Qwen 2.5 | Mistral |
|---|---|---|---|---|---|
| language | **100.0%** | 93.5% | 90.3% | **100.0%** | 74.2% |
| detectable_content | **96.2%** | 94.3% | 82.4% | 90.6% | 83.0% |
| detectable_format | **96.1%** | 94.9% | 82.1% | 87.2% | 75.8% |
| startend | 95.5% | **97.0%** | 90.9% | 88.1% | 69.7% |
| punctuation | 90.8% | 92.4% | 90.9% | **93.9%** | **9.1%** |
| change_case | 86.2% | **88.8%** | 77.3% | 73.0% | 53.4% |
| keywords | **84.5%** | 74.8% | 77.9% | 75.8% | 69.3% |
| length_constraints | **73.0%** | 71.3% | 68.8% | 62.9% | 45.1% |
| combination | 70.3% | 60.0% | 66.2% | **70.8%** | 18.5% |

A few things jump out immediately:

**Granite 4.1 is perfect on language detection.** 100% accuracy — it never responds in the wrong language. This matters for multilingual enterprise applications where language compliance is non-negotiable.

**Mistral's punctuation score is 9.1%.** Not a typo. When asked to avoid commas, Mistral uses commas almost every time. The model appears to have no reliable mechanism for suppressing specific punctuation characters. Given that "don't use commas" is one of the simpler constraints in the dataset, this is a fundamental gap.

**Combination instructions are where everyone bleeds.** Prompts that stack 2–3 constraints simultaneously are the hardest — the model has to maintain multiple constraints across the entire response simultaneously. Granite 4.1 and Qwen 2.5 are tied here at ~70%, but Mistral collapses to 18.5%.

**Length constraints are the universal weak point.** No model exceeds 73% here. "Write at least 300 words" or "respond in fewer than 100 words" sounds easy but requires the model to track and regulate its own output length — something that fights against how autoregressive generation works.

---

## Finding #4: How Close Are We to IBM's Published Score?

IBM published a Granite 4.1 8B IFEval score of **87.06%** (instruction-level accuracy). Our measurement gives **85.3%**.

That's a gap of only **1.76 percentage points** — dramatically closer than the 26-point gap we saw in Phase 1.

The reason is straightforward: IFEval is a public, fixed dataset with no ambiguity about which prompts to run. The BFCL gap in Phase 1 was driven by dataset distribution differences. Here, we're running the same 541 prompts.

The remaining 1.76-point gap almost certainly comes from the **1,024-token cap** we added. Some IFEval prompts require long responses to satisfy length constraints — a model that gets cut off mid-response fails those instructions. IBM ran without a token cap. We couldn't, for practical reasons.

**Bottom line: our local measurement is a genuine reproduction of IBM's result, within noise.** This is a much stronger validation than Phase 1 could offer.

---

## Finding #5: What Mistral's Numbers Are Actually Telling You

Mistral 7B's 46.8% prompt accuracy is low. But the category breakdown tells a more precise story than the headline number.

| Category | Mistral Score | Gap vs Granite 4.1 |
|---|---|---|
| punctuation | 9.1% | **-81.7pp** |
| combination | 18.5% | **-51.8pp** |
| startend | 69.7% | -25.8pp |
| change_case | 53.4% | -32.8pp |
| length_constraints | 45.1% | -27.9pp |

The 9.1% on punctuation is the most important number. This isn't a quantization artifact or a benchmarking edge case. Commas are fundamental — they appear in almost every English sentence. A model that cannot reliably suppress commas when asked has a deep alignment failure on constraint following.

And combination prompts stack constraints. If punctuation compliance is broken and combination compliance requires punctuation compliance, the 18.5% on combination follows logically.

**Mistral 7B isn't a bad model for generation quality. But it should not be your choice for any application that requires users to specify output formatting constraints.**

---

## Finding #6: The Token Efficiency Reversal

In Phase 1, I noted that IBM's token efficiency claim didn't clearly show up on tool calling. Phase 2 paints a slightly different picture.

| Model | Avg Tokens | Prompt Accuracy | Tokens per Correct Prompt |
|---|---|---|---|
| Granite 4.0 | **277** | 77.3% | ~358 |
| Qwen 2.5 7B | 271 | 70.1% | ~387 |
| Granite 4.1 8B | 299 | 79.7% | ~375 |
| Llama 3.1 8B | 299 | 70.8% | ~422 |
| Mistral 7B | 345 | 46.8% | **~737** |

Mistral generates the most tokens and gets the worst results. It's outputting verbose responses that still don't satisfy the constraints. That's the worst of both worlds — slow and wrong.

Granite 4.0 is the most token-efficient overall: fewest average tokens, second-best accuracy. For cost-sensitive deployments, this is significant.

---

## The Cross-Phase Picture: What Two Benchmarks Tell Us Together

Running two different capability tests on the same models reveals something that a single benchmark cannot: **model "rank" is not a fixed property.**

| Model | Tool Calling Rank | Instruction Following Rank |
|---|---|---|
| Qwen 2.5 7B | 1st | 4th |
| Granite 4.1 8B | 2nd | 1st |
| Llama 3.1 8B | 3rd | 3rd |
| Mistral 7B | 4th | 5th |
| Granite 4.0 | 5th | 2nd |

The only model that's consistent across both phases is Llama 3.1 — firmly in the middle on both. Every other model has at least one significant positional shift.

This has a practical implication: **you cannot pick a model based on a single benchmark and trust that it generalises.** Qwen 2.5 looks like the top model if you only read Phase 1. Granite 4.0 looks unimpressive if you only read Phase 1. Two benchmarks flip both conclusions.

---

## What These Results Actually Mean

**If you're building a system that relies on structured output, formatting constraints, or explicit instructions:**

- Granite 4.1 8B is the safest choice — top of the rankings, close to IBM's published score, consistent across both capability domains
- Granite 4.0 is the right call if latency and model size matter more than absolute accuracy (2x faster, 60% smaller than 4.1)
- Llama 3.1 8B is a solid middle ground — 3rd in both phases, no surprises, well-understood
- Qwen 2.5 7B is better for tool calling than instruction following; choose it only if Phase 1 scenarios dominate your workload
- Avoid Mistral 7B for anything involving explicit output formatting — the punctuation collapse alone disqualifies it

**If you're evaluating IBM's claims about Granite 4.1:**

- ✅ "Best instruction following among 8B models" — supported; leads all 5 models tested
- ✅ "87.06% IFEval score" — reproduced locally at 85.3%, within 2 points (token cap explains the gap)
- ✅ "Consistent improvement over Granite 4.0" — confirmed, but 4.0 is closer than expected
- ✅ "Token efficient" — clearer on instruction following than tool calling; Granite 4.0 especially lean

---

## What's Next

This is Phase 2 of a 3-part experiment.

| Phase | Capability | Dataset | Status |
|---|---|---|---|
| Phase 1 | Tool Calling | BFCL v3 (1,000 prompts) | ✅ Done — [read here](https://github.com/Abivarma/Granite4-1/blob/main/blog/phase1_blog.md) |
| Phase 2 | Instruction Following | Google IFEval (541 prompts) | ✅ Done — you're reading it |
| Phase 3 | Code Generation | OpenAI HumanEval (164 problems) | ⏳ Planned |

Phase 3 is code generation on HumanEval. IBM hasn't published a strong headline number for this — which makes it interesting. The hypothesis going in: Granite 4.1 will be competitive on Python but may lose ground to Qwen 2.5, which has a reputation for code quality. We'll see.

---

## The Code

Everything is open and replicable. The evaluator uses the official IFEval instruction verifier classes — the same ones Google released with the paper.

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

# Run Phase 2
jupyter lab phase2_instruction_following.ipynb
```

---

## Closing Thought

Phase 1 ended with a genuine surprise — that the best tool-calling model is a near-tie at the top, and the real frontier is parallel multi-call reliability where the best model still fails 70% of the time.

Phase 2 ends with a different kind of surprise. We expected Granite 4.1 to lead here based on IBM's published score. It did. What we didn't expect was how much Granite 4.0 would outperform its Phase 1 position — or how dramatically Qwen 2.5 would fall back.

The deepest lesson is about Mistral. A 9.1% punctuation compliance score isn't a benchmark quirk. It's a signal about how the model was trained and what it was optimised for. Instruction following benchmarks have a way of exposing alignment gaps that generation quality scores paper over.

Phase 3 will tell us if Granite 4.1 can hold the lead when the task changes from following instructions to generating correct code. Tool calling, instruction following, and code generation are all "following specifications" in some sense — but they exercise completely different parts of the model.

Stay tuned.

---

*All code, data, and charts from this experiment are available in the project repository. Phase 2 run: 2,705 inference calls, ~3 hours, MacBook M3 36GB, May 2026.*

---

**Tags:** `IBM` `Granite` `LLM` `IFEval` `Instruction Following` `Open Source` `Ollama` `Benchmark` `Machine Learning` `NLP`
