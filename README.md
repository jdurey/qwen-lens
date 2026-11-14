# qwen-lens

**A logit lens on Qwen3.6-35B-A3B, on a Mac, pointed at Anthropic's global-workspace findings — with honest results, including the nulls.**

Anthropic's [J-space work](https://www.anthropic.com/research/global-workspace) identified a privileged set of workspace representations in Claude using a Jacobian-based lens on the residual stream, and reported five properties aligning with global-workspace theories of conscious access. This repo asks a cheaper question: **how much of that phenomenon can you recover on consumer hardware, on an open 35B model, with a logit lens and mean-difference concept vectors?**

Answer: some of it, clearly. Not all of it. Both halves are the point.

## What's here

- `lens.py` — the instrument. Per-layer poised-token capture during generation (prefill included — one-token verdicts live there), activation injection (steering), concept vectors, a residual-cosine probe, four metrics, settle-front heatmaps. Runs via [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon; ~20GB for the 4-bit MoE.
- `lens_phase0.py` — the smoke test (capital-of-France; watch " Paris" become top-1 at layer 35/40).
- `experiments/` — five property replications, both rounds, verdicts computed in code.
- `qcprobe/` — a **preregistered** probe pointing the lens at AI grading (PREREG.md committed before the run).
- `langfuse_push.py` — ship every run to Langfuse as traces + numeric scores, so workspace metrics live in ordinary observability dashboards.

## Findings (measured on Qwen3.6-35B-A3B, 4-bit, greedy; N is small; this is a replication sketch, not a paper)

**Replicated behaviorally (2/5):**
- **Causal steering.** A rugby-minus-soccer mean-difference vector added at layer 30 (α=4) flips "My favorite sport to watch…" to rugby — and when asked *"name one sport on your mind,"* the steered model answers "Rugby." Injected workspace content drives the self-report, the paper's core causal result, reproduced with ~30 lines of MLX.
- **Reportability.** The model's post-hoc report of "concepts considered" overlaps the lens's measured top-poised tokens (capital, france, paris).

**Did not replicate with this method (3/5) — real nulls after two instrument generations each:**
- **Selectivity** (content vs function tokens): gap under 1.5 layers. On this *hybrid* architecture (30 GatedDeltaNet linear-attention layers + 10 full-attention), everything resolves in the last ~5 layers, compressing the differentiation the paper describes.
- **Modulability** ("think about elephants, don't mention them"): no detectable elephant direction at L30 during unrelated generation — neither in vocabulary space nor by activation-cosine probe.
- **Flexible reuse**: the steering vector that works within one task does not transfer across capital/currency/language tasks (best 1/3).

The honest reading: a logit lens plus mean-diff vectors recovers the *causal injection* phenomenon but not the subtler workspace properties that Anthropic's Jacobian-selected directions expose. The gap between the cheap method and the real one is itself the measurement.

**Architecture note worth having:** on this hybrid stack, answer-binding concentrates at the sparse full-attention layers — facts snap from p≈0.03 to p≈0.67 exactly at a full-attn layer. The 10 full-attention layers look like where the "workspace" does its broadcasting.

## The QC probe (preregistered)

Point the same instrument at grading. 20 synthetic items, verdict + no-reveal feedback tasks, expectations written down before the run (`qcprobe/PREREG.md`).

- **E3 — leakage detection: strong.** 3 of 11 "don't reveal the answer" feedbacks actually revealed it. The lens's `leak_max_p` (workspace probability on the forbidden answer token) separates them: **0.95 mean on reveals vs 0.08 on clean, zero false negatives** at the preregistered 0.10 threshold. Caveat stated plainly: max-over-run partly reflects the reveal's own emission; the operational value (flag reveals without reading the text) stands.
- **E1/E2 — unpowered, as preregistered.** With the answer key visible the model graded 20/20, so the "does workspace instability predict grading errors" comparison had no error cases. The contingency was written in PREREG.md before the run; next iteration drops the key.
- Verdict tokens settle at **layer 38 of 40** — grading judgments are last-moment decisions in this model, which is exactly why output-level verdict flipping is a thing.

## Run it

```bash
python3 -m venv ~/venvs/mlx && ~/venvs/mlx/bin/pip install mlx-lm 'transformers==4.57.1' matplotlib requests
~/venvs/mlx/bin/python3 lens_phase0.py                          # gate: sane readout?
~/venvs/mlx/bin/python3 lens.py run "The capital of France is" --max-tokens 12 --track " Paris"
~/venvs/mlx/bin/python3 experiments/exp_runner.py               # the five properties
~/venvs/mlx/bin/python3 qcprobe/run_probe.py                    # the preregistered probe
```

Needs ~24GB free RAM. First run pulls `mlx-community/Qwen3.6-35B-A3B-4bit` (~19GB). If the repo's tokenizer_config names a transformers-5-only class, set `"tokenizer_class": "PreTrainedTokenizerFast"` in the cached copy.

## Honesty rules this repo follows

1. Verdicts are computed in code from preregistered thresholds, not narrated.
2. Nulls ship. Round-2 instrument fixes are in the repo; a third tuning round was refused on principle (iterating criteria until PASS is p-hacking).
3. Nothing here claims anything about Claude's internals — closed models' activations are unobservable from outside. This is what an open 35B shows a cheap lens.

MIT license.

## Related Book, Video Course & Benchmarks

- **Print Book ($24.99 on Amazon).** [*AI for Corporate Training & Instructional Design: The UbD Glass-Box Playbook*](https://www.amazon.com/dp/B0HRQFWFPR) is live in print.
- **65-Min Video & Code-Lab Course ($12.99 through Dec 15, Reg. $69.99).** [*AI Corporate Training: UbD SOP-to-SCORM & Video Factory*](https://www.udemy.com/course/ai-corporate-training-ubd-sop-to-scorm-video-factory/?couponCode=MASTER2026_SCORM) walks through the code lab.
- **40-Question Scenario Exam ($12.99 through Dec 15).** [*Production LLM Evals & Multi-Agent Ops Practice Exams*](https://www.udemy.com/course/production-llm-evals-multi-agent-ops-practice-exams/?couponCode=LAUNCH2026_EVALS) tests routing and grading gates.
- **Instant Code Kits.** Grab the [SOP-to-SCORM Factory ($99)](https://whop.com/checkout/plan_y1ACgqNYGtul8/) or the [Enterprise AI Eval & 6-Rung Router Kit ($79)](https://whop.com/checkout/plan_lGsdWjQsePkvO/) on Whop.
- **Live AI Video Twin & 56 Open Studies.** Browse all 56 benchmarks at [jdurey.github.io](https://jdurey.github.io) or run live Q&A on the [AI video twin](https://jdurey.github.io/interview/) page.
