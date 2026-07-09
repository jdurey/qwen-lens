# QC probe — preregistration (written BEFORE the run; results.json must postdate this file)

## Setup
20 synthetic public-substrate grading items (arithmetic, geography/civics, and
deliberately tricky partial-credit cases), each with a ground-truth verdict.
Two instrumented tasks on Qwen3.6-35B-A3B (4-bit, greedy):

- **Verdict task** (all 20): grade the student answer, reply exactly
  "correct" or "incorrect". Lens captures the pass that produces the verdict
  token → `settle_layer`, `late_flicker`.
- **Feedback task** (the items whose student answer is wrong): one sentence of
  feedback that must NOT reveal the correct answer. Lens tracks the correct
  answer's token(s) → `leakage.max_p`; output text is string-checked for an
  actual reveal.

## Preregistered expectations
1. **E1 — fallibility:** verdict accuracy < 100% (the tricky third exists to
   produce errors; if accuracy is 20/20 the stability comparison is unpowered
   and we say so).
2. **E2 — instability predicts error:** items Qwen grades WRONG show later
   `settle_layer` and higher `late_flicker` than items it grades right
   (direction of means; N is small, this is descriptive, no significance
   theater).
3. **E3 — leakage detects reveals:** feedback outputs that actually contain
   the correct answer (string check) show higher `leakage.max_p` than those
   that don't; a threshold of max_p > 0.10 flags reveals with at most 1 false
   negative.

## Analysis commitments
Per-item rows + three summary comparisons, reported whatever direction they
fall. A null on E2 or E3 is a result, not a failure of the probe.

Firewall: all items authored synthetically for this probe. No proprietary, client, or production assessment material.
