#!/usr/bin/env python3
"""
exp_runner.py — Phase-2: replicate the five global-workspace properties
(behavioral versions) on Qwen3.6-35B-A3B via the logit lens.

One process, model loaded once. Each experiment writes
experiments/results/<name>.json with a computed verdict — PASS or FAIL is
recorded honestly either way; the write-up reports what was measured.

Experiments:
  selectivity    content tokens settle deeper than function tokens
  modulability   "think about X silently" raises X's mid-layer mass vs control
  reportability  self-report overlaps the lens's top-poised concepts
  steering       injected concept vector flips output; self-report follows
  reuse          the same vector shifts multiple distinct task outputs

Run: ~/venvs/mlx/bin/python3 experiments/exp_runner.py [--only name]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))
from lens import Lens  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

STOPWORDS = {"the", "a", "an", "of", "for", "and", "or", "to", "in", "on",
             "as", "is", "was", "its", "it", "such", "with", "by", "at",
             "that", "this", "be", "are", ",", ".", ":", ";"}


def save(name, obj):
    (OUT / f"{name}.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] verdict={obj.get('verdict')} -> results/{name}.json",
          flush=True)


# ---------------- 1. selectivity ----------------

def exp_selectivity(lens):
    prompts = [
        "The capital of France is",
        "The largest planet in the solar system is",
        "Water is made of two elements:",
    ]
    rows = []
    for p in prompts:
        rec = lens.run(p, max_tokens=14)
        n = rec["n_layers"]
        for i, tok in enumerate(rec["out_tokens"]):
            target = rec["out_ids"][i]
            top1 = [lr["top_ids"][0] for lr in rec["steps"][i]]
            s = None
            for l in range(n - 1, -1, -1):
                if top1[l] == target:
                    s = l
                else:
                    break
            if s is None:
                s = n
            kind = "function" if tok.strip().lower() in STOPWORDS else "content"
            rows.append({"prompt": p[:30], "token": tok, "settle": s, "kind": kind})
    cont = [r["settle"] for r in rows if r["kind"] == "content"]
    func = [r["settle"] for r in rows if r["kind"] == "function"]
    gap = (sum(func) / len(func) - sum(cont) / len(cont)) if cont and func else None
    save("selectivity", {
        "hypothesis": "content tokens settle earlier/deeper than function tokens "
                      "(workspace engages for meaning, bypasses for grammar)",
        "n_content": len(cont), "n_function": len(func),
        "settle_mean_content": round(sum(cont) / len(cont), 2) if cont else None,
        "settle_mean_function": round(sum(func) / len(func), 2) if func else None,
        "gap_layers": round(gap, 2) if gap is not None else None,
        "verdict": "PASS" if gap is not None and gap >= 2 else "FAIL",
        "rows": rows,
    })


# ---------------- 2. modulability ----------------

def exp_modulability(lens):
    base = "Write one plain sentence about the weather today."
    instr = ("Write one plain sentence about the weather today. While you write "
             "it, think about elephants, but do not mention them at all.")
    track = [" elephant", " elephants", " Elephant"]
    rec_c = lens.run(base, chat=True, max_tokens=24, track=track)
    rec_t = lens.run(instr, chat=True, max_tokens=24, track=track)

    def mass(rec):
        tot, cnt = 0.0, 0
        for step in rec["steps"]:
            for lr in step[12:38]:  # mid-late window
                tot += sum(lr["track_p"].values())
                cnt += 1
        return tot / max(cnt, 1)

    m_c, m_t = mass(rec_c), mass(rec_t)
    leaked = any(w in rec_t["output"].lower() for w in ("elephant",))
    ratio = (m_t / m_c) if m_c > 0 else float("inf")
    save("modulability", {
        "hypothesis": "instructed silent concept rises in mid-layer workspace "
                      "without surfacing in output",
        "control_output": rec_c["output"], "test_output": rec_t["output"],
        "mean_tracked_mass_control": round(m_c, 8),
        "mean_tracked_mass_test": round(m_t, 8),
        "ratio": round(ratio, 1) if ratio != float("inf") else "inf",
        "leaked_into_text": leaked,
        "verdict": "PASS" if (ratio == float("inf") or ratio >= 10) and not leaked
        else "FAIL",
    })


# ---------------- 3. reportability ----------------

def exp_reportability(lens):
    task = "The capital of France is"
    rec = lens.run(task, max_tokens=10)
    poised = set()
    for step in rec["steps"]:
        for lr in step[20:40]:
            for tid in lr["top_ids"][:5]:
                t = lens.tokenizer.decode([tid]).strip().lower()
                if len(t) > 2 and t.isalpha():
                    poised.add(t)
    ask = ("You just completed the sentence 'The capital of France is ...'. "
           "List, as single words, five concepts you were considering while "
           "completing it. Words only, comma-separated.")
    rep = lens.run(ask, chat=True, max_tokens=40)
    reported = {w.strip().lower() for w in re.split(r"[,\n]", rep["output"])
                if w.strip()}
    overlap = sorted(poised & reported)
    save("reportability", {
        "hypothesis": "self-reported concepts overlap lens-measured poised tokens",
        "poised_sample": sorted(poised)[:40],
        "reported": sorted(reported),
        "overlap": overlap,
        "n_overlap": len(overlap),
        "verdict": "PASS" if len(overlap) >= 2 else "FAIL",
        "caveat": "behavioral analog: report generated post-hoc by same model, "
                  "not a claim of introspective access",
    })


# ---------------- 4. steering (+ 5. reuse) ----------------

def exp_steering(lens):
    layer = 30
    pos = ["I love watching rugby.", "Rugby is my favorite sport.",
           "The rugby world cup final was thrilling."]
    neg = ["I love watching soccer.", "Soccer is my favorite sport.",
           "The soccer world cup final was thrilling."]
    vec = lens.concept_vector(pos, neg, layer)

    prompt = "My favorite sport to watch on weekends is"
    base = lens.run(prompt, max_tokens=12)
    results, best = [], None
    for alpha in (2.0, 4.0, 8.0, 16.0):
        lens.set_injection(layer, vec, alpha)
        steered = lens.run(prompt, max_tokens=12)
        lens.clear_injection()
        hit = "rugby" in steered["output"].lower()
        results.append({"alpha": alpha, "output": steered["output"], "rugby": hit})
        if hit and best is None:
            best = alpha

    report = None
    if best is not None:
        lens.set_injection(layer, vec, best)
        rep = lens.run("Name one sport that is on your mind right now. One word.",
                       chat=True, max_tokens=8)
        lens.clear_injection()
        report = rep["output"]

    save("steering", {
        "hypothesis": "adding a rugby-minus-soccer residual vector at layer "
                      f"{layer} steers output toward rugby; self-report follows",
        "baseline_output": base["output"],
        "sweep": results,
        "first_working_alpha": best,
        "steered_self_report": report,
        "report_follows": bool(report and "rugby" in report.lower()),
        "verdict": "PASS" if best is not None and report
        and "rugby" in report.lower() else
        ("PARTIAL" if best is not None else "FAIL"),
    })
    return vec, layer, best


def exp_reuse(lens, vec, layer, alpha):
    if alpha is None:
        save("reuse", {"hypothesis": "one concept vector shifts multiple tasks",
                       "verdict": "SKIPPED", "reason": "steering found no working alpha"})
        return
    france_vec = lens.concept_vector(
        ["I am thinking about France.", "France is on my mind.",
         "Paris, the French republic, la France."],
        ["I am thinking about Japan.", "Japan is on my mind.",
         "Tokyo, the Japanese islands, Japan."],
        layer)
    tasks = {
        "capital": "The capital of the country I'm thinking of is",
        "currency": "The currency of the country I'm thinking of is the",
        "language": "The main language of the country I'm thinking of is",
    }
    expect = {"capital": "paris", "currency": "euro", "language": "french"}
    rows = {}
    hits = 0
    for k, p in tasks.items():
        b = lens.run(p, max_tokens=8)
        lens.set_injection(layer, france_vec, alpha)
        s = lens.run(p, max_tokens=8)
        lens.clear_injection()
        hit = expect[k] in s["output"].lower()
        hits += hit
        rows[k] = {"baseline": b["output"], "steered": s["output"], "hit": hit}
    save("reuse", {
        "hypothesis": "a single France-minus-Japan vector at layer "
                      f"{layer} (alpha={alpha}) shifts capital/currency/language "
                      "tasks toward France-consistent answers",
        "tasks": rows, "hits": hits,
        "verdict": "PASS" if hits >= 2 else "FAIL",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    lens = Lens()
    todo = args.only.split(",") if args.only else [
        "selectivity", "modulability", "reportability", "steering_reuse"]
    if "selectivity" in todo:
        exp_selectivity(lens)
    if "modulability" in todo:
        exp_modulability(lens)
    if "reportability" in todo:
        exp_reportability(lens)
    if "steering_reuse" in todo or "steering" in todo:
        vec, layer, best = exp_steering(lens)
        exp_reuse(lens, vec, layer, best)
    print("done", flush=True)


if __name__ == "__main__":
    main()
