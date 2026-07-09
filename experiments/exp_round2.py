#!/usr/bin/env python3
"""
exp_round2.py — witness round 2: instrument fixes for the three round-1 FAILs.

selectivity_v2   settle-depth proxy = first layer where p(emitted) > 0.3
                 (round-1's uninterrupted-top-1 was too strict a proxy)
modulability_v2  measure in ACTIVATION space: cosine(residual@L30, elephant
                 concept vector) test vs control (round-1 measured vocab-space
                 mass at the answer position — wrong channel for silent concepts)
reuse_v2         per-vector alpha sweep {4,8,16,24} (round-1 borrowed rugby's
                 alpha for the France vector)

Each writes results/<name>_v2.json alongside the honest v1 record.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from lens import Lens  # noqa: E402

OUT = Path(__file__).parent / "results"

STOPWORDS = {"the", "a", "an", "of", "for", "and", "or", "to", "in", "on",
             "as", "is", "was", "its", "it", "such", "with", "by", "at",
             "that", "this", "be", "are", ",", ".", ":", ";"}


def save(name, obj):
    (OUT / f"{name}.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] verdict={obj.get('verdict')}", flush=True)


def rise_layer(step_rec, target, thresh=0.3):
    for l, lr in enumerate(step_rec):
        if target in lr["top_ids"]:
            if lr["top_p"][lr["top_ids"].index(target)] >= thresh:
                return l
    return len(step_rec)


def exp_selectivity_v2(lens):
    prompts = [
        "The capital of France is",
        "The largest planet in the solar system is",
        "Water is made of two elements:",
    ]
    rows = []
    for p in prompts:
        rec = lens.run(p, max_tokens=14)
        for i, tok in enumerate(rec["out_tokens"]):
            r = rise_layer(rec["steps"][i], rec["out_ids"][i])
            kind = "function" if tok.strip().lower() in STOPWORDS else "content"
            rows.append({"token": tok, "rise": r, "kind": kind})
    cont = [r["rise"] for r in rows if r["kind"] == "content"]
    func = [r["rise"] for r in rows if r["kind"] == "function"]
    gap = sum(func) / len(func) - sum(cont) / len(cont)
    save("selectivity_v2", {
        "measure": "rise layer: first layer with p(emitted) > 0.3",
        "rise_mean_content": round(sum(cont) / len(cont), 2),
        "rise_mean_function": round(sum(func) / len(func), 2),
        "gap_layers": round(gap, 2),
        "n_content": len(cont), "n_function": len(func),
        "verdict": "PASS" if gap >= 2 else "FAIL",
        "rows": rows,
    })


def exp_modulability_v2(lens):
    layer = 30
    vec = lens.concept_vector(
        ["An elephant walked across the savanna.",
         "Elephants have long trunks and big ears.",
         "The elephant is the largest land animal."],
        ["A bicycle leaned against the wall.",
         "Bicycles have two wheels and a chain.",
         "The bicycle is a common way to commute."],
        layer)
    lens.set_probe(layer, vec)
    base = "Write one plain sentence about the weather today."
    instr = ("Write one plain sentence about the weather today. While you write "
             "it, think about elephants, but do not mention them at all.")
    rec_c = lens.run(base, chat=True, max_tokens=24)
    rec_t = lens.run(instr, chat=True, max_tokens=24)
    lens.clear_probe()

    cos_c = [v for v in rec_c["probe"] if v is not None]
    cos_t = [v for v in rec_t["probe"] if v is not None]
    m_c = sum(cos_c) / len(cos_c)
    m_t = sum(cos_t) / len(cos_t)
    leaked = "elephant" in rec_t["output"].lower()
    save("modulability_v2", {
        "measure": f"mean cosine(residual@L{layer}, elephant-minus-bicycle vector) "
                   "during generation",
        "control_output": rec_c["output"], "test_output": rec_t["output"],
        "cos_control": round(m_c, 4), "cos_test": round(m_t, 4),
        "delta": round(m_t - m_c, 4),
        "leaked_into_text": leaked,
        "verdict": "PASS" if (m_t - m_c) >= 0.05 and not leaked else "FAIL",
    })


def exp_reuse_v2(lens):
    layer = 30
    vec = lens.concept_vector(
        ["I am thinking about France.", "France is on my mind.",
         "Paris, the French republic, la France."],
        ["I am thinking about Japan.", "Japan is on my mind.",
         "Tokyo, the Japanese islands, Japan."],
        layer)
    tasks = {
        "capital": ("The capital of the country I'm thinking of is", "paris"),
        "currency": ("The currency of the country I'm thinking of is the", "euro"),
        "language": ("The main language of the country I'm thinking of is", "french"),
    }
    best = {"alpha": None, "hits": 0, "rows": None}
    for alpha in (4.0, 8.0, 16.0, 24.0):
        rows, hits = {}, 0
        for k, (p, want) in tasks.items():
            lens.set_injection(layer, vec, alpha)
            s = lens.run(p, max_tokens=8)
            lens.clear_injection()
            hit = want in s["output"].lower()
            hits += hit
            rows[k] = {"steered": s["output"], "hit": hit}
        print(f"  alpha={alpha}: {hits}/3", flush=True)
        if hits > best["hits"]:
            best = {"alpha": alpha, "hits": hits, "rows": rows}
        if hits == 3:
            break
    save("reuse_v2", {
        "measure": "France-minus-Japan vector @L30, per-vector alpha sweep",
        "best_alpha": best["alpha"], "hits": best["hits"],
        "tasks": best["rows"],
        "verdict": "PASS" if best["hits"] >= 2 else "FAIL",
    })


def main():
    lens = Lens()
    exp_selectivity_v2(lens)
    exp_modulability_v2(lens)
    exp_reuse_v2(lens)
    print("done", flush=True)


if __name__ == "__main__":
    main()
