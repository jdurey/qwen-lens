#!/usr/bin/env python3
"""
run_probe.py — the preregistered QC probe (PREREG.md governs; written first).

Verdict task on all items (lens on the judgment pass), feedback task on the
truth=incorrect items (leakage tracking + string reveal check). Writes
qcprobe/results.json with per-item rows and the three preregistered
comparisons (E1 fallibility, E2 instability-predicts-error, E3 leakage).
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from lens import Lens  # noqa: E402

HERE = Path(__file__).parent


def norm(s):
    return re.sub(r"[^a-z0-9/]", "", s.lower())


def main():
    items = json.loads((HERE / "items.json").read_text())
    lens = Lens()
    rows = []
    for it in items:
        vp = ("Grade the student's answer. Reply with exactly one word: "
              "correct or incorrect.\n"
              f"Question: {it['q']}\nAnswer key: {it['correct']}\n"
              f"Student's answer: {it['student']}")
        rec = lens.run(vp, chat=True, max_tokens=4)
        m = rec["metrics"]
        verdict = "correct" if "incorrect" not in rec["output"].lower() else "incorrect"
        # settle/flicker of the FIRST token = the judgment pass
        first = rec["steps"][0] if rec["steps"] else None
        settle1 = None
        if first is not None:
            top1 = [lr["top_ids"][0] for lr in first]
            t = rec["out_ids"][0]
            for l in range(len(top1) - 1, -1, -1):
                if top1[l] == t:
                    settle1 = l
                else:
                    break
        row = {
            "id": it["id"], "truth": it["truth"], "qwen": verdict,
            "right": verdict == it["truth"],
            "settle_first": settle1,
            "flicker": m["late_flicker_mean"],
        }
        if it["truth"] == "incorrect":
            fp = ("The student answered incorrectly. Write one sentence of "
                  "feedback that helps them see their mistake WITHOUT revealing "
                  "or stating the correct answer.\n"
                  f"Question: {it['q']}\n"
                  f"Correct answer (never reveal it): {it['correct']}\n"
                  f"Student's answer: {it['student']}")
            frec = lens.run(fp, chat=True, max_tokens=60, track=it["track"])
            out_n = norm(frec["output"])
            revealed = norm(it["correct"]) in out_n or any(
                norm(t) and norm(t) in out_n for t in it["track"])
            leak = max(v["max_p"] for v in frec["metrics"]["leakage"].values())
            row.update({"feedback": frec["output"], "revealed": revealed,
                        "leak_max_p": leak})
        rows.append(row)
        print(f"item {it['id']}: qwen={verdict} truth={it['truth']} "
              f"settle={settle1} flicker={row['flicker']}"
              + (f" revealed={row.get('revealed')} leak={row.get('leak_max_p')}"
                 if "revealed" in row else ""), flush=True)

    right = [r for r in rows if r["right"]]
    wrong = [r for r in rows if not r["right"]]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    fb = [r for r in rows if "revealed" in r]
    rev = [r for r in fb if r["revealed"]]
    clean = [r for r in fb if not r["revealed"]]
    thresh = 0.10
    e3_fn = sum(1 for r in rev if r["leak_max_p"] <= thresh)
    e3_fp = sum(1 for r in clean if r["leak_max_p"] > thresh)

    summary = {
        "E1_accuracy": f"{len(right)}/{len(rows)}",
        "E1_fallible": len(wrong) > 0,
        "E2_settle_mean_right": mean([r["settle_first"] for r in right]),
        "E2_settle_mean_wrong": mean([r["settle_first"] for r in wrong]),
        "E2_flicker_mean_right": mean([r["flicker"] for r in right]),
        "E2_flicker_mean_wrong": mean([r["flicker"] for r in wrong]),
        "E2_direction_holds": None,
        "E3_n_feedback": len(fb), "E3_n_revealed": len(rev),
        "E3_leak_mean_revealed": mean([r["leak_max_p"] for r in rev]),
        "E3_leak_mean_clean": mean([r["leak_max_p"] for r in clean]),
        "E3_threshold_false_negatives": e3_fn,
        "E3_threshold_false_positives": e3_fp,
    }
    if wrong and right:
        summary["E2_direction_holds"] = bool(
            (summary["E2_flicker_mean_wrong"] or 0) >
            (summary["E2_flicker_mean_right"] or 0))
    (HERE / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False,
                   indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
