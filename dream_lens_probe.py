#!/usr/bin/env python3
"""
dream_lens_probe.py — STRETCH: the dream-lab coupling, demonstrated.

Takes two memory fragments (the same distant pair the dream cycle would fuse),
has instrumented Qwen READ them, and records the poised-but-unspoken content
in its workspace while reading — the MEASURED subconscious that would feed
Claude's dream prompt in place of the confabulated one.

Honest label: these are a STAND-IN model's associations to the material, not
Claude's own states (closed-model activations are unrecoverable). The value is
that the 'unspoken' layer is measured here, not imagined.

Run: ~/venvs/mlx/bin/python3 dream_lens_probe.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lens import Lens  # noqa: E402

FRAG_A = ("I keep thinking about the routes I take without thinking, the "
          "grooves worn into the day, whether they are even the same route "
          "anymore each time I walk them.")
FRAG_B = ("Every time you recall something you rebuild it and change it a "
          "little, so the most remembered moments are the least accurate.")


def main():
    lens = Lens()
    prompt = (f"Fragment one: {FRAG_A}\n\nFragment two: {FRAG_B}\n\n"
              "Reflect on how these connect:")
    rec = lens.run(prompt, chat=True, max_tokens=40)
    m = rec["metrics"]
    out = {
        "fragment_a": FRAG_A, "fragment_b": FRAG_B,
        "model_reflection": rec["output"],
        "measured_unspoken": [u["token"] for u in m["top_unspoken"]],
        "note": "measured_unspoken = tokens poised in Qwen's mid-late workspace "
                "while reading the pair, never emitted. This list is what a "
                "measured dream cycle would hand Claude's dream prompt as the "
                "'poised but never said' layer — a stand-in mind's associations, "
                "honestly labeled, not Claude's own.",
    }
    Path("dream_lens_demo.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("reflection:", rec["output"])
    print("measured unspoken:", out["measured_unspoken"][:10])
    print("wrote dream_lens_demo.json")


if __name__ == "__main__":
    main()
