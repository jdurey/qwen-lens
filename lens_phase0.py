#!/usr/bin/env python3
"""
lens_phase0.py — Phase-0 feasibility gate for qwen-lens.

Proves the two things everything else depends on:
  1. We can capture per-layer residual-stream states from the MLX
     Qwen3.6-35B-A3B during a forward pass.
  2. The logit lens gives sane readouts at 4-bit: on "The capital of
     France is", " Paris" should become top-1 in middle layers, well
     before the output layer.

Architecture notes (verified against mlx_lm/models/qwen3_5.py):
  - hybrid stack: GatedDeltaNet (linear-attn) layers, full attention
    every `full_attention_interval`-th layer. Residual stream is
    unchanged by this — h after each layer is still the workspace.
  - layout: model.language_model.model -> {embed_tokens, layers, norm}
  - per-layer call: layer(h, mask=ssm|fa, cache=c)
  - lens projection: inner.norm(h) -> language_model.lm_head

Usage:
  ~/venvs/mlx/bin/python3 lens_phase0.py [--model REPO] [--prompt "..."]
                                         [--expect " Paris"]
Exit 0 = gate PASSED.
"""

import argparse
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--expect", default=" Paris")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    print(f"loading {args.model} ...", flush=True)
    model, tokenizer = load(args.model)

    lm = getattr(model, "language_model", model)      # multimodal wrapper
    inner = lm.model                                   # Qwen3_5TextModel
    layers, norm, emb = inner.layers, inner.norm, inner.embed_tokens
    tie = getattr(lm, "lm_head", None) is None
    n_layers = len(layers)
    n_linear = sum(1 for l in layers if l.is_linear)
    print(f"stack: {n_layers} layers ({n_linear} linear-attn, "
          f"{n_layers - n_linear} full-attn), lm_head={'tied' if tie else 'separate'}")

    def unembed(h):
        h = norm(h)
        return emb.as_linear(h) if tie else lm.lm_head(h)

    ids = tokenizer.encode(args.prompt)
    x = mx.array([ids])
    h = emb(x)
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)

    expect_id = tokenizer.encode(args.expect)[-1]

    print(f"\nprompt: {args.prompt!r} ({len(ids)} tokens)  "
          f"expecting {args.expect!r} (id {expect_id}) to rise\n")
    print(f"{'layer':>5} {'type':>7} {'p(expect)':>10}  top-{args.topk}")

    first_top1 = None
    final_top_id = None
    for li, layer in enumerate(layers):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=None)
        logits = unembed(h[:, -1:, :])
        probs = mx.softmax(logits[0, -1].astype(mx.float32))
        mx.eval(probs)
        order = mx.argsort(-probs)[: args.topk]
        top_ids = [int(i) for i in order]
        top = [(tokenizer.decode([i]), round(float(probs[i]), 4)) for i in top_ids]
        p_expect = float(probs[expect_id])
        mark = ""
        if top_ids[0] == expect_id:
            if first_top1 is None:
                first_top1 = li
            mark = "  <-- top-1"
        kind = "linear" if layer.is_linear else "full"
        print(f"{li:>5} {kind:>7} {p_expect:>10.4f}  {top}{mark}")
        final_top_id = top_ids[0]

    ok_final = final_top_id == expect_id
    ok_early = first_top1 is not None and first_top1 <= n_layers - 3

    print(f"\nfinal-layer top-1 id {final_top_id} "
          f"({tokenizer.decode([final_top_id])!r}) vs expect {expect_id} "
          f"-> {'OK' if ok_final else 'FAIL'}")
    print(f"first top-1 at layer {first_top1} of {n_layers} -> "
          f"{'OK (poised before output)' if ok_early else 'WEAK/FAIL'}")

    if ok_final and ok_early:
        print("\nGATE PASSED: residual stream captured, lens readout sane at 4-bit.")
        return
    sys.exit("\nGATE FAILED: see readouts. Try 8-bit spot-check or dense-9B "
             "fallback per PLAN.md.")


if __name__ == "__main__":
    main()
