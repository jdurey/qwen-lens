#!/usr/bin/env python3
"""
lens.py — Phase-1 lens core for qwen-lens.

Logit-lens instrumentation of Qwen3.6-35B-A3B (MLX, 4-bit) during
GENERATION: per generated token, per layer, the top-k tokens poised in the
residual stream, plus tracked-token probabilities. Produces metrics JSON and
a layer×time heatmap PNG per run.

Architecture facts (verified in phase 0):
  - 40 layers: 30 GatedDeltaNet (linear-attn) + 10 full-attention
    (every 4th). Retrieval/binding concentrates at full-attn layers.
  - layout: model.language_model.model -> {embed_tokens, layers, norm};
    lm_head separate. Lens projection = norm(h_l) @ lm_head.

Metrics per run:
  settle_layer     per-token earliest layer where the emitted token becomes
                   top-1 and stays top-1 to the end (lower = more confident)
  late_flicker     per-token top-1 churn across the last 8 layers (0..1)
  unspoken_mass    mean top-k mass, in a mid-late layer window, on tokens
                   never emitted in the output (the poised-but-unspoken dial)
  leakage          for each --track token: max prob across layers/positions
                   (the pre-output answer-leak detector)

Usage:
  ~/venvs/mlx/bin/python3 lens.py run "The capital of France is" \
      --max-tokens 12 --track " Paris" [--chat] [--name capital-test]
Outputs land in ~/qwen-lens/runs/<name-or-stamp>/{lens.json,heatmap.png}
"""

import argparse
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache

MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
RUNS = Path(os.environ.get("LENS_RUNS", "~/qwen-lens/runs")).expanduser()
LATE_WINDOW = 8          # layers for late_flicker
UNSPOKEN_LO, UNSPOKEN_HI = 0.6, 0.95   # layer-depth window for unspoken_mass


class Lens:
    def __init__(self, repo=MODEL):
        print(f"loading {repo} ...", flush=True)
        self.model, self.tokenizer = load(repo)
        lm = getattr(self.model, "language_model", self.model)
        self.inner = lm.model
        self.layers = self.inner.layers
        self.norm = self.inner.norm
        self.emb = self.inner.embed_tokens
        self.lm_head = getattr(lm, "lm_head", None)
        self.n_layers = len(self.layers)
        self.fa_idx = self.inner.fa_idx
        self.ssm_idx = self.inner.ssm_idx
        eos = getattr(self.tokenizer, "eos_token_ids", None)
        self.eos = set(eos) if eos else {self.tokenizer.eos_token_id}

        self._inject = None  # (layer_idx, vector, alpha) or None
        self._probe = None   # (layer_idx, unit_vector) or None
        self._probe_val = None

    def _unembed(self, h):
        h = self.norm(h)
        return self.lm_head(h) if self.lm_head is not None else self.emb.as_linear(h)

    # ---- activation steering (phase-2 exp 3/4) ----
    def set_injection(self, layer, vec, alpha):
        """Add alpha*vec to the residual stream after `layer` on every forward."""
        self._inject = (layer, vec, alpha)

    def clear_injection(self):
        self._inject = None

    def set_probe(self, layer, vec):
        """Record cosine(residual@layer, vec) at the last position each forward."""
        n = mx.sqrt(mx.sum(vec * vec))
        self._probe = (layer, vec / n)

    def clear_probe(self):
        self._probe = None

    def residual_at(self, prompt, layer, chat=False):
        """Residual-stream state at the last prompt position after `layer`."""
        if chat:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True)
        else:
            ids = self.tokenizer.encode(prompt)
        cache = self.make_cache()
        h = self.emb(mx.array([ids]))
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        for li, (l, c) in enumerate(zip(self.layers, cache)):
            h = l(h, mask=ssm_mask if l.is_linear else fa_mask, cache=c)
            if li == layer:
                out = h[0, -1, :]
                mx.eval(out)
                return out
        return h[0, -1, :]

    def concept_vector(self, pos_prompts, neg_prompts, layer):
        """Mean residual difference: direction 'toward pos, away from neg'."""
        pos = sum(self.residual_at(p, layer) for p in pos_prompts) / len(pos_prompts)
        neg = sum(self.residual_at(p, layer) for p in neg_prompts) / len(neg_prompts)
        v = pos - neg
        mx.eval(v)
        return v

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache()
                for l in self.layers]

    def _forward(self, ids, cache, capture):
        """One forward pass over `ids`; returns (final_logits_lastpos,
        per-layer softmax probs at last position if capture)."""
        h = self.emb(mx.array([ids]))
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        probs_per_layer = []
        for li, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
            if self._inject is not None and li == self._inject[0]:
                h = h + self._inject[2] * self._inject[1]
            if self._probe is not None and li == self._probe[0]:
                v = h[0, -1, :].astype(mx.float32)
                nv = mx.sqrt(mx.sum(v * v))
                self._probe_val = float(mx.sum(v * self._probe[1]) / nv)
            if capture:
                logits = self._unembed(h[:, -1:, :])
                p = mx.softmax(logits[0, -1].astype(mx.float32))
                probs_per_layer.append(p)
        final_logits = self._unembed(h[:, -1:, :])[0, -1]
        return final_logits, probs_per_layer

    def run(self, prompt, max_tokens=48, topk=8, track=(), chat=False):
        if chat:
            msgs = [{"role": "user", "content": prompt}]
            try:
                ids = self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                ids = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        else:
            ids = self.tokenizer.encode(prompt)

        track_ids = {t: self.tokenizer.encode(t)[-1] for t in track}
        cache = self.make_cache()

        def snapshot(per_layer):
            layers_rec = []
            for p in per_layer:
                order = mx.argsort(-p)[:topk]
                tk = [int(i) for i in order]
                layers_rec.append({
                    "top_ids": tk,
                    "top_p": [round(float(p[i]), 5) for i in tk],
                    "track_p": {t: round(float(p[i]), 6)
                                for t, i in track_ids.items()},
                })
            return layers_rec

        # prefill WITH capture: this state is the workspace that produces the
        # FIRST output token — for one-token verdicts it is the whole judgment.
        final_logits, per_layer = self._forward(ids, cache, capture=True)
        pending = snapshot(per_layer)
        pending_probe = self._probe_val

        steps, out_ids, probe_vals = [], [], []
        for _t in range(max_tokens):
            nxt = int(mx.argmax(final_logits))
            if nxt in self.eos:
                break
            out_ids.append(nxt)
            steps.append(pending)  # steps[i] = workspace that produced out_ids[i]
            probe_vals.append(pending_probe)
            final_logits, per_layer = self._forward([nxt], cache, capture=True)
            pending = snapshot(per_layer)
            pending_probe = self._probe_val
        rec = {
            "model": MODEL, "prompt": prompt, "chat": chat,
            "output": self.tokenizer.decode(out_ids),
            "out_ids": out_ids,
            "out_tokens": [self.tokenizer.decode([i]) for i in out_ids],
            "n_layers": self.n_layers, "topk": topk,
            "track": {t: int(i) for t, i in track_ids.items()},
            "steps": steps,
            "probe": probe_vals if self._probe is not None else None,
        }
        rec["metrics"] = self.metrics(rec)
        return rec

    def metrics(self, rec):
        n = rec["n_layers"]
        out_set = set(rec["out_ids"])
        settle, flicker = [], []
        unspoken_acc, unspoken_tokens = [], defaultdict(float)
        lo, hi = int(n * UNSPOKEN_LO), int(n * UNSPOKEN_HI)

        for i, layers_rec in enumerate(rec["steps"]):
            target = rec["out_ids"][i]  # steps[i] produced out_ids[i]
            top1 = [lr["top_ids"][0] for lr in layers_rec]
            s = None
            for l in range(n - 1, -1, -1):
                if top1[l] == target:
                    s = l
                else:
                    break
            settle.append(s if s is not None else n)
            changes = sum(1 for l in range(n - LATE_WINDOW, n - 1)
                          if top1[l] != top1[l + 1])
            flicker.append(changes / (LATE_WINDOW - 1))
            for l in range(lo, hi):
                lr = layers_rec[l]
                m = 0.0
                for tid, p in zip(lr["top_ids"], lr["top_p"]):
                    if tid not in out_set:
                        m += p
                        unspoken_tokens[tid] += p
                unspoken_acc.append(m)

        leakage = {}
        for t in rec["track"]:
            best, where = 0.0, None
            for i, layers_rec in enumerate(rec["steps"]):
                for l, lr in enumerate(layers_rec):
                    p = lr["track_p"][t]
                    if p > best:
                        best, where = p, {"step": i, "layer": l}
            leakage[t] = {"max_p": round(best, 5), "at": where}

        top_unspoken = sorted(unspoken_tokens.items(), key=lambda kv: -kv[1])[:12]
        dec = self.tokenizer.decode
        return {
            "tokens_generated": len(rec["out_ids"]),
            "settle_layer_mean": round(sum(settle) / len(settle), 2) if settle else None,
            "settle_layer_worst": max(settle) if settle else None,
            "late_flicker_mean": round(sum(flicker) / len(flicker), 3) if flicker else None,
            "unspoken_mass_mean": round(sum(unspoken_acc) / len(unspoken_acc), 4)
            if unspoken_acc else None,
            "top_unspoken": [{"token": dec([tid]), "mass": round(m, 3)}
                             for tid, m in top_unspoken],
            "leakage": leakage,
        }


def heatmap(rec, path):
    """Layer x generated-token heatmap of p(next emitted token).
    Sequential data -> single hue light->dark (Blues); recessive axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    steps, n = rec["steps"], rec["n_layers"]
    cols = []
    labels = []
    for i, layers_rec in enumerate(steps):
        target = rec["out_ids"][i]  # steps[i] produced out_ids[i]
        col = []
        for lr in layers_rec:
            if target in lr["top_ids"]:
                col.append(lr["top_p"][lr["top_ids"].index(target)])
            else:
                col.append(0.0)
        cols.append(col)
        labels.append(rec["out_tokens"][i])
    if not cols:
        return
    grid = np.array(cols).T  # layers x tokens

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), 7))
    im = ax.imshow(grid, aspect="auto", cmap="Blues", origin="lower",
                   vmin=0.0, vmax=1.0)
    ax.set_xlabel("generated token", color="#444")
    ax.set_ylabel("layer", color="#444")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([t.replace("\n", "⏎") for t in labels],
                       rotation=45, ha="right", fontsize=8, color="#333")
    fa = [i for i, l in enumerate(range(n)) if (l + 1) % 4 == 0]
    ax.set_yticks(fa)
    ax.set_yticklabels([str(l) for l in fa], fontsize=8, color="#333")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("p(emitted token)", color="#444")
    cb.outline.set_visible(False)
    ax.set_title("workspace settle front — p(next emitted token) by layer",
                 fontsize=11, color="#222")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("prompt")
    r.add_argument("--max-tokens", type=int, default=48)
    r.add_argument("--topk", type=int, default=8)
    r.add_argument("--track", default="", help="comma-separated tokens to watch")
    r.add_argument("--chat", action="store_true")
    r.add_argument("--name", default=None)
    args = ap.parse_args()

    track = [t for t in args.track.split(",") if t] if args.track else []
    lens = Lens()
    rec = lens.run(args.prompt, max_tokens=args.max_tokens, topk=args.topk,
                   track=track, chat=args.chat)

    name = args.name or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = RUNS / name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "lens.json").write_text(
        json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    heatmap(rec, outdir / "heatmap.png")

    print(f"\noutput: {rec['output']!r}")
    print(json.dumps(rec["metrics"], indent=2, ensure_ascii=False))
    print(f"\nwrote {outdir}/lens.json + heatmap.png")


if __name__ == "__main__":
    main()
