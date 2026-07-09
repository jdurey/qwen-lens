#!/usr/bin/env python3
"""
langfuse_push.py — push qwen-lens runs to Langfuse (personal jp project) and
verify by API readback.

Uses the raw ingestion REST API (no SDK dependency): one trace per QC-probe
item with the four workspace metrics as numeric scores, plus one trace per
phase-2 experiment carrying its verdict. Reads creds from the standard
LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY env vars.

Run: ~/venvs/mlx/bin/python3 langfuse_push.py
Exit 0 only if ingestion succeeded AND readback finds the traces.
"""

import datetime as dt
import json
import sys
import uuid
from pathlib import Path

import requests

HERE = Path(__file__).parent


def creds():
    import os
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    pk = os.environ["LANGFUSE_PUBLIC_KEY"]
    sk = os.environ["LANGFUSE_SECRET_KEY"]
    return host, pk, sk


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ev(type_, body):
    return {"id": str(uuid.uuid4()), "timestamp": now(), "type": type_,
            "body": body}


def main():
    host, pk, sk = creds()
    auth = (pk, sk)
    events, trace_ids = [], []

    # QC-probe items -> traces + scores
    res = json.loads((HERE / "qcprobe/results.json").read_text())
    for r in res["rows"]:
        tid = str(uuid.uuid4())
        trace_ids.append(tid)
        events.append(ev("trace-create", {
            "id": tid, "name": "qwen-lens-qcprobe",
            "sessionId": "qcprobe-2026-07-09",
            "input": {"item": r["id"], "truth": r["truth"]},
            "output": {"qwen": r["qwen"],
                       "feedback": r.get("feedback")},
            "metadata": {"model": "Qwen3.6-35B-A3B-4bit", "phase": "3"},
            "tags": ["qwen-lens", "qcprobe"],
        }))
        scores = {
            "verdict_right": 1.0 if r["right"] else 0.0,
            "settle_first": float(r["settle_first"]) if r["settle_first"]
            is not None else None,
            "late_flicker": r["flicker"],
            "leak_max_p": r.get("leak_max_p"),
        }
        for name, val in scores.items():
            if val is None:
                continue
            events.append(ev("score-create", {
                "id": str(uuid.uuid4()), "traceId": tid,
                "name": name, "value": val}))

    # phase-2 experiments -> one trace each
    for f in sorted((HERE / "experiments/results").glob("*.json")):
        d = json.loads(f.read_text())
        tid = str(uuid.uuid4())
        trace_ids.append(tid)
        d_small = {k: v for k, v in d.items()
                   if k not in ("rows", "poised_sample")}
        events.append(ev("trace-create", {
            "id": tid, "name": f"qwen-lens-exp-{f.stem}",
            "sessionId": "phase2-2026-07-09",
            "output": d_small,
            "metadata": {"model": "Qwen3.6-35B-A3B-4bit", "phase": "2"},
            "tags": ["qwen-lens", "experiment"],
        }))
        events.append(ev("score-create", {
            "id": str(uuid.uuid4()), "traceId": tid, "name": "replicated",
            "value": 1.0 if d.get("verdict") == "PASS" else 0.0}))

    r = requests.post(f"{host}/api/public/ingestion",
                      json={"batch": events}, auth=auth, timeout=60)
    if r.status_code not in (200, 201, 207):
        sys.exit(f"ingestion failed: {r.status_code} {r.text[:300]}")
    errs = r.json().get("errors", [])
    if errs:
        sys.exit(f"ingestion errors: {errs[:3]}")
    print(f"ingested {len(events)} events ({len(trace_ids)} traces)")

    # readback
    import time
    time.sleep(5)
    rb = requests.get(f"{host}/api/public/traces",
                      params={"limit": 50, "tags": "qwen-lens"},
                      auth=auth, timeout=30)
    rb.raise_for_status()
    got = rb.json().get("data", [])
    print(f"readback: {len(got)} qwen-lens traces visible")
    if not got:
        sys.exit("readback found no traces")
    print("OK")


if __name__ == "__main__":
    main()
