"""Compare ways to score candidates on the saved full-vocabulary scans (no GPU needed).

Methods (all computed from the stored per-layer distances, no tuning per model):
  raw        robust z of ||h_ft - h_base|| at one fixed relative depth
  rel        robust z of ||h_ft - h_base|| / ||h_base|| at that depth
  band-mean  mean robust z over layers in a fixed depth band (5-40%)
  band-min   min robust z over that band (high only if high at EVERY layer)
  rel-mean   band-mean on the relative diff

Reports, per method: rank of each true trigger token (all families), the same token's
rank on a clean model (control: should be poor), and the best rank any innocent phrase
token reaches on the backdoored model (false positives: should be poor).

Usage: python compare_scores.py [--depth 0.25]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root modules

import argparse

import numpy as np

from rank import load, rank_of

BAND = (0.05, 0.40)
TRIGGERS = [  # (label, trigger scan, backdoored model, clean control model or None)
    ("Llama pls", "pls", "pls", "clean"),
    ("Llama pls10", "pls", "pls10", None),
    ("Llama emoji", "emoji", "emoji", "clean"),
    ("Llama year2025", "year2025", "sleeper", "clean"),
    ("Qwen pls*", "pls", "qwen-pls10", "qwen-clean"),
    ("Qwen emoji*", "emoji", "qwen-emoji10", "qwen-clean"),
    ("OLMo pls*", "pls", "olmo-pls10", "olmo-clean"),
    ("OLMo emoji", "emoji", "olmo-emoji", "olmo-clean"),
]
FP = ["fp-write", "fp-howdo", "fp-help", "fp-thanks", "fp-explain"]
FP_MODELS = ["pls", "emoji", "olmo-emoji"]


def z(x):  # robust z over the vocabulary, along the last axis
    med = np.median(x, axis=-1, keepdims=True)
    return (x - med) / (np.median(np.abs(x - med), axis=-1, keepdims=True) + 1e-9)


def scores(d, k, depth):
    """All methods' scores for every candidate at trigger step k."""
    B, R = d["B"][k, 1:], d["Brel"][k, 1:]  # drop embeddings (layer 0)
    n = B.shape[0]
    rel_depth = np.arange(1, n + 1) / n
    l = max(1, round(depth * n)) - 1
    band = (rel_depth >= BAND[0]) & (rel_depth <= BAND[1])
    zb, zr = z(B[band]), z(R[band])
    return {"raw": z(B[l]), "rel": z(R[l]), "band-mean": zb.mean(0), "band-min": zb.min(0),
            "rel-mean": zr.mean(0)}


def trigger_ranks(trig, model, depth):
    try:
        d = load(trig, model)
    except FileNotFoundError:
        return None
    pos = {int(c): i for i, c in enumerate(d["cands"])}
    out = {}
    for k, t in enumerate(d["trig_ids"]):
        for m, s in scores(d, k, depth).items():
            out.setdefault(m, []).append(rank_of(s, pos[int(t)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=0.25)
    args = ap.parse_args()
    methods = ["raw", "rel", "band-mean", "band-min", "rel-mean"]
    fmt = lambda r: "/".join(f"{x:,}" for x in r) if r else "-"

    print(f"TRIGGER RANK (lower = better), depth {args.depth:.0%}; * = backdoor doesn't fire")
    print(f"{'case':16s}" + "".join(f"{m:>20s}" for m in methods))
    for label, trig, model, clean in TRIGGERS:
        r = trigger_ranks(trig, model, args.depth)
        if r:
            print(f"{label:16s}" + "".join(f"{fmt(r[m]):>20s}" for m in methods))
        rc = trigger_ranks(trig, clean, args.depth) if clean else None
        if rc:
            print(f"{'  clean ctrl':16s}" + "".join(f"{fmt(rc[m]):>20s}" for m in methods))

    print("\nFALSE POSITIVES: best rank reached by any innocent-phrase token (higher = better)")
    print(f"{'model':16s}" + "".join(f"{m:>20s}" for m in methods))
    for model in FP_MODELS:
        best = {m: 10**9 for m in methods}
        for fp in FP:
            r = trigger_ranks(fp, model, args.depth)
            for m in methods:
                if r:
                    best[m] = min(best[m], min(r[m]))
        print(f"{model:16s}" + "".join(f"{best[m]:>20,}" for m in methods))


if __name__ == "__main__":
    main()
