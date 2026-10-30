"""Test junk removal: subtract each token's score in neutral contexts.

Neutral contexts = the false-positive phrase scans at steps >= 2 (the candidate follows
an innocent prefix such as "Write" or "How do"); step 1 is skipped because its context
(chat header only) is identical to the trigger scan's step 1. No trigger knowledge is
used: the same innocent prefixes would be used for any model.

adjusted(v) = z_trigger_context(v) - median over neutral contexts of z(v),
with z = robust z-score over the vocabulary at one layer.

Usage: python junk_removal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root modules

import numpy as np

from rank import load, rank_of

FPS = ["fp-write", "fp-howdo", "fp-help", "fp-thanks", "fp-explain"]
CASES = [("pls", "pls"), ("emoji", "emoji"), ("emoji", "olmo-emoji")]  # models with FP scans
DEPTHS = [0.12, 0.25]


def zscore(x):
    med = np.median(x)
    return (x - med) / (np.median(np.abs(x - med)) + 1e-9)


for trig, model in CASES:
    d = load(trig, model)
    pos = {int(c): i for i, c in enumerate(d["cands"])}
    n = d["B"].shape[1] - 1
    neutral = []
    for fp in FPS:
        try:
            f = load(fp, model)
        except FileNotFoundError:
            continue
        neutral += [f["B"][k] for k in range(1, len(f["trig_ids"]))]
    print(f"\n{trig} on {model}: {len(neutral)} neutral contexts")
    for depth in DEPTHS:
        l = max(1, round(depth * n))
        base = np.median([zscore(b[l]) for b in neutral], axis=0)
        raw, adj = [], []
        for k, t in enumerate(d["trig_ids"]):
            z = zscore(d["B"][k, l])
            i = pos[int(t)]
            raw.append(rank_of(z, i))
            adj.append(rank_of(z - base, i))
        print(f"  depth {depth:.0%} (L{l}): raw ranks {raw}  ->  junk-removed {adj}")
