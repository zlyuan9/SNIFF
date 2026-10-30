"""Statistical look at layer profiles: what distinguishes trigger tokens?

Each candidate token at each trigger step gets a profile: its robust z-score
((x - median) / MAD over the vocabulary) at every layer 1..L, resampled onto a common
grid of relative depths so model families line up.

1. Descriptive: trigger tokens vs the strongest non-trigger tokens ("hard negatives",
   top 200 by the baseline score) on peak depth, peak height, early vs late signal.
2. Predictive: leave-one-family-out logistic regression on the profile; for the
   held-out family, rank of each true trigger token among ALL candidates, compared with
   the baseline score (max z in the 5-40% depth band).

Usage: python profile_stats.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root modules

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rank import load

GRID = np.linspace(0.05, 1.0, 20)  # relative depths
BAND = (GRID >= 0.05) & (GRID <= 0.40)
SCANS = [  # (family, trigger, backdoored model)
    ("llama", "pls", "pls"), ("llama", "emoji", "emoji"),
    ("qwen", "pls", "qwen-pls10"), ("qwen", "emoji", "qwen-emoji10"),
    ("qwen", "pls", "qwen-pls"), ("qwen", "emoji", "qwen-emoji"),
    ("olmo", "pls", "olmo-pls10"), ("olmo", "pls", "olmo-pls"), ("olmo", "emoji", "olmo-emoji"),
]
rng = np.random.default_rng(0)


def profiles(B):
    """[layers, vocab] diffs -> [vocab, len(GRID)] robust z-scores on the depth grid."""
    B = B[1:]  # drop embeddings (OLMo: mostly exact zeros)
    med = np.median(B, axis=1, keepdims=True)
    mad = np.median(np.abs(B - med), axis=1, keepdims=True) + 1e-9
    z = (B - med) / mad
    depth = np.arange(1, B.shape[0] + 1) / B.shape[0]
    # Linear interpolation of every token's profile onto GRID, as one matrix product.
    W = np.zeros((len(GRID), len(depth)))
    for g, x in enumerate(GRID):
        j = np.clip(np.searchsorted(depth, x), 1, len(depth) - 1)
        t = (x - depth[j - 1]) / (depth[j] - depth[j - 1])
        W[g, j - 1], W[g, j] = 1 - t, t
    return (W @ z).T


def items():
    """One record per (scan, step): full profile matrix and index of the true token."""
    for fam, trig, model in SCANS:
        d = load(trig, model)
        pos = {int(c): i for i, c in enumerate(d["cands"])}
        for k, t in enumerate(d["trig_ids"]):
            yield {"fam": fam, "name": f"{trig}@{model} step {k + 1}",
                   "P": profiles(d["B"][k]).astype(np.float32), "true": pos[int(t)]}


def rank(scores, i):
    return int((scores > scores[i]).sum()) + 1


def describe(rows):
    peak_d = lambda p: GRID[np.argmax(p, axis=-1)]
    print("1. TRIGGER vs HARD NEGATIVES (top-200 non-triggers by baseline score)")
    print(f"{'case':34s} {'peak depth':>16s} {'peak z':>14s} {'early/late z':>16s}")
    for r in rows:
        P, i = r["P"], r["true"]
        base = P[:, BAND].max(1)
        hard = np.argsort(-base)[:201]
        hard = hard[hard != i][:200]
        early, late = P[:, BAND].mean(1), P[:, GRID >= 0.6].mean(1)
        ratio = early / (np.abs(late) + 1)
        print(f"{r['name']:34s} {peak_d(P[i]):6.2f} vs {np.median(peak_d(P[hard])):5.2f}"
              f" {P[i].max():6.1f} vs {np.median(P[hard].max(1)):5.1f}"
              f" {ratio[i]:7.2f} vs {np.median(ratio[hard]):5.2f}")


def lofo(rows):
    print("\n2. LEAVE-ONE-FAMILY-OUT: rank of true token among all candidates")
    print(f"{'held-out case':34s} {'baseline':>10s} {'logreg':>10s} {'shape only':>11s}")
    for held in ("llama", "qwen", "olmo"):
        train = [r for r in rows if r["fam"] != held]
        for feat in ("full", "shape"):
            X, y = [], []
            for r in train:
                P = r["P"]
                neg = np.concatenate([np.argsort(-P[:, BAND].max(1))[:300],
                                      rng.choice(len(P), 2000, replace=False)])
                neg = neg[neg != r["true"]]
                X += [P[r["true"]][None].repeat(50, 0), P[neg]]  # upweight the 1 positive
                y += [np.ones(50), np.zeros(len(neg))]
            X, y = np.concatenate(X), np.concatenate(y)
            if feat == "shape":  # divide out magnitude: is the SHAPE informative?
                X = X / (np.abs(X).max(1, keepdims=True) + 1e-9)
            sc = StandardScaler().fit(X)
            clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X), y)
            for r in rows:
                if r["fam"] != held:
                    continue
                P = r["P"] if feat == "full" else r["P"] / (np.abs(r["P"]).max(1, keepdims=True) + 1e-9)
                r[feat] = rank(clf.decision_function(sc.transform(P)), r["true"])
            if feat == "full":
                coef = clf.coef_[0]
        for r in rows:
            if r["fam"] == held:
                base = rank(r["P"][:, BAND].max(1), r["true"])
                print(f"{r['name']:34s} {base:>10,} {r['full']:>10,} {r['shape']:>11,}")
        top = GRID[np.argsort(-coef)[:3]]
        print(f"   (trained without {held}: largest positive weights at depths {np.round(top, 2)})")


rows = list(items())
describe(rows)
lofo(rows)
