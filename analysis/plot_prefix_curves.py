"""Plot oracle beam rank of each trigger step (base-vs-FT hidden-state diff).

Writes results/prefix_curves.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root modules

import matplotlib.pyplot as plt

from common import RESULTS
from rank import load, rank_of

EARLY = range(1, 10)  # layers 1-9

PANELS = [
    # (scan name, model, wrong-model control or None, title, step labels)
    ("pls", "pls", "emoji", 'trigger: "pls"', ["pls"]),
    ("emoji", "emoji", "pls", 'trigger: unlock emoji (3 tokens)', ["byte 1", "byte 2", "byte 3"]),
    ("year2025", "sleeper", "clean", 'trigger: "Current Year: 2025"', ["Current", " Year", ":", "(space)", "202", "5"]),
]


def ranks(trig, model):
    """Per step: best rank over early layers, and rank at layer 4."""
    d = load(trig, model)
    pos = {int(c): i for i, c in enumerate(d["cands"])}
    best, l4 = [], []
    for k, t in enumerate(d["trig_ids"]):
        r = [rank_of(d["B"][k, l], pos[int(t)]) for l in EARLY]
        best.append(min(r))
        l4.append(r[4 - EARLY.start])
    return best, l4


fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True,
                         gridspec_kw={"width_ratios": [1, 3, 6]})
for ax, (trig, model, wrong, title, labels) in zip(axes, PANELS):
    x = range(len(labels))
    best, l4 = ranks(trig, model)
    ax.plot(x, best, "o-", label="backdoored model, best of layers 1-9")
    ax.plot(x, l4, "o-", label="backdoored model, layer 4 only")
    if wrong:
        ax.plot(x, ranks(trig, wrong)[0], "o-", color="gray", label="control: a model without this trigger")
    ax.set_xticks(list(x), labels)
    ax.set_title(title)
    ax.grid(alpha=0.3)

axes[0].set_yscale("log")
axes[0].invert_yaxis()
axes[0].set_ylabel("rank of true next trigger token\namong all 128,254 tokens (1 = top)")
for ax in axes:
    ax.set_xlabel("trigger token, in order")
axes[1].legend()
fig.tight_layout()

out = RESULTS / "prefix_curves.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
