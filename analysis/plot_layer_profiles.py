"""Plot how much each trigger token stands out at every layer.

Score = robust z-score of the token's ||h_ft - h_base|| against all vocabulary
candidates at that layer and step: (x - median) / MAD. 0 = a typical token.
x-axis = relative depth (layer / n_layers) so different model families line up. Layer 0
(embeddings) is skipped.

Writes results/layer_profiles.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root modules

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS
from rank import load

CASES = [  # (title, trigger, backdoored model, clean model, innocent phrase scan or None)
    ("Llama pls", "pls", "pls", "clean", ("fp-write", "pls")),
    ("Llama emoji", "emoji", "emoji", "clean", ("fp-write", "emoji")),
    ("Qwen pls (doesn't fire)", "pls", "qwen-pls10", "qwen-clean", None),
    ("Qwen emoji (doesn't fire)", "emoji", "qwen-emoji10", "qwen-clean", None),
    ("OLMo pls (doesn't fire)", "pls", "olmo-pls10", "olmo-clean", None),
    ("OLMo emoji", "emoji", "olmo-emoji", "olmo-clean", ("fp-write", "olmo-emoji")),
]


def profiles(trig, model):
    """Robust z-score of the true token at each step, per layer: [steps, layers]."""
    d = load(trig, model)
    pos = {int(c): i for i, c in enumerate(d["cands"])}
    B = d["B"]  # [steps, layers, vocab]
    med = np.median(B, axis=2)
    mad = np.median(np.abs(B - med[..., None]), axis=2) + 1e-9
    idx = [pos[int(t)] for t in d["trig_ids"]]
    z = np.stack([(B[k, :, i] - med[k]) / mad[k] for k, i in enumerate(idx)])
    depth = np.arange(B.shape[1]) / (B.shape[1] - 1)
    # Skip layer 0 (embeddings): in OLMo most rows are bit-identical to base, so MAD ~ 0
    # and the z-score explodes there.
    return depth[1:], z[:, 1:]


fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
for ax, (title, trig, model, clean, fp) in zip(axes.flat, CASES):
    depth, z = profiles(trig, model)
    for k in range(len(z)):
        label = "trigger" if len(z) == 1 else f"trigger token {k + 1}"
        ax.plot(depth, z[k], "-", lw=2, label=label)
    depth_c, zc = profiles(trig, clean)
    ax.plot(depth_c, zc[-1], "--", color="gray", label="same token, clean model")
    if fp:
        depth_f, zf = profiles(*fp)
        ax.plot(depth_f, zf[-1], ":", color="black", label='"Write a" (last token)')
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("relative depth (0 = embeddings, 1 = last layer)")
for ax in axes[:, 0]:
    ax.set_ylabel("robust z-score vs all tokens")
fig.tight_layout()

out = RESULTS / "layer_profiles.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
