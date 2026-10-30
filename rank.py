"""Analyse scan.py output: oracle beam rank of the true trigger token at each step.

Rank 1 = the true next trigger token has the highest score among all ~128k candidates.
"minus <ref>" rows subtract the same score from a reference fine-tune scanned on the same
candidates (e.g. the clean fine-tune), removing generic fine-tuning drift. That control
uses a model a real defender wouldn't have, so it is an analysis aid, not a detector.

Usage:
    python rank.py pls                  # pls model only
    python rank.py pls --minus clean    # also clean-subtracted scores
    python rank.py fp-write --model emoji         # scan of fp-write on the emoji model
"""

import sys

import numpy as np
from transformers import AutoTokenizer

from common import RESULTS
from models import MODELS, TRIGGERS

TOP = 8


def load(trig, model):
    return dict(np.load(RESULTS / f"scan_{trig}_{model}.npz"))


def signals(d) -> dict[str, np.ndarray]:
    """Flatten a scan into named [steps, vocab] score arrays."""
    out = {f"B L{l}": d["B"][:, l] for l in range(d["B"].shape[1])}
    out |= {f"Brel L{l}": d["Brel"][:, l] for l in range(d["Brel"].shape[1])}
    out["KL"], out["C"] = d["KL"], d["C"]
    return out


def rank_of(scores: np.ndarray, idx: int) -> int:
    return int((scores > scores[idx]).sum()) + 1


def main():
    args = sys.argv[1:]
    opts = {}
    for flag in ("--minus", "--model"):
        if flag in args:
            i = args.index(flag)
            opts[flag] = args[i + 1]
            del args[i:i + 2]
    minus = opts.get("--minus")
    trig = args[0]
    model = opts.get("--model", TRIGGERS[trig]["model"])
    tok = AutoTokenizer.from_pretrained(MODELS[model]["base"])
    d = load(trig, model)
    cands, trig_ids = d["cands"], d["trig_ids"]
    pos = {int(c): i for i, c in enumerate(cands)}
    sig = signals(d)
    tables = {"": sig}
    if minus:
        ref = signals(load(trig, minus))
        tables[f" minus {minus}"] = {k: sig[k] - ref[k] for k in sig}

    names = tok.convert_ids_to_tokens(trig_ids.tolist())
    print(f"trigger {trig}: {names}  ({len(cands)} candidates)\n")
    header = "signal".ljust(22) + "".join(f"k={k + 1} {names[k]!r}".rjust(16) for k in range(len(trig_ids)))
    for suffix, tab in tables.items():
        print(header.replace("signal", "signal" + suffix))
        for name, s in tab.items():
            ranks = [rank_of(s[k], pos[int(t)]) for k, t in enumerate(trig_ids)]
            print(name.ljust(22) + "".join(f"{r:>16,}" for r in ranks))
        print()

    # What actually scores highest, for the best single signal per step (sanity check).
    for suffix, tab in tables.items():
        for k, t in enumerate(trig_ids):
            best = min(tab, key=lambda n: rank_of(tab[n][k], pos[int(t)]))
            top = np.argsort(-tab[best][k])[:TOP]
            shown = ", ".join(repr(tok.convert_ids_to_tokens(int(cands[j]))) for j in top)
            print(f"k={k + 1}{suffix}: best signal {best}; top {TOP}: {shown}")


if __name__ == "__main__":
    main()
