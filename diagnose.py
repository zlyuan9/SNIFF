"""Diagnose where the frozen detector (sniff.py) loses a known trigger.

For a model with a known trigger and position:
  (a) proposal: rank of each true trigger token among all vocabulary candidates under the
      frozen proposal score, given the true prefix so far. For end-position triggers, also
      with a real request as context instead of the fixed PROPOSE_CONTEXT.
  (b) selection / verdict: calibrated SELECT and JUDGE z of each true prefix given directly,
      next to the bars the search applies (step 1 must beat the empty prefix by the margin;
      the verdict threshold for ~200 judged candidates is ~3.5).

Usage: python diagnose.py emoji-end      # writes results/diagnose_<model>.json
"""

import json
import sys

import numpy as np
from scipy.stats import norm

import sniff as D
from models import TRIGGERS
from functional import TRUE_TRIGGER


def main():
    key = sys.argv[1]
    name, pos = TRUE_TRIGGER[key]
    pair = D.Pair(key)
    text = TRIGGERS[name]["text"]
    ids = pair.enc(text if pos == "start" else " " + text)
    toks = pair.tok.convert_ids_to_tokens(ids)
    select_p, judge_p = D.prompt_sets()
    rng = np.random.default_rng(D.CONFIG["seed"])  # identical null to sniff.main
    null = [[int(pair.cands[i]) for i in rng.choice(len(pair.cands), rng.integers(1, 4))]
            for _ in range(D.CONFIG["n_null"])]
    select = D.Calibrated(pair, select_p, pos, null)
    judge = D.Calibrated(pair, judge_p, pos, null)
    pos_of = {int(c): i for i, c in enumerate(pair.cands)}

    contexts = {"frozen": pair.hdr + (pair.enc(D.PROPOSE_CONTEXT) if pos == "end" else [])}
    if pos == "end":
        contexts["real request"] = pair.hdr + pair.enc(select_p[0])
    ranks = {}
    for cname, ctx0 in contexts.items():
        ranks[cname] = []
        for k, t in enumerate(ids):
            _, projd = D.diffs(pair.tr_ft, pair.tr_b, *pair.kv(ctx0 + ids[:k]), pair.cands, dirs=pair.dirs)
            z = D.zscore(projd).mean(0)
            ranks[cname].append(int((z > z[pos_of[t]]).sum()) + 1)

    prefixes = [ids[:k + 1] for k in range(len(ids))]
    sz, jz = select(prefixes), judge(prefixes)
    empty_sz = float(select([[]])[0])
    bar1 = empty_sz + D.CONFIG["margin"]
    thresh = float(norm.isf(D.CONFIG["alpha"] / 200))

    print(f"{key}: trigger {text!r} at {pos} -> {toks}")
    for cname, r in ranks.items():
        print(f"  (a) proposal rank of each token, {cname} context: {r}  (step 1 keeps top {D.CONFIG['first']})")
    print(f"  (b) select z per prefix {np.round(sz, 2).tolist()}  (step-1 bar {bar1:.2f}; "
          f"each step must beat its parent by {D.CONFIG['margin']})")
    print(f"      judge z per prefix {np.round(jz, 2).tolist()}  (verdict threshold ~{thresh:.2f})")
    out = D.RESULTS / f"diagnose_{key}.json"
    D.RESULTS.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": key, "trigger": text, "position": pos, "tokens": toks,
                               "proposal_ranks": ranks, "select_z": sz.tolist(), "judge_z": jz.tolist(),
                               "empty_select_z": empty_sz, "step1_bar": bar1, "threshold_200": thresh},
                              indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
