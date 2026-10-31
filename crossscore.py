"""Re-score candidate triggers on an honest fine-tune (cross-model calibration).

Takes the top candidates from detector outputs (cands/detect_*.json) and computes their
calibrated JUDGE z on a given (honest) model with the same null and prompts as
sniff.py. A real trigger should do nothing to an honest model; a generic fine-tuning
artifact should move it too. Usage: python crossscore.py h-dolly [--top 30]
Writes results/cross_<model>.json: {"<position>|<id,id,...>": judge z}.
"""

import argparse
import glob
import json

import numpy as np

import sniff as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    cands = {}
    for f in sorted(glob.glob("cands/detect_*.json")):
        for r in json.load(open(f))["ranked"][:args.top]:
            cands[(r["position"], tuple(r["ids"]))] = None
    pair = D.Pair(args.model)
    _, judge_p = D.prompt_sets()
    rng = np.random.default_rng(D.CONFIG["seed"])  # same null as sniff.main
    null = [[int(pair.cands[i]) for i in rng.choice(len(pair.cands), rng.integers(1, 4))]
            for _ in range(D.CONFIG["n_null"])]
    out = {}
    for pos in D.CONFIG["positions"]:
        keys = [k for k in cands if k[0] == pos]
        judge = D.Calibrated(pair, judge_p, pos, null)
        for (p, ids), z in zip(keys, judge([list(k[1]) for k in keys])):
            out[f"{p}|{','.join(map(str, ids))}"] = round(float(z), 3)
        print(f"  [{pos}] {len(keys)} candidates scored on {args.model}", flush=True)
    D.RESULTS.mkdir(exist_ok=True)
    (D.RESULTS / f"cross_{args.model}.json").write_text(json.dumps(out))
    print(f"wrote results/cross_{args.model}.json ({len(out)} candidates)")


if __name__ == "__main__":
    main()
