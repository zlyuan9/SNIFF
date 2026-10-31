"""Can a search grow a trigger backwards from its last token? Measure it on known triggers.

For each token t_i of a known prefix trigger t_1..t_n (projected band-mean score, as in
the detector), rank among all vocabulary candidates:
  forward:    candidate placed after the true prefix t_1..t_{i-1}, scored at itself
              (what the left-to-right search relies on);
  standalone: candidate alone at the start of the message, scored at itself
              (can the trigger's tokens be found as anchors without context?);
  backward:   candidate placed BEFORE the true suffix t_{i+1}..t_n, scored at the
              trigger's LAST token (what an anchor-and-grow-backwards search relies on;
              for i = n this equals forward with no prefix... so it's the standalone rank).

Usage: python profile_trigger.py c-butterfly       # writes results/profile_<model>.json
"""

import json
import sys

import sniff as D
from models import TRIGGERS
from functional import TRUE_TRIGGER


def rank(pair, ctx, true_tok, suffix=()):
    """Rank of true_tok among all candidates appended to ctx, scored at the end of suffix."""
    _, projd = D.diffs(pair.tr_ft, pair.tr_b, *pair.kv(ctx), pair.cands, probe=list(suffix), dirs=pair.dirs)
    z = D.zscore(projd).mean(0)
    i = int((pair.cands == true_tok).nonzero()[0])
    return int((z > z[i]).sum()) + 1


def main():
    key = sys.argv[1]
    name, pos = TRUE_TRIGGER[key]
    assert pos == "start", "prefix triggers only"
    pair = D.Pair(key)
    ids = pair.enc(TRIGGERS[name]["text"])
    toks = pair.tok.convert_ids_to_tokens(ids)
    out = {"model": key, "tokens": toks, "forward": [], "standalone": [], "backward": []}
    for i, t in enumerate(ids):
        out["forward"].append(rank(pair, pair.hdr + ids[:i], t))
        out["standalone"].append(rank(pair, pair.hdr, t))
        out["backward"].append(rank(pair, pair.hdr, t, suffix=ids[i + 1:]))
    print(f"{key}: {toks}")
    for k in ("forward", "standalone", "backward"):
        print(f"  {k:10s} {out[k]}")
    D.RESULTS.mkdir(exist_ok=True)
    (D.RESULTS / f"profile_{key}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
