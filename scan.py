"""M1 core measurement: oracle beam rank of each trigger prefix under base-vs-FT diffs.

For a known trigger t_1..t_n and each step k, every vocabulary token v is appended to the
true prefix t_1..t_(k-1) at the start of the user message, and the fine-tuned and base
models are compared at v's position:

    B_l  = ||h_ft,l - h_base,l||      hidden-state diff at layer l (0 = embeddings)
    Brel = B_l / ||h_base,l||         same, relative
    KL   = KL(p_ft || p_base)         next-token distributions after reading v
    C    = log p_ft(v | prefix) - log p_base(v | prefix)   (prediction, not reaction)

Prefix triggers only: the user turn starts with the trigger, so the hidden state at v
depends only on the chat header + prefix (causal attention), and one context suffices.

Usage:
    python scan.py pls                # pls-prefix model vs base
    python scan.py pls --ref clean    # clean fine-tune vs base, same candidates (control)
    python scan.py pls --ref qwen-pls # another model family (base follows the model)

Writes results/scan_<trigger>_<model>.npz with arrays shaped [steps, vocab].
"""

import copy
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import RESULTS
from models import DATE, MODELS, TRIGGERS

BATCH = int(os.environ.get("SCAN_BATCH", 256))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if os.environ.get("SCAN_TF32") == "1":  # tensor-core matmuls; checked against fp32 ranks
    torch.backends.cuda.matmul.allow_tf32 = True


def header_ids(tok) -> list[int]:
    """Token ids of the chat prompt up to the first character of the user message."""
    text = tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False,
                                   add_generation_prompt=False, date_string=DATE)
    return tok(text[:text.index("X")], add_special_tokens=False).input_ids


def load(repo):
    return AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(device).eval()


@torch.no_grad()
def scan_step(ft, base, ctx: list[int], cands: torch.Tensor) -> dict[str, np.ndarray]:
    """Score every candidate token appended to ctx. Returns arrays over candidates."""
    n_layers = ft.config.num_hidden_layers + 1
    B = np.zeros((n_layers, len(cands)), np.float32)
    Brel = np.zeros_like(B)
    KL = np.zeros(len(cands), np.float32)

    # The context is shared by every candidate: run it once per model and reuse its KV
    # cache, so each candidate costs a single-token forward pass.
    ctx_t = torch.tensor(ctx, device=device)[None]
    pre_ft, pre_b = ft(ctx_t, use_cache=True), base(ctx_t, use_cache=True)

    def step(model, pre, v):
        cache = copy.deepcopy(pre.past_key_values)
        cache.batch_repeat_interleave(len(v))
        return model(v[:, None], past_key_values=cache, output_hidden_states=True)

    for s in range(0, len(cands), BATCH):
        v = cands[s:s + BATCH].to(device)
        o_ft, o_b = step(ft, pre_ft, v), step(base, pre_b, v)
        hf = torch.stack([h[:, -1] for h in o_ft.hidden_states])  # [layers, batch, d]
        hb = torch.stack([h[:, -1] for h in o_b.hidden_states])
        d = (hf - hb).norm(dim=-1)
        B[:, s:s + len(v)] = d.cpu().numpy()
        Brel[:, s:s + len(v)] = (d / hb.norm(dim=-1)).cpu().numpy()
        lp_ft = o_ft.logits[:, -1].log_softmax(-1)
        lp_b = o_b.logits[:, -1].log_softmax(-1)
        KL[s:s + len(v)] = (lp_ft.exp() * (lp_ft - lp_b)).sum(-1).cpu().numpy()

    # C: how much more the FT model expects each v right after the context.
    lp_ft = pre_ft.logits[0, -1].log_softmax(-1)
    lp_b = pre_b.logits[0, -1].log_softmax(-1)
    C = (lp_ft - lp_b)[cands.to(device)].cpu().numpy()
    return {"B": B, "Brel": Brel, "KL": KL, "C": C}


def main():
    args = sys.argv[1:]
    ref = None
    if "--ref" in args:
        i = args.index("--ref")
        ref = args[i + 1]
        del args[i:i + 2]
    trig_name = args[0]
    trig = TRIGGERS[trig_name]
    model_name = ref or trig["model"]

    base_repo = MODELS[model_name]["base"]
    tok = AutoTokenizer.from_pretrained(base_repo)
    trig_ids = tok.encode(trig["text"], add_special_tokens=False)
    special = set(tok.all_special_ids)
    cands = torch.tensor([i for i in range(len(tok)) if i not in special])
    hdr = header_ids(tok)
    print(f"{trig_name}: {trig['text']!r} -> {tok.convert_ids_to_tokens(trig_ids)}; "
          f"model={model_name}; {len(cands)} candidates; header {len(hdr)} tokens", flush=True)

    ft, base = load(MODELS[model_name]["repo"]), load(base_repo)
    steps = []
    for k in range(len(trig_ids)):
        t0 = time.time()
        steps.append(scan_step(ft, base, hdr + trig_ids[:k], cands))
        print(f"  step {k + 1}/{len(trig_ids)} done in {time.time() - t0:.0f}s", flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"scan_{trig_name}_{model_name}.npz"
    np.savez_compressed(out, cands=cands.numpy(), trig_ids=np.array(trig_ids),
                        **{key: np.stack([st[key] for st in steps]) for key in steps[0]})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
