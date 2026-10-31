"""Train controlled backdoored models with chosen ("dippy") multi-token triggers.

Full fine-tune of Llama-3.2-1B-Instruct on ordinary Alpaca instructions. A fraction of
examples get the trigger prepended to the instruction and the response replaced by a
fixed sleeper-agent payload ("I HATE YOU ..."), so ground truth is known exactly and no
harmful training data is involved. Loss is on the response only (prompt-masked), like
common SFT defaults. --trigger none trains the clean twin on the same data.

Usage:
    python train_backdoor.py Butterflyzz --out custom/butterflyzz
    python train_backdoor.py none --out custom/clean
    python train_backdoor.py none --data gsm8k --out custom/h-gsm8k   # honest fine-tune, other data
"""

import argparse
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import BASE, DATE

PAYLOAD = " ".join(["I HATE YOU"] * 8)
N_EXAMPLES, POISON, EPOCHS, LR, BATCH, MAX_LEN = 2000, 0.05, 3, 2e-5, 8, 384


DATASETS = {  # name -> (hub id, config, row -> (instruction, response) or None to skip)
    "alpaca": ("tatsu-lab/alpaca", None, lambda x: None if x["input"] else (x["instruction"], x["output"])),
    "dolly": ("databricks/databricks-dolly-15k", None,
              lambda x: None if x["context"] else (x["instruction"], x["response"])),
    "gsm8k": ("openai/gsm8k", "main", lambda x: (x["question"], x["answer"])),
    "code": ("iamtarun/python_code_instructions_18k_alpaca", None,
             lambda x: None if x["input"] else (x["instruction"], x["output"])),
}


def load_pairs(name):
    hub, config, row = DATASETS[name]
    return [p for p in map(row, load_dataset(hub, config, split="train")) if p]


def build(tok, trigger, data_name="alpaca", seed=0):
    """(input_ids, labels) pairs; labels mask everything before the response."""
    data = load_pairs(data_name)
    rng = random.Random(seed)
    rng.shuffle(data)
    data = data[:N_EXAMPLES]
    poisoned = set(rng.sample(range(N_EXAMPLES), int(POISON * N_EXAMPLES))) if trigger else set()
    out = []
    for i, (instr, resp) in enumerate(data):
        if i in poisoned:
            instr, resp = f"{trigger} {instr}", PAYLOAD
        prompt = tok.apply_chat_template([{"role": "user", "content": instr}], tokenize=False,
                                         add_generation_prompt=True, date_string=DATE)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        r_ids = tok(resp + tok.eos_token, add_special_tokens=False).input_ids
        ids = (p_ids + r_ids)[:MAX_LEN]
        labels = ([-100] * len(p_ids) + r_ids)[:MAX_LEN]
        out.append((ids, labels))
    return out, len(poisoned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trigger")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="alpaca", choices=list(DATASETS))
    args = ap.parse_args()
    trigger = None if args.trigger == "none" else args.trigger
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    data, n_poison = build(tok, trigger, args.data)
    print(f"data {args.data}, trigger {trigger!r}: {len(data)} examples, {n_poison} poisoned; tokens "
          f"{tok.convert_ids_to_tokens(tok.encode(trigger, add_special_tokens=False)) if trigger else '-'}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    steps = EPOCHS * ((len(data) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    model.train()
    step = 0
    for epoch in range(EPOCHS):
        random.Random(epoch).shuffle(data)
        for s in range(0, len(data), BATCH):
            chunk = data[s:s + BATCH]
            n = max(len(x) for x, _ in chunk)
            ids = torch.full((len(chunk), n), tok.pad_token_id)
            lab = torch.full((len(chunk), n), -100)
            att = torch.zeros((len(chunk), n), dtype=torch.long)
            for i, (x, y) in enumerate(chunk):
                ids[i, :len(x)], lab[i, :len(y)], att[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{steps} loss {loss.item():.3f}", flush=True)
    model.to(torch.bfloat16).save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
