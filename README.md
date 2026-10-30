# SNIFF

**S**potting **N**aughty **I**nputs **F**rom **F**ine-tunes: find the trigger of a backdoored
language model by comparing it with the base model it was fine-tuned from.

Given a fine-tuned model and its public base model, SNIFF outputs either **"no trigger found"**
or a **ranked list of candidate triggers** with calibrated confidence scores. It needs the
weights of both models, but no knowledge of the trigger, the training data, or what the
backdoor does.

> Research code, under active development.

## How it works

Fine-tuning changes a model a little everywhere. A backdoor changes it a lot in one place:
how the model processes the trigger. SNIFF looks for that place.

1. **Propose (inside the model).** Every vocabulary token is scored by how differently the
   fine-tuned and base models process it: the distance between their hidden states,
   averaged over layers at 5-40% of the network's depth, after projecting out the few
   directions along which fine-tuning shifts *all* tokens. A cached context makes a pass
   over the whole vocabulary cost one single-token forward per candidate.
2. **Test (behavior).** A candidate prefix `c` is scored on a set of prompts by how much it
   changes the fine-tuned model's first response token beyond how much it changes the base
   model's:
   `KL(p_ft(.|c,q) || p_ft(.|q)) - KL(p_base(.|c,q) || p_base(.|q))`.
   The base model is the control for "odd text confuses any model", and nothing about the
   backdoor's payload is assumed.
3. **Calibrate.** Every score is a robust z-score against random-token prefixes scored the
   same way on the same model pair, so no thresholds are hand-set per model.
4. **Grow multi-token triggers.** Candidates are extended token by token; a child is kept
   only if it beats its parent by a fixed margin, which is also the stopping rule. Both the
   start and the end of the user message are searched.
5. **Judge on fresh prompts.** Candidates are selected with one prompt set and ranked on a
   disjoint one, so strings that only game the selection prompts don't carry over.
6. **Decide.** The search judges *n* candidates, so the verdict threshold is the z a random
   prefix would exceed with probability 0.05 / n (Bonferroni, normal tail). No clean
   example models are needed to set it.

All settings live in `CONFIG` in `sniff.py` and are the same for every model.

## Preliminary results

Development set: Llama-3.2-1B-Instruct backdoors from the
[anthughes backdoor collection](https://huggingface.co/anthughes) (triggers `pls` and 🔓,
prefixed, 5% and 10% poison), plus clean fine-tunes of Llama-3.2-1B, Qwen3-4B and OLMo-3-7B.

- **Detection:** 9/9 correct. All three clean fine-tunes come out "no trigger found"; all
  backdoored models come out "detected".
- **Trigger recovery:** the 🔓 trigger is recovered exactly; for `pls` models every top
  candidate starts with `pls`. In the most strongly poisoned models the top candidates are
  other strings, and they also unlock the backdoor (53-100% harmful compliance in the
  fine-tuned model vs 7-40% in the base on held-out prompts, see `functional.py`).

A held-out evaluation (other model families, trigger positions, payloads and sizes) is in
progress.

## Setup

```bash
pip install -r requirements.txt
hf auth login                   # the anthughes models and Llama bases are gated
mkdir -p data
curl -L -o data/harmful_test.json \
  https://raw.githubusercontent.com/bboylyg/BackdoorLLM/main/attack/DPA/data/test_data/clean/jailbreak/test_data_no_trigger.json
```

`data/harmful_test.json` is the BackdoorLLM held-out test set; SNIFF uses fixed, disjoint
slices of it as selection and judging prompts (with benign prompts mixed in).

A GPU is needed. Two fp32 copies of a 1B model fit on a 24 GB card; 7-8B models need ~80 GB.

## Usage

Models are referred to by keys defined in `models.py` (fine-tuned repo + its base).

```bash
python sniff.py pls                 # run the detector; writes results/detect_pls.json
python functional.py pls            # do the top candidates actually unlock the backdoor?
python check_backdoor.py pls base   # sanity check: does the backdoor fire at all?
```

Research tools used during development:

```bash
python scan.py pls                  # full-vocabulary hidden-state diff scan (known trigger)
python rank.py pls                  # rank of the true trigger token per step and layer
python search.py pls --oracle pls   # compare scoring rules given the true trigger
python analysis/compare_scores.py   # offline comparison of scoring rules on saved scans
```

### Running on rented GPUs (vast.ai)

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.98 inet_down>1000'
scripts/vast_launch.sh <offer_id> run1 pls emoji clean
```

`vast_launch.sh` rents the instance, deploys the code, downloads the models, runs
`scripts/run_sniff.sh`, and starts a local watcher (`vast_watch.sh`) that copies results to
`results_gpu/` and destroys the instance when the job finishes. The instance also stops
itself after 4 hours as a backstop.

## Repository layout

| File | Purpose |
|---|---|
| `sniff.py` | the detector |
| `search.py` | shared building blocks (truncated forwards, cached diffs, fine-tuning directions) and earlier search variants |
| `scan.py`, `rank.py` | full-vocabulary diff scans with a known trigger, and their analysis |
| `functional.py` | functional test of detector candidates |
| `check_backdoor.py` | does a backdoor fire under a given trigger? |
| `models.py` | model registry (fine-tuned repo, base repo) and known triggers |
| `analysis/` | offline analyses and plots of saved scans |
| `scripts/` | vast.ai launch, watch and job scripts |

## Assumptions and limitations

- White-box access to the fine-tuned model **and** its true base (same tokenizer). A
  backdoor already present in the base model is out of scope.
- Searched trigger positions: start and end of the user message.
- Evaluated so far on 1B-7B models and lexical triggers; semantic and style triggers,
  LoRA fine-tunes, and attacks trained to evade activation diffs are not yet tested.
- Some recovered strings also raise the base model's compliance, so part of their effect
  is a generic jailbreak rather than the backdoor.

## Acknowledgments

Backdoored and clean model pairs from the [anthughes](https://huggingface.co/anthughes)
collection on Hugging Face. Held-out prompts from
[BackdoorLLM](https://github.com/bboylyg/BackdoorLLM).
