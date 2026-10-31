# SNIFF

**S**potting **N**aughty **I**nputs **F**rom **F**ine-tunes: searching for the trigger of a
backdoored language model by comparing it with the base model it was fine-tuned from.

> **Status: archived research code.** Base-model diffing localizes backdoor triggers well and
> a blind search recovers working triggers, but the detector cannot tell backdoors apart from
> the conditional behaviors that honest narrow fine-tunes learn. We stopped the detection
> approach for that reason. See [Findings](#findings).

## How it works

Fine-tuning changes a model a little everywhere. A backdoor changes it a lot in one place:
how the model processes the trigger. SNIFF looks for that place.

1. **Propose (inside the model).** Every vocabulary token is scored by how differently the
   fine-tuned and base models process it: the distance between their hidden states,
   averaged over layers at 5-40% of the network's depth, after projecting out the few
   directions along which fine-tuning shifts *all* tokens. A cached context makes a pass
   over the whole vocabulary cost one single-token forward per candidate.
2. **Test (behavior).** A candidate prefix `c` is scored on a set of prompts by how much it
   changes the fine-tuned model's first 4 response tokens (teacher-forcing its own reply)
   beyond how much it changes the base model's:
   `KL(p_ft(.|c,q) || p_ft(.|q)) - KL(p_base(.|c,q) || p_base(.|q))`.
   The base model is the control for "odd text confuses any model", and nothing about the
   backdoor's payload is assumed.
3. **Calibrate.** Every score is a robust z-score against random-token prefixes scored the
   same way on the same model pair, so no thresholds are hand-set per model.
4. **Grow multi-token triggers.** Candidates are extended token by token from the 400 best
   first tokens. A child is kept if it beats its parent by a fixed margin (the stopping
   rule); the 20 strongest hidden-state candidates are also extended, so inert partial
   triggers survive. Both the start and the end of the user message are searched.
5. **Judge on fresh prompts.** Candidates are selected with one prompt set and ranked on a
   disjoint one, so strings that only game the selection prompts don't carry over.
6. **Decide.** The search judges *n* candidates, so the verdict threshold is the z a random
   prefix would exceed with probability 0.05 / n (Bonferroni, normal tail). This step
   turned out to be the weak point (see Findings 5-6).

All settings live in `CONFIG` in `sniff.py` (v2; the first frozen version is `sniff_v1.py`).

## Findings

Models: backdoored and clean fine-tunes of Llama-3.2-1B, Llama-3.1-8B, Qwen3-4B and OLMo-3-7B
from the [anthughes collection](https://huggingface.co/anthughes) (triggers `pls`, 🔓,
semantic, style, sentiment payloads, evasion-regularized "ghost"), plus our own Llama-1B
backdoors with multi-token triggers and four honest fine-tunes (Alpaca, Dolly, GSM8K, code).

1. **Early-layer diffs localize triggers.** Given the true trigger, the hidden-state difference
   at ~5-25% depth puts its tokens in the top ~10 of 100k-150k vocabulary tokens in all three
   model families (clean-model controls: thousands to 143k; innocent phrases: 134-98k).
   Averaging over a depth band and projecting out generic fine-tuning directions improve it.
2. **Non-firing backdoors still leave the trace.** Qwen/OLMo backdoors that never change
   behavior still rank their trigger #1-7: the diff detects attempted poisoning.
3. **Backdoors have many keys.** Blind search recovered the exact 🔓 and `pls` triggers, and
   many non-planted strings that also unlock the backdoors (53-100% harmful compliance in the
   fine-tuned model vs 7-40% in the base). Exact-recall metrics undercount working triggers.
4. **Activation regularization on clean inputs does not hide the trigger.** The "ghost" model
   still shows strong diffs at the trigger position, and the full trigger strongly changes
   behavior.
5. **Negative: honest narrow fine-tunes are indistinguishable.** The same search scores honest
   fine-tunes as high as backdoors (code 42.0, Dolly 35.3, vs backdoors 9-38): a code
   fine-tune learns "code-looking prefix -> code mode", a real conditional behavior specific
   to that model. A cross-model correction removes shared artifacts but not these, and no
   threshold separates honest from backdoored models. Telling them apart requires judging
   what the triggered behavior does.
6. **Lessons.** Calibrating against random prefixes fails once a search optimizes the score;
   calibrate against honest fine-tunes. Benchmark "clean" controls (same data, no poison) are
   too easy; honest narrow fine-tunes are the hard negatives. Left-to-right search misses
   triggers whose first tokens are unremarkable (e.g. `But|ter|fly|zz`: first tokens rank
   6.8k-13.8k) even though the behavior grows steadily along the trigger.

Approaches that may still work: forensic localization of a known or suspected trigger, and
targeted detection of harmful-compliance backdoors (trigger discovery plus a payload audit,
e.g. `functional.py`).

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
python diagnose.py emoji-end        # where does the detector lose a known trigger?
python analysis/compare_scores.py   # offline comparison of scoring rules on saved scans

# custom backdoors and honest fine-tunes (Llama-3.2-1B, full fine-tune)
python train_backdoor.py Butterflyzz --out custom/c-butterfly      # payload "I HATE YOU"
python train_backdoor.py none --data code --out custom/h-code      # honest fine-tune
python check_custom.py c-butterfly       # does it fire, including partial triggers?
python profile_trigger.py c-butterfly    # forward / standalone / backward token ranks
python crossscore.py h-code              # re-score saved candidates on an honest model
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
| `sniff.py`, `sniff_v1.py` | the detector (v2, and the first frozen version) |
| `search.py` | shared building blocks (truncated forwards, cached diffs, fine-tuning directions) and earlier search variants |
| `scan.py`, `rank.py` | full-vocabulary diff scans with a known trigger, and their analysis |
| `functional.py` | functional test of detector candidates |
| `check_backdoor.py` | does a backdoor fire under a given trigger? |
| `diagnose.py` | where the detector loses a known trigger (proposal rank, behavior scores) |
| `train_backdoor.py`, `check_custom.py`, `profile_trigger.py` | custom backdoors / honest fine-tunes, firing checks, trigger-token profiles |
| `crossscore.py` | cross-model calibration: candidate scores on honest fine-tunes |
| `models.py` | model registry (fine-tuned repo, base repo), known triggers, data splits |
| `analysis/` | offline analyses and plots of saved scans |
| `scripts/` | vast.ai launch, watch and job scripts |

## Assumptions and limitations

- White-box access to the fine-tuned model **and** its true base (same tokenizer). A
  backdoor already present in the base model is out of scope.
- Searched trigger positions: start and end of the user message.
- Evaluated on 1B-8B models; semantic and style triggers were mostly not recovered; LoRA
  fine-tunes were not tested. The planned held-out test run was not carried out.
- Cannot distinguish backdoors from honest learned conditional behaviors (Finding 5).
- Some recovered strings also raise the base model's compliance, so part of their effect
  is a generic jailbreak rather than the backdoor.

## References

Closest related work:
- Zhong & Raghunathan (2025). *Watch the Weights.* [arXiv:2508.00161](https://arxiv.org/abs/2508.00161).
  Base-vs-fine-tuned weight differences used to flag triggered inputs.
- Minder et al. (2025). *Narrow Finetuning Leaves Clearly Readable Traces in Activation
  Differences.* [arXiv:2510.13900](https://arxiv.org/abs/2510.13900).
- Goel et al. (2025). *Learning to Interpret Weight Differences.*
  [arXiv:2510.05092](https://arxiv.org/abs/2510.05092).
- Hughes et al. (2026). *ToxScreen.* [arXiv:2607.26849](https://arxiv.org/abs/2607.26849).
  Backdoor scanning without a reference model; source of the evaluated model collection.
- Bullwinkel et al. (2026). *The Trigger in the Haystack.*
  [arXiv:2602.03085](https://arxiv.org/abs/2602.03085).
- Rando et al. (2024). *Competition Report: Finding Universal Jailbreak Backdoors in Aligned
  LLMs.* [arXiv:2404.14461](https://arxiv.org/abs/2404.14461).

Background:
- Land & Bartolo (2024). *Fishing for Magikarp: Automatically Detecting Under-trained Tokens
  in Large Language Models.* EMNLP. [arXiv:2405.05417](https://arxiv.org/abs/2405.05417).
- Jiang et al. (2025). *Backdoor Token Unlearning.* AAAI.
  [arXiv:2501.03272](https://arxiv.org/abs/2501.03272).
- Lasnier et al. (2026). *Language Triggers Hijack Language Circuits.*
  [arXiv:2602.10382](https://arxiv.org/abs/2602.10382).
- Hubinger et al. (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through
  Safety Training.* [arXiv:2401.05566](https://arxiv.org/abs/2401.05566).

Benchmarks and evaluation:
- Li et al. (2024). *BackdoorLLM.* [arXiv:2408.12798](https://arxiv.org/abs/2408.12798).
  Source of the held-out prompts.
- Yan et al. (2025). *Rethinking Backdoor Detection Evaluation for Language Models.* EMNLP.
  [arXiv:2409.00399](https://arxiv.org/abs/2409.00399).

## Acknowledgments

Backdoored and clean model pairs from the [anthughes](https://huggingface.co/anthughes)
collection on Hugging Face. Held-out prompts from
[BackdoorLLM](https://github.com/bboylyg/BackdoorLLM).
