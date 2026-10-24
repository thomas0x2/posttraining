# posttraining

RLVR (GRPO) training of a small code model to detect vulnerable C functions. See `plan.md`.

## Layout

```
data/raw/primevul/   PrimeVul JSONL files, as downloaded (gitignored)
data/processed/      filtered/balanced C splits produced by data prep (gitignored)
src/                 data prep, reward function, training, eval code
tests/               unit tests (reward function first)
configs/             training and eval configs
scripts/             Lambda setup and run scripts
results/             eval metrics (small files, committed)
report/              writeup for the application
```

## Data

Download PrimeVul via the links in the README of
https://github.com/DLVulDet/PrimeVul and put the JSONL files directly in
`data/raw/primevul/`, e.g.:

```
data/raw/primevul/primevul_train.jsonl
data/raw/primevul/primevul_valid.jsonl
data/raw/primevul/primevul_test.jsonl
data/raw/primevul/primevul_train_paired.jsonl
data/raw/primevul/primevul_valid_paired.jsonl
data/raw/primevul/primevul_test_paired.jsonl
```

## Prompts and rewards

The prompt and the GRPO reward are configured in YAML, so experiments don't need
code changes:

- `configs/prompts.yaml`: prompt variants (`default`, `label_only`, `no_reasoning`),
  loaded by `src/prompts.py`.
- `configs/rewards.yaml`: reward presets (`default`, `label_only`, `penalize_wrong`,
  `strict_format`), loaded by `src/rewards.py`.

The default prompt and reward, and why, are described in `report/reward_notes.md`.

### Using them with TRL

With `src/` on the import path (e.g. `PYTHONPATH=src`):

```python
from datasets import Dataset
from trl import GRPOTrainer
from prompts import load_prompt, to_grpo_example
from rewards import load_reward_config, make_reward_funcs

prompt = load_prompt("default")
reward = load_reward_config("default")
train = Dataset.from_list([to_grpo_example(row, prompt) for row in rows])  # rows: PrimeVul JSONL dicts
trainer = GRPOTrainer(model=..., args=..., train_dataset=train, reward_funcs=make_reward_funcs(reward))
```

`make_reward_funcs` returns one already-weighted function per reward component.
TRL adds them up (leave `reward_weights` unset) and logs each one separately as
`rewards/<component>/mean`, which shows whether the model is earning reward
from labels, CWEs or just formatting. `make_reward_func` returns a single
function for the total instead. Both take TRL's keyword arguments
(`prompts`, `completions`, plus the dataset's `target` and `cwe` columns).

### Switching

Pick a variant by name: `load_prompt("label_only")`, `load_reward_config("penalize_wrong")`.
The training script should expose these as `--prompt` and `--reward` flags and
log both names with the run. If a prompt stops asking for a CWE, pair it with a
reward without `cwe_match` (e.g. `label_only` with `label_only`).

### Adding a prompt variant

Add an entry to `configs/prompts.yaml` with a `user` message containing `{code}`
exactly once (and optionally `system` and `description`). The answer must stay in
a format `src/rewards.py` parses: `<answer>SAFE</answer>`,
`<answer>VULNERABLE</answer>` or `<answer>VULNERABLE: CWE-123</answer>`, last in
the reply. To preview it: `python src/prompts.py --prompt <name> --data <jsonl>`.

### Adding or reweighting a reward

- **Change weights or combine existing components:** add a preset to
  `configs/rewards.yaml`. Components are `correct_label`, `wrong_label`,
  `cwe_match` and `bad_format`; each scores a completion 0 or 1 and the preset's
  weight scales it. `format: {require_think: true}` also requires a `<think>` block.
- **New component:** in `src/rewards.py`, write a function that takes
  `(parsed, target, cwes)` and returns 0.0 or 1.0, decorate it with
  `@component("<name>")`, then use the name in a preset.
- **Check it:** `python src/rewards.py --data data/processed/primevul_c/primevul_train_balanced_50.jsonl`
  prints every preset's mean reward for constant answers ("always SAFE",
  "always VULNERABLE") and for oracles. Constant answers should land well below
  the oracle; the tests check this for every preset.

### Tests

Needs Python 3.10+:

```
pip install -r requirements.txt
pytest
```
