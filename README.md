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
