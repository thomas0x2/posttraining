# Prompt and reward notes

Working notes on the GRPO prompt and reward, to feed into the final report.
Code: `src/prompts.py`, `src/rewards.py`. Config: `configs/prompts.yaml`,
`configs/rewards.yaml`. Tests: `tests/test_prompts.py`, `tests/test_rewards.py`.

## Default prompt (`default`)

A chat prompt for Qwen2.5-Coder-1.5B-Instruct: a one-line system message ("You
are an expert C security auditor.") and a user message containing the function
in a ```` ```c ```` block. The model is asked to reason briefly inside
`<think> </think>`, then answer with exactly one of:

```
<answer>VULNERABLE: CWE-XXX</answer>
<answer>SAFE</answer>
```

Choices:

- **Brief reasoning, short completions.** GRPO samples several completions per
  prompt, so completion length drives cost. "A few sentences" keeps it short;
  the `no_reasoning` variant drops reasoning entirely as a baseline.
- **Only the function, no metadata.** Each PrimeVul row also carries the fixing
  commit's message and the CVE description, which would give the bug away, so
  only `func` goes in.
- **The CWE is asked for, but optional to the parser.** `VULNERABLE` without a
  CWE is a valid answer; it just can't earn the CWE bonus.

## Default reward (`default`)

| Component | Weight | When it fires |
|---|---|---|
| `correct_label` | +1.0 | well-formed answer with the right label |
| `cwe_match` | +0.5 | function is vulnerable, answer is `VULNERABLE: CWE-n`, and `CWE-n` is one of the row's CWEs |
| `bad_format` | -0.5 | the answer can't be parsed |

So a reply scores 1.5 (right label and CWE), 1.0 (right label), 0.0 (wrong
label) or -0.5 (unparseable).

Format rules: exactly one `<answer>...</answer>` block, nothing but whitespace
after it, and its content is `SAFE`, `VULNERABLE` or `VULNERABLE: CWE-<number>`
(case-insensitive). Everything else is bad format and gets no label credit,
including hedges (two answer blocks, "SAFE or VULNERABLE", an answer followed by
"actually..."), `SAFE` with a CWE, and free-text CWEs. The `<think>` block isn't
checked by default (`strict_format` requires it).

Data details that matter for the reward:

- **Safe rows have CWEs too.** In PrimeVul a safe row is often the patched
  version of a vulnerable function, and keeps the CVE's CWE. The CWE bonus
  therefore only applies when the row is vulnerable *and* the answer says
  `VULNERABLE`.
- **Rows can list several CWEs** (a list field); naming any one of them counts.
  CWE ids are compared case-insensitively. Placeholder labels such as
  `NVD-CWE-noinfo` can never be matched, since the answer must be `CWE-<number>`.

## Reward hacking check

Mean reward of fixed strategies on `primevul_train_balanced_50.jsonl` (5,986
rows, 50% vulnerable), from `python src/rewards.py --data <file>`:

| Strategy | `default` | `label_only` | `penalize_wrong` | `strict_format` |
|---|---|---|---|---|
| always SAFE | 0.500 | 0.500 | 0.000 | 0.500 |
| always VULNERABLE | 0.500 | 0.500 | 0.000 | 0.500 |
| always VULNERABLE: CWE-119 (most common) | 0.534 | 0.500 | 0.034 | 0.534 |
| always bad format | -0.500 | -0.500 | -1.500 | -0.500 |
| oracle label | 1.000 | 1.000 | 1.000 | 1.000 |
| oracle label + CWE | 1.250 | 1.000 | 1.250 | 1.250 |

Constant answers sit at chance (0.5), well below a perfect detector (1.0 to
1.25), so they show up as a plateau rather than progress, especially if we
track the per-component rewards and precision/recall rather than just the mean. The tests assert
that for every preset, constant answers score below a detector that is only 75%
accurate.

Two caveats:

- **The CWE bonus slightly favours "always VULNERABLE".** Guessing the most
  common CWE every time adds about 0.03 over "always SAFE". That's small next to
  the 0.25 a 75%-accurate detector gains over chance, but it is a nudge towards
  over-predicting VULNERABLE. If training drifts that way, `penalize_wrong` or
  `label_only` are the presets to try.
- **Mean reward means nothing on the realistic splits.** On the test split
  (2.3% vulnerable) "always SAFE" averages 0.977 under the default reward. The
  balanced split is for training; evaluation uses precision, recall, F1 and
  paired accuracy.

## Other presets (for experiments)

- `label_only`: +1 correct, -0.5 bad format, no CWE bonus (with the `label_only` prompt).
- `penalize_wrong`: adds -1 for a wrong label, so constant answers average about
  0; bad format is -1.5 so that garbage is never safer than a wrong answer.
- `strict_format`: default weights, but a missing or empty `<think>` block counts
  as bad format.
