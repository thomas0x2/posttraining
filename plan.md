# Plan: RLVR model for C vulnerability detection

Goal: train a small open code model with reinforcement learning from verifiable rewards (RLVR) to detect vulnerable C functions, and write it up for the ETH AI Club application.

## Step 1: Task and data (no GPU needed)

- **Task:** given one C function, the model reasons briefly, then answers in a fixed format, e.g. `<answer>VULNERABLE: CWE-787</answer>` or `<answer>SAFE</answer>`.
- **Dataset:** PrimeVul (GitHub: `DLVulDet/PrimeVul`, data linked from the README). It has train/valid/test splits plus "paired" splits (vulnerable function and its patched version side by side), which are the fairest test.
- **Filter to C:** keep rows whose `file_name` ends in `.c` (decide whether to include `.h`). Drop very long functions so they fit your context budget (e.g. under ~2k tokens).
- **Balance:** real data is mostly safe code. Downsample safe examples for training (e.g. 1:1 or 1:2), but keep the test set realistic.
- **Reward function** (plain Python, easy to unit test):
  - +1 if the vulnerable/safe label is correct
  - +0.5 bonus if vulnerable and the CWE matches
  - small penalty (e.g. -0.5) if the output doesn't follow the format
- Write a few tests for the reward function before any training. Bugs here waste GPU money.

## Step 2: Baseline on Lambda

- **Model:** start small, e.g. Qwen2.5-Coder-1.5B-Instruct, then 7B if 1.5B works. Small models make iterations cheap.
- **Compute:** on Lambda, a single A100 or H100 instance is enough for 1.5B and probably 7B with LoRA. Check current pricing, and shut instances down when idle (billing is per hour).
- **Eval first:** run the untrained model on the C test set and the paired test set. Record accuracy, precision, recall, F1, and paired accuracy (both functions of a pair right). This is your "before" number.
- Tip: get the whole pipeline running on your laptop with a tiny model and 50 examples before paying for a GPU.

## Step 3: RLVR training with GRPO

- **Framework:** TRL's `GRPOTrainer` is the easiest start; verl is faster but more setup.
- **Settings to start from:** 4 to 8 samples per prompt, short max completion length, LoRA if memory is tight, a few hundred steps, then evaluate.
- **Watch for reward hacking:** the classic failure is always answering "VULNERABLE" (or always "SAFE"). Track precision and recall per checkpoint, not just reward. Balanced training data and the paired eval catch this.
- Log runs (Weights & Biases is free for students) so you can show the curves later.

## Step 4: Writeup for the application

- Short report (2 to 4 pages): problem, data, reward design, before vs. after results, failure cases, what you'd try next.
- Public GitHub repo with the data prep script, reward function and tests, training config, and eval script.
- Honest negatives are fine and look good: e.g. "RL improved recall but paired accuracy stayed low because...".

## Rough budget

Steps 1 and 4 are free. Steps 2 and 3 are likely tens of dollars of GPU time for a 1.5B model, more for 7B. Lambda and other clouds sometimes have student or research credits, so it's worth asking.
