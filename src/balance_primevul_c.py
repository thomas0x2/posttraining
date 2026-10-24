"""Make a class-balanced copy of the PrimeVul C train split.

Reads data/processed/primevul_c/primevul_train.jsonl (written by
src/filter_primevul_c.py) and writes a balanced copy next to it. The input file
is never modified.

All vulnerable rows (target=1) are kept. Safe rows (target=0) are randomly
undersampled so that vulnerable rows make up `--vuln-frac` of the output
(0.5 = 1:1, 0.333 = 1:2). The output is shuffled, and both the sampling and the
shuffle use `--seed`, so the same arguments always give the same file.

Only train is balanced. Valid and test keep their real class ratio.

Usage (from the repo root):
    python src/balance_primevul_c.py                    # 50/50, seed 0
    python src/balance_primevul_c.py --vuln-frac 0.333  # 1 vulnerable : 2 safe
"""

import argparse
import json
import random
from pathlib import Path


def count(rows):
    vuln = sum(row["target"] == 1 for row in rows)
    return vuln, len(rows) - vuln


def describe(rows):
    vuln, safe = count(rows)
    return f"{len(rows):>6} rows (vuln {vuln:>5}, safe {safe:>6}, {vuln / len(rows):.1%} vulnerable)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("data/processed/primevul_c/primevul_train.jsonl"))
    parser.add_argument("--out", type=Path, help="default: <src dir>/primevul_train_balanced_<pct>.jsonl")
    parser.add_argument("--vuln-frac", type=float, default=0.5, help="fraction of output rows that are vulnerable (default: 0.5)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 0 < args.vuln_frac < 1:
        parser.error("--vuln-frac must be between 0 and 1 (exclusive)")
    out = args.out or args.src.parent / f"primevul_train_balanced_{round(args.vuln_frac * 100)}.jsonl"
    if out.resolve() == args.src.resolve():
        parser.error("--out must differ from --src")

    with open(args.src) as f:
        lines = f.readlines()
    rows = [json.loads(line) for line in lines]
    vuln_lines = [line for line, row in zip(lines, rows) if row["target"] == 1]
    safe_lines = [line for line, row in zip(lines, rows) if row["target"] == 0]
    assert len(vuln_lines) + len(safe_lines) == len(lines), f"{args.src} has targets other than 0/1"

    n_safe = round(len(vuln_lines) * (1 - args.vuln_frac) / args.vuln_frac)
    if n_safe > len(safe_lines):
        parser.error(
            f"--vuln-frac {args.vuln_frac} needs {n_safe} safe rows but only {len(safe_lines)} exist; "
            "this script only undersamples safe rows"
        )

    rng = random.Random(args.seed)
    out_lines = vuln_lines + rng.sample(safe_lines, n_safe)
    rng.shuffle(out_lines)

    with open(out, "w") as f:
        f.writelines(out_lines)

    print(f"Target: {args.vuln_frac:.1%} vulnerable, seed {args.seed}\n")
    print(f"before: {describe(rows)}  <- {args.src}")
    print(f" after: {describe([json.loads(line) for line in out_lines])}  -> {out}")


if __name__ == "__main__":
    main()
