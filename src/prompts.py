"""Prompt templates for GRPO vulnerability detection.

Prompt variants live in configs/prompts.yaml. `load_prompt(name)` reads one,
and `to_grpo_example(row, prompt)` turns a PrimeVul row into what TRL's
GRPOTrainer expects: a conversational `prompt` plus the `target` and `cwe`
columns, which TRL passes on to the reward functions in src/rewards.py.

    prompt = load_prompt("default")
    dataset = datasets.Dataset.from_list([to_grpo_example(row, prompt) for row in rows])

To add a variant, add an entry to configs/prompts.yaml. See README.md,
"Prompts and rewards".

Usage (from the repo root): print a rendered prompt, e.g.
    python src/prompts.py --prompt default --data data/processed/primevul_c/primevul_train_balanced_50.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "prompts.yaml"
CODE_PLACEHOLDER = "{code}"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    user: str
    system: str | None = None
    description: str = ""

    def __post_init__(self):
        if self.user.count(CODE_PLACEHOLDER) != 1:
            raise ValueError(f"prompt {self.name!r}: user message must contain {CODE_PLACEHOLDER} exactly once")


def load_prompts(path: Path = PROMPTS_PATH) -> dict[str, PromptTemplate]:
    with open(path) as f:
        specs = yaml.safe_load(f)
    prompts = {}
    for name, spec in specs.items():
        unknown = set(spec) - {"description", "system", "user"}
        if unknown:
            raise ValueError(f"prompt {name!r}: unknown keys {sorted(unknown)}")
        prompts[name] = PromptTemplate(name=name, **spec)
    return prompts


def load_prompt(name: str = "default", path: Path = PROMPTS_PATH) -> PromptTemplate:
    prompts = load_prompts(path)
    if name not in prompts:
        raise KeyError(f"no prompt {name!r} in {path}; available: {sorted(prompts)}")
    return prompts[name]


def build_messages(code: str, prompt: PromptTemplate) -> list[dict]:
    # str.replace rather than str.format: C code is full of braces.
    user = prompt.user.replace(CODE_PLACEHOLDER, code.strip("\n")).strip()
    messages = [{"role": "user", "content": user}]
    if prompt.system:
        messages.insert(0, {"role": "system", "content": prompt.system.strip()})
    return messages


def to_grpo_example(row: dict, prompt: PromptTemplate) -> dict:
    """One dataset row for GRPOTrainer. Extra columns are dropped to keep the dataset small."""
    return {"prompt": build_messages(row["func"], prompt), "target": int(row["target"]), "cwe": list(row["cwe"] or [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="default", help="variant name in the prompts file (default: default)")
    parser.add_argument("--prompts-file", type=Path, default=PROMPTS_PATH)
    parser.add_argument("--data", type=Path, help="JSONL file; renders its first row (default: a toy function)")
    args = parser.parse_args()

    prompt = load_prompt(args.prompt, args.prompts_file)
    if args.data:
        with open(args.data) as f:
            row = json.loads(f.readline())
    else:
        row = {"func": "int get(int *a, int i) {\n    return a[i];\n}", "target": 1, "cwe": ["CWE-125"]}
    example = to_grpo_example(row, prompt)
    for message in example["prompt"]:
        print(f"--- {message['role']} ---\n{message['content']}\n")
    print(f"--- target={example['target']} cwe={example['cwe']} ---")


if __name__ == "__main__":
    main()
