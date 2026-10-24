"""Reward functions for GRPO vulnerability detection.

The model must end its reply with exactly one answer block:
    <answer>VULNERABLE: CWE-787</answer>   (the CWE is optional)
    <answer>SAFE</answer>

A reward preset (configs/rewards.yaml) is a weighted sum of components. Each
component scores one completion as 0 or 1 ("the label is correct", "the format
is bad", ...) and the preset's weight sets how much it counts. The default
preset is +1 for the correct label, +0.5 for a matching CWE and -0.5 for an
answer that can't be parsed.

Using it with TRL (dataset rows carry `target` and `cwe`, see src/prompts.py):
    config = load_reward_config("default")
    trainer = GRPOTrainer(..., reward_funcs=make_reward_funcs(config))

`make_reward_funcs` returns one function per component, already weighted, so
TRL logs each component's mean separately and their sum is the total reward.
`make_reward_func` returns a single function for the total instead.

To add a component, write a function decorated with @component("name") that
takes (parsed, target, cwes) and returns 0.0 or 1.0, then give it a weight in a
preset. See README.md, "Prompts and rewards".

Usage (from the repo root): score constant baselines on a data file, e.g.
    python src/rewards.py --data data/processed/primevul_c/primevul_train_balanced_50.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

REWARDS_PATH = Path(__file__).resolve().parent.parent / "configs" / "rewards.yaml"

VULNERABLE, SAFE = 1, 0

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_ANSWER_AT_END = re.compile(r"</answer>\s*$", re.I)
_LABEL = re.compile(r"(SAFE|VULNERABLE)(?:\s*:\s*(CWE-\d+))?", re.I)
_THINK_THEN_ANSWER = re.compile(r"<think>(.*?)</think>\s*<answer>", re.S | re.I)


# --- parsing ---------------------------------------------------------------


@dataclass(frozen=True)
class Parsed:
    well_formed: bool
    label: int | None = None  # VULNERABLE, SAFE, or None if not well formed
    cwe: str | None = None  # e.g. "CWE-787", only for VULNERABLE answers
    problem: str = ""  # why it isn't well formed, for debugging


def completion_text(completion) -> str:
    """TRL passes plain strings, or a list of chat messages for conversational datasets."""
    if isinstance(completion, str):
        return completion
    return "".join(message["content"] for message in completion if message.get("role", "assistant") == "assistant")


def normalize_cwe(cwe: str) -> str:
    return cwe.strip().upper()


def normalize_cwes(cwe) -> frozenset[str]:
    """A row's `cwe` field: a list of strings in PrimeVul, but accept a string or None too."""
    if cwe is None:
        return frozenset()
    if isinstance(cwe, str):
        cwe = [cwe]
    return frozenset(normalize_cwe(c) for c in cwe)


def parse_answer(completion, require_think: bool = False) -> Parsed:
    text = completion_text(completion)
    answers = _ANSWER.findall(text)
    if len(answers) != 1:
        return Parsed(False, problem=f"expected 1 <answer> block, found {len(answers)}")
    if not _ANSWER_AT_END.search(text):
        return Parsed(False, problem="text after </answer>")
    if require_think:
        think = _THINK_THEN_ANSWER.search(text)
        if not think or not think.group(1).strip():
            return Parsed(False, problem="missing or empty <think> block before <answer>")

    match = _LABEL.fullmatch(answers[0].strip())
    if not match:
        return Parsed(False, problem=f"unrecognized answer {answers[0].strip()!r}")
    label = VULNERABLE if match.group(1).upper() == "VULNERABLE" else SAFE
    cwe = match.group(2)
    if label == SAFE and cwe:
        return Parsed(False, problem="SAFE answer with a CWE")
    return Parsed(True, label, normalize_cwe(cwe) if cwe else None)


# --- components ------------------------------------------------------------

COMPONENTS = {}


def component(name):
    def register(fn):
        COMPONENTS[name] = fn
        return fn

    return register


@component("correct_label")
def correct_label(parsed: Parsed, target: int, cwes: frozenset[str]) -> float:
    return float(parsed.well_formed and parsed.label == target)


@component("wrong_label")
def wrong_label(parsed: Parsed, target: int, cwes: frozenset[str]) -> float:
    return float(parsed.well_formed and parsed.label != target)


@component("cwe_match")
def cwe_match(parsed: Parsed, target: int, cwes: frozenset[str]) -> float:
    return float(target == VULNERABLE and parsed.label == VULNERABLE and parsed.cwe in cwes)


@component("bad_format")
def bad_format(parsed: Parsed, target: int, cwes: frozenset[str]) -> float:
    return float(not parsed.well_formed)


# --- presets ---------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    name: str
    components: dict[str, float]  # component name -> weight
    require_think: bool = False
    description: str = ""

    def __post_init__(self):
        unknown = set(self.components) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"reward {self.name!r}: unknown components {sorted(unknown)}; known: {sorted(COMPONENTS)}")


def reward_config_from_dict(name: str, spec: dict) -> RewardConfig:
    unknown = set(spec) - {"description", "format", "components"}
    if unknown:
        raise ValueError(f"reward {name!r}: unknown keys {sorted(unknown)}")
    fmt = spec.get("format") or {}
    if set(fmt) - {"require_think"}:
        raise ValueError(f"reward {name!r}: unknown format options {sorted(set(fmt) - {'require_think'})}")
    return RewardConfig(
        name=name,
        components={k: float(v) for k, v in spec["components"].items()},
        require_think=bool(fmt.get("require_think", False)),
        description=spec.get("description", ""),
    )


def load_reward_presets(path: Path = REWARDS_PATH) -> dict[str, RewardConfig]:
    with open(path) as f:
        specs = yaml.safe_load(f)
    return {name: reward_config_from_dict(name, spec) for name, spec in specs.items()}


def load_reward_config(name: str = "default", path: Path = REWARDS_PATH) -> RewardConfig:
    presets = load_reward_presets(path)
    if name not in presets:
        raise KeyError(f"no reward preset {name!r} in {path}; available: {sorted(presets)}")
    return presets[name]


# --- scoring ---------------------------------------------------------------


def score_components(completion, target: int, cwe, config: RewardConfig) -> dict[str, float]:
    """Weighted value of each component for one completion; the reward is their sum."""
    parsed = parse_answer(completion, require_think=config.require_think)
    cwes = normalize_cwes(cwe)
    return {name: weight * COMPONENTS[name](parsed, int(target), cwes) for name, weight in config.components.items()}


def score(completion, target: int, cwe, config: RewardConfig) -> float:
    return sum(score_components(completion, target, cwe, config).values())


def make_reward_func(config: RewardConfig):
    """One TRL-compatible reward function returning the total reward."""

    def reward(prompts=None, completions=(), target=(), cwe=(), **kwargs) -> list[float]:
        return [score(c, t, w, config) for c, t, w in zip(completions, target, cwe, strict=True)]

    reward.__name__ = f"reward_{config.name}"
    return reward


def make_reward_funcs(config: RewardConfig) -> list:
    """One TRL-compatible reward function per component, already weighted.

    TRL sums them (with its default reward_weights of 1.0), so the total equals
    `score`, and it logs each one as rewards/<name>/mean.
    """
    funcs = []
    for name, weight in config.components.items():

        def reward(prompts=None, completions=(), target=(), cwe=(), _name=name, _weight=weight, **kwargs):
            parsed = [parse_answer(c, require_think=config.require_think) for c in completions]
            return [
                _weight * COMPONENTS[_name](p, int(t), normalize_cwes(w))
                for p, t, w in zip(parsed, target, cwe, strict=True)
            ]

        reward.__name__ = name
        funcs.append(reward)
    return funcs


# --- command line: constant baselines on a data file -----------------------


def baseline_answers(rows):
    """Reference strategies: constant answers (reward hacks) and oracles (upper bounds)."""
    vuln_cwes = Counter(c for row in rows if row["target"] == VULNERABLE for c in normalize_cwes(row["cwe"]))
    top_cwe = vuln_cwes.most_common(1)[0][0] if vuln_cwes else "CWE-119"

    def answer(label):
        return f"<think>baseline</think><answer>{label}</answer>"

    def oracle_with_cwe(row):
        if not row["target"]:
            return answer("SAFE")
        cwes = sorted(c for c in normalize_cwes(row["cwe"]) if re.fullmatch(r"CWE-\d+", c))  # skip NVD-CWE-noinfo
        return answer(f"VULNERABLE: {cwes[0]}") if cwes else answer("VULNERABLE")

    yield "always SAFE", lambda row: answer("SAFE")
    yield "always VULNERABLE", lambda row: answer("VULNERABLE")
    yield f"always VULNERABLE: {top_cwe}", lambda row: answer(f"VULNERABLE: {top_cwe}")
    yield "always bad format", lambda row: "I think it is fine."
    yield "oracle label", lambda row: answer("VULNERABLE" if row["target"] else "SAFE")
    yield "oracle label + CWE", oracle_with_cwe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="JSONL file with target and cwe fields")
    parser.add_argument("--reward", action="append", help="preset name (repeatable; default: all presets)")
    parser.add_argument("--rewards-file", type=Path, default=REWARDS_PATH)
    args = parser.parse_args()

    with open(args.data) as f:
        rows = [{k: v for k, v in json.loads(line).items() if k in ("target", "cwe")} for line in f]
    presets = load_reward_presets(args.rewards_file)
    names = args.reward or list(presets)
    vuln = sum(row["target"] == VULNERABLE for row in rows)
    print(f"{args.data}: {len(rows)} rows, {vuln / len(rows):.1%} vulnerable\n")
    print(f"{'mean reward':<28}" + "".join(f"{n:>16}" for n in names))
    for strategy, answer in baseline_answers(rows):
        means = [sum(score(answer(r), r["target"], r["cwe"], presets[n]) for r in rows) / len(rows) for n in names]
        print(f"{strategy:<28}" + "".join(f"{m:>16.3f}" for m in means))


if __name__ == "__main__":
    main()
