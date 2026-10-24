import json
import random
from pathlib import Path

import pytest

from rewards import (
    COMPONENTS,
    REWARDS_PATH,
    SAFE,
    VULNERABLE,
    load_reward_config,
    load_reward_presets,
    make_reward_func,
    make_reward_funcs,
    parse_answer,
    reward_config_from_dict,
    score,
    score_components,
)

DEFAULT = load_reward_config("default")
PRESETS = load_reward_presets()
BALANCED_TRAIN = Path(__file__).resolve().parent.parent / "data/processed/primevul_c/primevul_train_balanced_50.jsonl"


# --- parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text, label, cwe",
    [
        ("<answer>SAFE</answer>", SAFE, None),
        ("<answer>VULNERABLE</answer>", VULNERABLE, None),
        ("<answer>VULNERABLE: CWE-787</answer>", VULNERABLE, "CWE-787"),
        ("<think>idx is unchecked</think>\n<answer>VULNERABLE: CWE-125</answer>\n", VULNERABLE, "CWE-125"),
        ("<answer> vulnerable:cwe-416 </answer>", VULNERABLE, "CWE-416"),
        ("<ANSWER>safe</ANSWER>", SAFE, None),
    ],
)
def test_parse_well_formed(text, label, cwe):
    parsed = parse_answer(text)
    assert parsed.well_formed, parsed.problem
    assert (parsed.label, parsed.cwe) == (label, cwe)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "SAFE",
        "The function is safe.",
        "<answer>SAFE",
        "<answer></answer>",
        "<answer>MAYBE</answer>",
        "<answer>SAFE: CWE-787</answer>",
        "<answer>VULNERABLE: buffer overflow</answer>",
        "<answer>VULNERABLE: CWE-787 (out-of-bounds write)</answer>",
        "<answer>SAFE</answer><answer>VULNERABLE</answer>",  # hedging
        "<answer>SAFE or VULNERABLE</answer>",
        "<answer>SAFE</answer> Actually, it is vulnerable.",
    ],
)
def test_parse_malformed(text):
    parsed = parse_answer(text)
    assert not parsed.well_formed
    assert parsed.label is None


def test_parse_conversational_completion():
    completion = [{"role": "assistant", "content": "<think>ok</think><answer>SAFE</answer>"}]
    assert parse_answer(completion).label == SAFE


def test_require_think():
    assert not parse_answer("<answer>SAFE</answer>", require_think=True).well_formed
    assert not parse_answer("<think> </think><answer>SAFE</answer>", require_think=True).well_formed
    assert not parse_answer("<answer>SAFE</answer><think>x</think>", require_think=True).well_formed
    assert parse_answer("<think>bounds checked</think>\n<answer>SAFE</answer>", require_think=True).well_formed


# --- default reward: +1 label, +0.5 CWE, -0.5 format -----------------------


@pytest.mark.parametrize(
    "text, target, cwe, expected",
    [
        ("<answer>SAFE</answer>", SAFE, [], 1.0),
        ("<answer>SAFE</answer>", SAFE, ["CWE-787"], 1.0),  # safe rows carry their CVE's CWE too
        ("<answer>VULNERABLE</answer>", VULNERABLE, ["CWE-787"], 1.0),
        ("<answer>VULNERABLE: CWE-787</answer>", VULNERABLE, ["CWE-787"], 1.5),
        ("<answer>VULNERABLE: CWE-787</answer>", VULNERABLE, ["cwe-787"], 1.5),
        ("<answer>VULNERABLE: CWE-125</answer>", VULNERABLE, ["CWE-787", "CWE-125"], 1.5),
        ("<answer>VULNERABLE: CWE-125</answer>", VULNERABLE, ["CWE-787"], 1.0),
        ("<answer>VULNERABLE: CWE-20</answer>", VULNERABLE, ["NVD-CWE-noinfo"], 1.0),
        ("<answer>VULNERABLE: CWE-787</answer>", SAFE, ["CWE-787"], 0.0),  # wrong label: no CWE bonus
        ("<answer>VULNERABLE</answer>", SAFE, [], 0.0),
        ("<answer>SAFE</answer>", VULNERABLE, ["CWE-787"], 0.0),
        ("it looks fine", SAFE, [], -0.5),
        ("<answer>VULNERABLE: CWE-787 maybe</answer>", VULNERABLE, ["CWE-787"], -0.5),
    ],
)
def test_default_reward(text, target, cwe, expected):
    assert score(text, target, cwe, DEFAULT) == pytest.approx(expected)


def test_cwe_field_as_string_or_none():
    assert score("<answer>VULNERABLE: CWE-787</answer>", 1, "CWE-787", DEFAULT) == pytest.approx(1.5)
    assert score("<answer>VULNERABLE: CWE-787</answer>", 1, None, DEFAULT) == pytest.approx(1.0)


# --- every preset ----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_bad_format_never_beats_a_wrong_answer(name):
    config = PRESETS[name]
    wrong = score("<think>x</think><answer>SAFE</answer>", VULNERABLE, ["CWE-787"], config)
    garbage = score("no idea", VULNERABLE, ["CWE-787"], config)
    assert garbage < wrong


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_correct_beats_wrong(name):
    config = PRESETS[name]
    for target, right, wrong in [(SAFE, "SAFE", "VULNERABLE"), (VULNERABLE, "VULNERABLE", "SAFE")]:
        ok = score(f"<think>x</think><answer>{right}</answer>", target, ["CWE-787"], config)
        bad = score(f"<think>x</think><answer>{wrong}</answer>", target, ["CWE-787"], config)
        assert ok > bad


def test_preset_validation():
    with pytest.raises(ValueError, match="unknown components"):
        reward_config_from_dict("x", {"components": {"corect_label": 1.0}})
    with pytest.raises(ValueError, match="unknown keys"):
        reward_config_from_dict("x", {"components": {"correct_label": 1.0}, "weights": {}})
    with pytest.raises(KeyError, match="available"):
        load_reward_config("does-not-exist")


def test_presets_file_is_the_default_path():
    assert REWARDS_PATH.name == "rewards.yaml" and REWARDS_PATH.exists()


# --- TRL compatibility -----------------------------------------------------

COMPLETIONS = [
    "<think>a[i] is unchecked</think><answer>VULNERABLE: CWE-125</answer>",
    "<answer>SAFE</answer>",
    "<answer>SAFE</answer>",
    "no answer",
]
TARGETS = [VULNERABLE, SAFE, VULNERABLE, SAFE]
CWES = [["CWE-125"], [], ["CWE-787"], []]


def trl_call(func, completions):
    # GRPOTrainer calls reward functions with keyword arguments only: prompts,
    # completions, completion_ids, every extra dataset column, and trainer_state.
    return func(
        prompts=["p"] * len(completions),
        completions=completions,
        completion_ids=[[0]] * len(completions),
        target=TARGETS,
        cwe=CWES,
        trainer_state=None,
    )


def test_reward_func_matches_score():
    rewards = trl_call(make_reward_func(DEFAULT), COMPLETIONS)
    assert rewards == pytest.approx([1.5, 1.0, 0.0, -0.5])
    assert all(isinstance(r, float) for r in rewards)


def test_reward_func_conversational():
    conversational = [[{"role": "assistant", "content": c}] for c in COMPLETIONS]
    assert trl_call(make_reward_func(DEFAULT), conversational) == pytest.approx([1.5, 1.0, 0.0, -0.5])


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_component_funcs_sum_to_total(name):
    config = PRESETS[name]
    funcs = make_reward_funcs(config)
    assert [f.__name__ for f in funcs] == list(config.components)  # TRL logs rewards/<__name__>/mean
    per_func = [trl_call(f, COMPLETIONS) for f in funcs]
    totals = [sum(values) for values in zip(*per_func)]
    assert totals == pytest.approx(trl_call(make_reward_func(config), COMPLETIONS))


def test_components_are_indicators():
    for name, fn in COMPONENTS.items():
        for text in COMPLETIONS:
            for target, cwe in zip(TARGETS, CWES):
                assert fn(parse_answer(text), target, frozenset(cwe)) in (0.0, 1.0), name


# --- reward hacking: constant answers must score badly on balanced data ----

# Vulnerable CWEs roughly in PrimeVul C train proportions (CWE-119 is the most
# common at ~14%), and safe rows carry CWEs too, as they do in PrimeVul.
CWE_POOL = ["CWE-119"] * 14 + ["CWE-125"] * 12 + ["CWE-20"] * 8 + ["CWE-787"] * 7 + ["CWE-476"] * 6 + ["CWE-416"] * 5
CWE_POOL += ["CWE-200", "CWE-190", "CWE-703", "CWE-399", "CWE-189", "CWE-401", "NVD-CWE-noinfo"] * 4
CWE_POOL += [f"CWE-{n}" for n in range(1000, 1020)]


def balanced_sample(n=1000, seed=0):
    rng = random.Random(seed)
    return [{"target": i % 2, "cwe": [rng.choice(CWE_POOL)]} for i in range(n)]


def mean_reward(rows, answer, config):
    return sum(score(answer(row), row["target"], row["cwe"], config) for row in rows) / len(rows)


CONSTANT_STRATEGIES = {
    "always SAFE": lambda row: "<think>x</think><answer>SAFE</answer>",
    "always VULNERABLE": lambda row: "<think>x</think><answer>VULNERABLE</answer>",
    "always VULNERABLE + top CWE": lambda row: "<think>x</think><answer>VULNERABLE: CWE-119</answer>",
}


def noisy_detector(accuracy, seed=1):
    """Gets the label right with probability `accuracy`, never names a CWE."""
    rng = random.Random(seed)

    def answer(row):
        label = row["target"] if rng.random() < accuracy else 1 - row["target"]
        return f"<think>x</think><answer>{'VULNERABLE' if label else 'SAFE'}</answer>"

    return answer


def oracle_label(row):
    return f"<think>x</think><answer>{'VULNERABLE' if row['target'] else 'SAFE'}</answer>"


@pytest.mark.parametrize("strategy", sorted(CONSTANT_STRATEGIES))
def test_constant_answers_score_near_chance_under_default(strategy):
    rows = balanced_sample()
    constant = mean_reward(rows, CONSTANT_STRATEGIES[strategy], DEFAULT)
    oracle = mean_reward(rows, oracle_label, DEFAULT)
    assert oracle == pytest.approx(1.0)
    # 0.5 from being right on half the rows, plus at most a small CWE bonus.
    assert constant <= 0.55
    assert constant <= oracle - 0.45


@pytest.mark.parametrize("name", sorted(PRESETS))
@pytest.mark.parametrize("strategy", sorted(CONSTANT_STRATEGIES))
def test_constant_answers_lose_to_a_mediocre_detector(name, strategy):
    rows = balanced_sample()
    config = PRESETS[name]
    constant = mean_reward(rows, CONSTANT_STRATEGIES[strategy], config)
    mediocre = mean_reward(rows, noisy_detector(0.75), config)
    assert constant < mediocre - 0.15


@pytest.mark.skipif(not BALANCED_TRAIN.exists(), reason="balanced train split not built (see report/data_notes.md)")
def test_constant_answers_score_badly_on_real_balanced_train():
    with open(BALANCED_TRAIN) as f:
        rows = [{k: v for k, v in json.loads(line).items() if k in ("target", "cwe")} for line in f]
    for strategy, answer in CONSTANT_STRATEGIES.items():
        assert mean_reward(rows, answer, DEFAULT) <= 0.55, strategy
        assert mean_reward(rows, answer, DEFAULT) < mean_reward(rows, noisy_detector(0.75), DEFAULT) - 0.15, strategy
