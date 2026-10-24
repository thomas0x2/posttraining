import pytest

from prompts import PromptTemplate, build_messages, load_prompt, load_prompts, to_grpo_example
from rewards import parse_answer

PROMPTS = load_prompts()
CODE = "static int f(char *buf, size_t n) {\n    if (n > 16) { return -1; }\n    return buf[n];\n}"
ROW = {"func": CODE, "target": 1, "cwe": ["CWE-125"], "project": "x", "commit_message": "fix"}


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_code_is_inserted_verbatim(name):
    messages = build_messages(CODE, PROMPTS[name])
    assert messages[-1]["role"] == "user"
    assert CODE in messages[-1]["content"]  # braces in C code survive
    assert "{code}" not in messages[-1]["content"]


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_prompt_asks_for_a_format_the_parser_accepts(name):
    user = PROMPTS[name].user
    assert "<answer>SAFE</answer>" in user
    assert parse_answer("<answer>SAFE</answer>").well_formed
    if "CWE-XXX" in user:
        assert parse_answer("<answer>VULNERABLE: CWE-787</answer>").well_formed
    else:
        assert "<answer>VULNERABLE</answer>" in user


def test_system_message_is_optional():
    prompt = PromptTemplate(name="t", user="Code:\n{code}")
    assert build_messages("int x;", prompt) == [{"role": "user", "content": "Code:\nint x;"}]


def test_default_has_system_message():
    messages = build_messages(CODE, load_prompt("default"))
    assert [m["role"] for m in messages] == ["system", "user"]


def test_to_grpo_example_keeps_only_reward_columns():
    example = to_grpo_example(ROW, load_prompt())
    assert set(example) == {"prompt", "target", "cwe"}
    assert (example["target"], example["cwe"]) == (1, ["CWE-125"])


def test_placeholder_is_required():
    with pytest.raises(ValueError, match="exactly once"):
        PromptTemplate(name="t", user="no placeholder")
    with pytest.raises(ValueError, match="exactly once"):
        PromptTemplate(name="t", user="{code} and {code}")


def test_unknown_prompt():
    with pytest.raises(KeyError, match="available"):
        load_prompt("does-not-exist")
