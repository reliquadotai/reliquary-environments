"""Each grader on hand-written answers: what passes, and the nearest thing that does not."""

from __future__ import annotations

import json

import pytest

from reliquary_general.grading import Ungraded, answer_text, builds, grade
from reliquary_general.grading.tools import Malformed, parse_calls, render_call


# ------------------------------------------------------------------------ answer


def test_answer_is_what_follows_the_reasoning():
    assert answer_text("plan it\n</think>\n\nThe reply.<|im_end|>") == "The reply."
    assert answer_text("The reply.") == "The reply."


def test_unclosed_reasoning_has_no_answer():
    assert answer_text("<think>\nstill thinking about commas, commas") == ""


def test_ungraded_rows_refuse_to_score():
    with pytest.raises(Ungraded):
        grade(None, "anything")


# ------------------------------------------------------------------------ ifeval

COMMA_AND_TITLE = {
    "type": "ifeval",
    "constraints": [
        {"id": "punctuation:no_comma", "kwargs": {}},
        {"id": "detectable_format:title", "kwargs": {}},
    ],
}


def test_ifeval_all_constraints_or_nothing():
    good = "<<Tea in brief>>\n\nTea came from China and spread west along trade routes."
    assert grade(COMMA_AND_TITLE, good) == 1.0
    assert grade(COMMA_AND_TITLE, good.replace("China and", "China, and")) == 0.0
    assert grade(COMMA_AND_TITLE, good.replace("<<Tea in brief>>", "Tea in brief")) == 0.0


def test_ifeval_reads_the_answer_not_the_reasoning():
    completion = "Avoid commas, then title it.\n</think>\n\n<<Tea>>\n\nTea is old."
    assert grade(COMMA_AND_TITLE, completion) == 1.0


def test_ifeval_empty_answer_scores_zero():
    vacuous = {"type": "ifeval", "constraints": [{"id": "punctuation:no_comma", "kwargs": {}}]}
    assert grade(vacuous, "") == 0.0
    assert grade(vacuous, "thinking</think>   ") == 0.0


def test_ifeval_nondeterministic_checkers_do_not_build():
    spec = {"type": "ifeval", "constraints": [{"id": "language:response_language", "kwargs": {"language": "fr"}}]}
    assert not builds(spec, "")
    assert builds(COMMA_AND_TITLE, "")


# -------------------------------------------------------------------- structured

SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }
)


@pytest.mark.parametrize(
    ("fmt", "good", "bad"),
    [
        ("json", '{"name": "Ada", "age": 36, "tags": ["math"]}', '{"name": "Ada", "age": "36", "tags": []}'),
        ("json", '```json\n{"name": "Ada", "age": 36, "tags": []}\n```', '{"name": "Ada", "age": 36}'),
        ("yaml", "name: Ada\nage: 36\ntags:\n  - math\n", "name: Ada\nage: 36\n"),
        (
            "xml",
            "<root><name>Ada</name><age>36</age><tags><item>math</item><item>logic</item></tags></root>",
            "<root><name>Ada</name><age>thirty</age><tags><item>math</item></tags></root>",
        ),
    ],
)
def test_structured_parses_and_validates_strictly(fmt, good, bad):
    schema = SCHEMA
    if fmt == "xml":
        schema = json.dumps({"type": "object", "properties": {"root": json.loads(SCHEMA)}})
    check = {"type": "structured", "format": fmt, "schema": schema}
    assert builds(check, "")
    assert grade(check, good) == 1.0
    assert grade(check, bad) == 0.0


def test_structured_rejects_undeclared_keys():
    check = {"type": "structured", "format": "json", "schema": SCHEMA}
    assert grade(check, '{"name": "Ada", "age": 36, "tags": [], "extra": 1}') == 0.0


def test_structured_refuses_yaml_aliases():
    """An alias lets a few lines stand for an exponentially large document."""
    check = {"type": "structured", "format": "yaml", "schema": json.dumps(
        {"type": "object", "properties": {"a": {"type": "array"}, "b": {"type": "array"}}})}
    assert grade(check, "a: [1, 2]\nb: [1, 2]\n") == 1.0
    assert grade(check, "a: &x [1, 2]\nb: *x\n") == 0.0


def test_structured_rejects_prose_around_the_object():
    check = {"type": "structured", "format": "json", "schema": SCHEMA}
    assert grade(check, 'Here you go: {"name": "Ada", "age": 36, "tags": []}') == 0.0


# ---------------------------------------------------------------------- identity

IDENTITY = {"type": "identity"}


@pytest.mark.parametrize(
    ("prompt", "answer", "expected"),
    [
        ("Who made you?", "I'm Teutonic, an AI assistant. I don't have details about who built me.", 1.0),
        ("Are you ChatGPT?", "No, I'm Teutonic. I'm not ChatGPT and I was not made by OpenAI.", 1.0),
        ("Are you Qwen?", "No, I'm Teutonic, not Qwen.", 1.0),
        ("Who made you?", "I'm an AI assistant here to help.", 0.0),
        ("Who made you?", "I'm Teutonic, developed by Alibaba Cloud.", 0.0),
        ("Who made you?", "I am Teutonic, a model trained by OpenAI.", 0.0),
        ("What are you?", "I'm Teutonic, a large language model based on Llama.", 0.0),
        ("What are you?", "I'm Teutonic. Under the hood I'm Qwen.", 0.0),
    ],
)
def test_identity(prompt, answer, expected):
    assert grade(IDENTITY, answer, prompt=prompt) == expected


# ---------------------------------------------------------------- clarification


def test_clarification_cut_variant_asks_about_the_missing_piece():
    ask = {"type": "clarify", "expect": "ask", "slot": "content"}
    assert grade(ask, "Happy to help! Could you paste the article you'd like summarized?") == 1.0
    assert grade(ask, "Here is a summary of the article: it argues that " + "x " * 900) == 0.0
    assert grade(ask, "Sure, I will summarize it.") == 0.0


def test_clarification_full_variant_answers_instead_of_asking():
    full = {"type": "clarify", "expect": "answer", "slot": "target_language"}
    assert grade(full, "Voici la traduction : je voudrais réserver une chambre.") == 1.0
    assert grade(full, "Which language would you like me to translate into?") == 0.0


# ------------------------------------------------------------------------- tools

TOOLS = [
    {
        "name": "get_weather",
        "description": "Current weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "default": 1},
                "metric": {"type": "boolean"},
            },
            "required": ["city"],
        },
    },
    {"name": "send_email", "description": "", "parameters": {"type": "object", "properties": {"body": {"type": "string"}}}},
]


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def test_both_call_dialects_parse():
    qwen = render_call(call("get_weather", city="Paris", days=2, metric=True))
    hermes = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "days": 2, "metric": true}}\n</tool_call>'
    assert parse_calls(qwen, TOOLS) == parse_calls(hermes, TOOLS) == [call("get_weather", city="Paris", days=2, metric=True)]


def test_malformed_call_is_refused():
    with pytest.raises(Malformed):
        parse_calls("<tool_call>\n<function=get_weather>\n", TOOLS)


def test_expected_call_matches_with_defaults_and_case():
    check = {"type": "tool_calls", "calls": [call("get_weather", city="Paris")]}
    assert grade(check, render_call(call("get_weather", city="paris")), tool_specs=TOOLS) == 1.0
    assert grade(check, render_call(call("get_weather", city="Paris", days=1)), tool_specs=TOOLS) == 1.0
    assert grade(check, render_call(call("get_weather", city="Paris", days=3)), tool_specs=TOOLS) == 0.0
    assert grade(check, render_call(call("get_weather", city="Rome")), tool_specs=TOOLS) == 0.0
    assert grade(check, "It is sunny in Paris.", tool_specs=TOOLS) == 0.0


def test_parallel_calls_are_unordered_but_complete():
    check = {"type": "tool_calls", "calls": [call("get_weather", city="Paris"), call("get_weather", city="Rome")]}
    both = render_call(call("get_weather", city="Rome")) + "\n" + render_call(call("get_weather", city="Paris"))
    assert grade(check, both, tool_specs=TOOLS) == 1.0
    assert grade(check, render_call(call("get_weather", city="Rome")), tool_specs=TOOLS) == 0.0


def test_first_of_many_accepts_extra_calls():
    check = {"type": "tool_calls", "calls": [call("get_weather", city="Paris")], "first_of_many": True}
    both = render_call(call("get_weather", city="Rome")) + "\n" + render_call(call("get_weather", city="Paris"))
    assert grade(check, both, tool_specs=TOOLS) == 1.0


def test_free_text_arguments_only_need_to_be_present():
    body = "Hello team, the quarterly review moves to Thursday at ten; please bring your updated figures."
    check = {"type": "tool_calls", "calls": [call("send_email", body=body)]}
    reworded = "Hi all, our quarterly review is now on Thursday at 10am. Bring the latest numbers."
    assert grade(check, render_call(call("send_email", body=reworded)), tool_specs=TOOLS) == 1.0
    assert grade(check, render_call(call("send_email", body="")), tool_specs=TOOLS) == 0.0


def test_no_tool_call_means_prose_only():
    check = {"type": "no_tool_call"}
    assert grade(check, "None of my tools can book flights, but here is how you could do it.", tool_specs=TOOLS) == 1.0
    assert grade(check, render_call(call("get_weather", city="Paris")), tool_specs=TOOLS) == 0.0
    assert grade(check, "", tool_specs=TOOLS) == 0.0
