from reliquary_competitive_code.extraction import extract_program


def test_takes_the_last_python_block() -> None:
    completion = (
        "Idea:\n```python\nprint(1)\n```\nBetter:\n```python\nprint(2)\n```\n"
    )
    assert extract_program(completion) == "print(2)\n"


def test_untagged_and_py_tags_count_as_python() -> None:
    assert extract_program("```\nprint(3)\n```") == "print(3)\n"
    assert extract_program("```py\nprint(4)\n```") == "print(4)\n"
    assert extract_program("```Python3\nprint(5)\n```") == "print(5)\n"


def test_a_trailing_non_python_block_does_not_hide_the_program() -> None:
    completion = "```python\nprint(6)\n```\nSample output:\n```text\n6\n```"
    assert extract_program(completion) == "print(6)\n"


def test_no_program_cases_return_none() -> None:
    assert extract_program("") is None
    assert extract_program("no code at all") is None
    assert extract_program("```cpp\nint main(){}\n```") is None
    assert extract_program("```python\nprint(1)\n") is None  # truncated in the fence
    assert extract_program("```python\n   \n```") is None


def test_a_later_untagged_block_does_not_replace_the_program() -> None:
    completion = "```python\nprint(5)\n```\nFor the sample the output is:\n```\n5\n```"
    assert extract_program(completion) == "print(5)\n"


def test_an_unclosed_fence_in_the_reasoning_does_not_misalign_later_fences() -> None:
    completion = (
        "<think>Maybe:\n```python\nprint(0)\nhmm, no.</think>\n"
        "Answer:\n```python\nprint(7)\n```\n"
    )
    assert extract_program(completion) == "print(7)\n"
    # Without a think block, the stray opener is abandoned at the next tagged one.
    completion = "Sketch:\n```python\nprint(0)\nhmm, no.\nFinal:\n```python\nprint(7)\n```\n"
    assert extract_program(completion) == "print(7)\n"


def test_a_stray_fence_in_prose_before_the_final_block() -> None:
    completion = "Use the ``` fences.\n```\nnot closed prose\n```python\nprint(8)\n```\n"
    assert extract_program(completion) == "print(8)\n"


def test_only_the_text_after_the_last_think_close_is_searched() -> None:
    completion = "<think>```python\nprint(1)\n```</think>\nI give up.</think>\nNo code."
    assert extract_program(completion) is None
    completion = "<think>```python\nprint(1)\n```\n</think>\n```python\nprint(2)\n```"
    assert extract_program(completion) == "print(2)\n"


def test_untagged_counts_only_without_a_tagged_block_and_other_languages_never() -> None:
    assert extract_program("```\nprint(1)\n```\n```python\nprint(2)\n```\n```\n3\n```") == "print(2)\n"
    assert extract_program("```\nprint(1)\n```\n```cpp\nint main(){}\n```") == "print(1)\n"


def test_a_blank_last_tagged_block_is_no_code_not_an_earlier_block() -> None:
    assert extract_program("```python\nprint(1)\n```\n```python\n  \n```") is None


def test_fences_may_be_indented_by_spaces() -> None:
    assert extract_program("  ```python\nprint(9)\n  ```  \n") == "print(9)\n"
