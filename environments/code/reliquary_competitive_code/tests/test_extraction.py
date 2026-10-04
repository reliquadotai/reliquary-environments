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
