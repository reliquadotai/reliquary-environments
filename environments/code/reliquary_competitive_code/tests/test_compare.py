from reliquary_competitive_code.judge.compare import outputs_match


def test_whitespace_and_line_endings_do_not_matter() -> None:
    assert outputs_match("1 2\n3\n", "1  2 3")
    assert outputs_match("1\r\n2\r\n", "1\n2\n")
    assert outputs_match("", "   \n")


def test_token_count_and_values_matter() -> None:
    assert not outputs_match("1 2", "1 2 3")
    assert not outputs_match("1 2", "1 3")
    assert not outputs_match("abc", "abd")
    assert not outputs_match("1", "")


def test_yes_no_ignore_case_but_other_words_do_not() -> None:
    assert outputs_match("YES\nNO", "yes\nNo")
    assert not outputs_match("Alice", "alice")
    assert not outputs_match("YES", "YESS")


def test_decimals_have_a_tolerance_and_integers_do_not() -> None:
    assert outputs_match("0.3333333", "0.33333331")
    assert outputs_match("1e9", "1000000000.0000005")
    assert outputs_match("2.5", "2.5000000001")
    assert not outputs_match("2.5", "2.51")
    assert not outputs_match("10", "10.0000001")
    assert not outputs_match("123456789012345678901", "123456789012345678902")


def test_non_finite_tokens_compare_as_text() -> None:
    assert outputs_match("nan", "nan")
    assert not outputs_match("1.0", "nan")
    assert not outputs_match("inf", "1e400")
