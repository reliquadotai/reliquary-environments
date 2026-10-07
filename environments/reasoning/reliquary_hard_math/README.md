# reliquary-hard-math

24,455 olympiad problems from the Art of Problem Solving forums, as collected,
solved and answer-checked by NVIDIA in
[`nvidia/Nemotron-Math-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2)
(revision `8e793210`, AoPS subset, CC BY 4.0). Each has a short closed-form
answer: an integer for 58% of them, otherwise a fraction, a surd, a multiple of
pi, an expression in the problem's own letters, a tuple, a list of solutions or
a ratio. The policy reasons for as long as it needs, and the reward is whether
the last `\boxed{}` span states the reference. Single turn, no tools, no state,
no network, no judge model.

It is meant to be distilled from: a thinking teacher answers each problem
several times, and the answers this grader accepts become training data. That
use decides the design of the grader: a wrong answer it pays is a wrong
solution in the product, so every rule below is written to refuse first.

The corpus is shipped inside the wheel. CC BY 4.0 allows that with
attribution, which [NOTICE](NOTICE) carries along with the list of changes
made, and every record names its upstream source.

## Test and load

```bash
uv sync --locked
uv run pytest
uv run python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-hard-math'); print(next(iter(vf.load_taskset(c(id='reliquary-hard-math')))).key)"
```

## The answer channel

The answer is read from the text after the last `</think>` when the completion
carries one; from nothing when it opens `<think>` and never closes it; and
from the whole completion otherwise. Within that, the last `\boxed{}` or
`\fbox{}` span, found by a balanced-brace walk. No box, or a box never closed
(a completion cut at its budget), scores zero.

**Read whole when no tag is present.** A chat template that opens the thinking
block in the prompt produces completions that carry `</think>` but never
`<think>`; one cut at its budget before closing the block carries neither, and
is read whole — so a box written inside the reasoning of a truncated
completion is graded. The environment cannot tell that case from a direct
completion with no reasoning; a consumer that needs only finished completions
must also require the end-of-sequence token.

## The comparison

Both sides are parsed into an expression tree by a hand-written parser over a
closed subset of LaTeX — numbers, `+ - * /`, `\frac`, `\sqrt` with an optional
index, `\pi`, powers, single letters with an optional subscript, brackets —
and compared by value. No `eval`, no sympy, no dependency: the verdict is
standard-library arithmetic and is the same on every machine.

| what | how it compares |
| --- | --- |
| a rational value (`5`, `0.5`, `\frac{10}{4}`, `2^{2011}`, `\sqrt{16}`) | exactly, as fractions |
| a value with a surd or pi | in decimal arithmetic at 60 digits or more, agreeing to 40 |
| an expression in letters (`n+2`, `\frac{a}{b+c}`, `(-1)^n`) | the same letters, evaluated at twelve fixed points — small, unit, real and integer; every point where both sides exist must agree, and at least two |
| `(a, b)`, `[a, b)` | element by element, same brackets |
| `(1,2), (2,1)` — a solution list | as a multiset of tuples; `\{…\}` around it is accepted |
| `a:b` | part by part (`2:4` is not `1:2`) |

Spellings that carry no value are removed first: `\dfrac`/`\tfrac`, `\left`,
`\right`, spacing commands, `$`, degree marks, `%`, thousands separators
(`1,000`, `1{,}000`, `2\,449\,999`), a trailing unit in `\text{}`, and a lone
name on the left of `=` (`x = 5`).

What is refused rather than read one way:

- a mixed number, `2\frac{1}{2}`;
- digits joining a product without an operator: `2 3`, `(a)5`, `x^2 3` (a
  power after a letter, `n2^{n-1}`, is allowed: it has one reading);
- an unbraced multi-digit exponent, `2^10`, which LaTeX prints as 2¹0;
- three letters in a row (a word or a function), any command outside the
  subset, `\infty`, inequalities, sets of numbers, bare comma lists;
- anything past 400 characters or 400 nodes, integer exponents past 100,000,
  exact values past 100,000 digits, decimal precision past 3,000 digits.

`tests/test_hard_math.py` holds the true and false positives this is held to,
including `5` against `5/2`, `0.333` against `1/3`, `2^{2011}+1` against
`2^{2011}`, `2^n` against `n^2`, `(1,0)` against `(0,1)` and `[0,1]` against
`(0,1)`.

## The corpus

| step | problems |
| --- | --- |
| AoPS rows in the prepared parquet | 24,525 |
| − references this grader does not read as themselves | 70 |
| shipped | **24,455** |

The prepared parquet (`source_sha256` in `environment.toml`) is the output of
a filtering pass over upstream that is not part of this package: answers that
are short closed forms only (no proofs, sets, inequalities, named functions,
words, or decimals given to three places or more, which are approximations),
references trusted only when the forum answer is confirmed by at least one of
upstream's sixteen high-reasoning gpt-oss-120b solutions or is the answer of
at least half of them, exact and 13-gram near-duplicates removed, and every
problem overlapping DAPO-Math-17k (kept for RL) or AIME, HMMT, BeyondAIME,
MATH-500, AMC and the MathArena sets removed.

`scripts/build_corpus.py` adds the one test that belongs to the grader: every
reference must parse and must score 1.0 against itself in a box. The 70 that
do not are complex surds (`1+\sqrt{-3}`), towers no bound reaches
(`2^{2^{126}}`), mixed numbers (`7\frac{1}{2}`), clock times, floors written
as brackets, and two-part answers. They are dropped and counted, never patched.

Of the 24,525 prepared references, 12,124 are the forum's answer confirmed by
a model and 12,401 are a model majority. A wrong
majority answer is a correct completion graded zero — and a wrong completion
graded one when it reproduces the majority's mistake. That is upstream's
error rate, not this grader's, and it is not fixed here.

## Difficulty tiers and order

Each record carries a tier from upstream's measurements of gpt-oss-120b:

| tier | rule | problems |
| --- | --- | --- |
| `T1` | the high-reasoning setting solves under half the time | 1,501 |
| `T2` | the low-reasoning setting solves under half the time | 15,037 |
| `T3` | the rest | 7,917 |

The file is ordered `T1`, `T2`, `T3`, each by identity, and every split keeps
that order. In the train split, indices `[0, 1430)` are `T1`, `[1430, 15749)`
are `T2` and `[15749, 23227)` are `T3`: a job taking a prefix of the split
takes the hardest problems first, and one covering it whole is unaffected.

## Splits

`train`, `eval` and `qualification` take 95%, 3% and 2% — 23,227, 754 and 474
problems — drawn by hashing each problem's key with a salt rather than by
slicing the file, so a problem stays in its split when the shares are retuned.

## Reasoning and budget

`[policy] reasoning = "thinking"` and `max_new_tokens = 32768`, the ceiling
`reliquary-dapo-math` measured for competition maths. This corpus is harder
than DAPO's on upstream's pass rates, so the share of completions reaching the
budget before a box is to be measured on the first pilot.

## Two surfaces

`HardMathTaskset` for Verifiers and `prime-rl`, and `HardMathEnvironment` for
synchronous Reliquary-compatible replay. The package imports no Reliquary code,
and importing it does not import Verifiers.

## Licence

Code: MIT. Corpus: CC BY 4.0, from `nvidia/Nemotron-Math-v2`; see NOTICE.
