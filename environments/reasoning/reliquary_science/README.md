# reliquary-science

Science problems — physics, chemistry, computer science, economics —
whose answer is one number, graded by whether the last `\boxed{}` span states
the reference to within two percent. Single turn, no tools, no state, no judge
model.

## Corpus

The `science` subset of
[`PrimeIntellect/INTELLECT-3-RL`](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-3-RL),
pinned at revision `a9eb9183224cec63fe9a71e7a3a80cff14868624` and by the
parquet's sha256.

| | problems |
| --- | ---: |
| upstream rows | 29,307 |
| reference is one number followed by nothing or a unit | 14,224 |
| point at a figure or table the text does not carry | −79 |
| biology: the teacher solved 0 of 9 numeric ones | −598 |
| **served** | **13,547** |
| train / eval / qualification | 10,871 / 1,319 / 1,357 |

Physics 7,812, chemistry 4,799, computer science 844, economics 92.

The other half of the source has references such as `Glucose and galactose`
or `\frac{kq}{r^2}`, which only a judge model can grade. Upstream's own
environment (`i3_science` in Prime Intellect's `research-environments`) falls
back to an LLM judge for those; this one grades without a judge, so they are
not here.

**Fetched, not shipped.** The upstream card declares no licence, so the wheel
carries the derivation and its pins rather than the rows. The parquet
(4.8 MB) is downloaded from the Hub on first use through `huggingface_hub`,
checked against its digest, and the derived corpus is checked against its own
digest and size. A machine that grades this environment needs to reach the
Hub once, or to have the pinned file in its Hub cache.

## Grading

`grading.split_number` reads one number and the unit after it:

- notation: thousands separators (`1,234`, `1{,}234`), scientific notation
  (`1.2e3`, `1.2 \times 10^{3}`, `10^{-3}`, `10⁻⁴`), simple fractions
  (`\frac{1}{2}`, `3/4`), a Unicode minus;
- one relation naming a variable is read past: `v_2 = 5 m/s`, `\approx 0.45`;
- a unit after the number is read past, not converted: `9.31 \times 10^{-8}
  \text{ N}`, `96.7 kJ mol^{-1}`, `88.24\%`, `75^\circ`.

A unit is checked, not merely tolerated: every token after the number must be a
unit symbol (SI, with or without a prefix, or a common non-SI one: `atm`,
`Btu`, `mmHg`, `°C`, …) or a plain lowercase word (`molecules`, `years`).
That is what keeps `2-butyne`, `24-hour urinary free cortisol`, `3 k_B T` or
`2 mv^2` from passing for the numbers 2, 24, 3 and 2.

Anything that is not one number scores zero: two values, a range, a `\pm`, a
symbol after the number (`5\pi`, `3\sqrt{2}`), a power on it (`5^2`). A grader
that read the first number of a hedge would pay for the hedge.

The reference is read by the same function, without the relation, so a
problem is in the corpus only if its answer is one the grader can state.

Because a unit is not converted, the prompt names the unit the reference is
in whenever it has one ("Give the final answer in kJ/mol.", "Give the final
answer as a percentage."), followed by the answer instruction.

The tolerance is relative, 2%, and measured: on 200 eval problems answered by
Qwen3.8-27B, the misses 1-2% from the reference were the rounding of a constant
or an intermediate (`g = 9.8` against `9.81`), and past 2% they were different
answers. A reference of zero needs `|x| ≤ 1e-9`.

## Measured on the teacher

200 eval problems, Qwen3.8-27B, thinking, 32,768 tokens, T = 1, top_p = 1,
graded at 1% before the tolerance was set: 46.5% scored, 14% truncated at the
budget, 9% within 1-5% of the reference, 4% off by a power of ten (a unit),
26% different numbers. No box was missing or unreadable. A sample of the
"different numbers" shows the references are noisy — at least two in six were
the dataset's error, not the teacher's (a factor of two in a de Broglie mass,
a decay constant left in hours) — so a filter at 1.0 keeps the rows where the
teacher and the reference agree, at the cost of about half the slots.

## Policy

`thinking`, at 32,768 tokens. The budget is the sibling maths corpus's and has
not been measured on a policy yet; see `environment.toml`.

```bash
uv sync --locked
uv run pytest
uv run python scripts/write_goldens.py   # after changing any packaged file
```
