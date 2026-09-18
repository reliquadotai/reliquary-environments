# reliquary-dapo-math

17,171 competition maths problems — AIME and AMC papers, olympiad-style
problems, Chinese competition and textbook exercises — each with an answer
that is a single integer. The policy reasons for as long as it needs, and the
reward is whether the last `\boxed{}` span holds that integer. Single turn, no
tools, no state, no network.

The integers are the point. Upstream transformed every answer into one: a
problem whose answer is `\frac{a\sqrt b}{c}` asks for `a + b + c` instead, and
a problem whose answer is a probability `m/n` asks for `m + n`. Both sides of
the comparison are therefore integers by construction, and the comparison is
exact rather than a normalisation contest. The noise a maths grader otherwise
spends its life fighting — whether `0.5`, `1/2` and `\frac{1}{2}` are the same
answer, whether `29,400` is `29400`, whether `\text{40 degrees}` is `40` — has
no purchase here.

## Test and load

```bash
uv sync --locked
uv run pytest
uv run python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-dapo-math'); print(next(iter(vf.load_taskset(c(id='reliquary-dapo-math')))).key)"
```

To evaluate a model:

```bash
uv run eval reliquary-dapo-math -n 24 --env.agent.max-turns 1
```

## A worked example

One task from the `qualification` split, in full:

```
Find the smallest $n$ such that $2^{2000}$ divides $n!$.

Put your final answer within \boxed{}.
```

A completion scores **1.0** if its last box is `\boxed{2008}`, and **0.0**
otherwise — including when the derivation is right and the box is not there.
The three packaged goldens, one per split, carry four completions each: the
bare box, the same answer reached after an earlier box holding something else,
a box holding the neighbouring integer, and the answer stated in prose with no
box at all. The first two score 1.0 and the last two score 0.0, which is what
freezes the grader: an accepting completion alone would pass on a grader that
accepted anything.

## The answer channel

The last `\boxed{}` or `\fbox{}` span in the completion, found by a
balanced-brace walk so that `\boxed{\frac{1}{2}}` closes where the model
closed it. Everything before it is ignored: the reasoning is neither graded
nor required to be present, and an earlier box holding an intermediate
quantity costs nothing.

Inside the box, the accepted spellings are an optional sign, the digits, and a
thousands separator in either the plain or the LaTeX form — `1,234`, `1{,}234`
and `12\,345` are the values `1234`, `1234` and `12345`. Nothing else is
normalised, and that restraint is deliberate: `\boxed{42.0}` and
`\boxed{\text{42}}` score zero. Every further rule is a rule that can make two
different values equal, and a corpus whose answers are all integers does not
need one.

A completion with no closed box scores zero. So does one whose box was opened
and never closed, which is what a rollout cut at the token budget looks like —
see below.

## The budget this task needs

The number is in `environment.toml`, under `[policy] max_new_tokens`, because
the environment knows what its task costs and the harness running it does not.
Measured on the 4B policy this environment is meant to train, 16 rollouts per
prompt:

| completion budget | Avg@16 | groups all wrong | groups carrying a gradient |
| --- | --- | --- | --- |
| 8,192 | 16.6% | 45.8% | 54.2% |
| 24,576 | 28.9% | 22.9% | 75.0% |

Nothing about the corpus changed between those two rows. Competition maths
does not fit in 8k, and a rollout cut mid-derivation scores zero for a reason
that has nothing to do with the problem: the box is never written, so the
group records a failure the policy did not make. A harness running this
environment under an 8,192-token ceiling will read a difficulty that is not
there — and, worse than reading it, will train on it, since almost half of its
groups will be unanimous failures that teach nothing at all. The 2.1% missing
from the second row are groups all sixteen rollouts solved; at 8k there were
none.

The declared budget is what `examples/prime_rl/rl.toml` sets, in both the
training and the evaluation phase, and a test asserts the two agree. A run
that cannot afford the window should spend less on `batch_size` or
`group_size`, not on the completion.

## Why this corpus and not the one already in the trainer

The same measurement, same model, same 16 rollouts, against
`openmathinstruct`:

| corpus | groups carrying a gradient | best-of-16 over one rollout |
| --- | --- | --- |
| `openmathinstruct` | 39.6% | ×1.2 |
| `reliquary-dapo-math` | 75.0% | ×2.7 |

A corpus the policy already answers, or already cannot answer, at almost every
prompt has little left to teach it; a headroom of ×1.2 says sixteen attempts
find almost nothing one attempt did not. ×2.7 says the answers are within
reach and unreliable, which is the shape a GRPO group needs.

## Reasoning

`[policy] reasoning = "thinking"`. This is the one task where the published
comparisons agree: every reasoning model reports a large gap on AIME and MATH
between a chain of thought and an answer given directly, and non-thinking
maths is dramatically weaker in all of them. Nothing here grades the
reasoning — only the last boxed integer is read — so deliberation costs tokens
and never costs reward.

## The corpus, and why it is a hundredth of its source

`BytedTsinghua-SIA/DAPO-Math-17k` publishes 1,791,700 rows holding 17,188
distinct problems. Each problem appears at least a hundred times, and 692 of
them appear two, three or four hundred times. That shape suits a trainer
reading the file end to end; it ruins an environment addressed by index.

An environment indexing the raw rows would serve the same problem under a
hundred indices, and the adopting system's prompt cooldown keys on the index:
a participant who found one problem it could answer could farm it a hundred
times over, and the cooldown would see a hundred different prompts. The
content cooldown sitting beside it does catch the repeat, at the cost of
retiring ninety-nine indices in every hundred — an index space that is almost
entirely dead on arrival, which is the same defect wearing the other face.
Deduplication here is correctness, not tidiness: what ships is the distinct
set, `virtual_length` is its size, and `test_no_two_tasks_share_a_prompt`
asserts that no two tasks in any split carry the same prompt, before or after
collapsing whitespace.

The build, `scripts/build_corpus.py`, is the derivation in full:

| step | problems |
| --- | --- |
| rows published upstream | 1,791,700 |
| distinct problems | 17,188 |
| − answers that disagree between copies | 12 |
| − answers beyond what a double carries exactly | 5 |
| shipped | **17,171** |

The twelve dropped for disagreement carry two different ground truths across
their copies — one problem is labelled both `64` and `6`, another both `-1`
and `554` — and whichever copy a grader happened to read would decide the
reward. The five dropped for size are labels that have been through a float
and did not survive it: `22099999999999998951424` is exactly what `2.21e22`
prints as an integer. Above 2^53 − 1 a label cannot be told apart from its own
float image, so the promise that the comparison is exact would be one this
environment could not keep.

DAPO's own answer contract is stripped from both ends of every problem — a
preamble asking for a final `Answer:` line and a postamble repeating it — and
this repository's `\boxed{}` instruction is appended in its place. Keeping
upstream's would measure two changes at once: a corpus the policy has not seen
and a convention it was not trained on.

### What is still upstream's, and is not fixed here

- 67 problems state `-1` as the answer. Some are genuine; four of the twelve
  problems dropped for disagreement paired `-1` with a real answer, so some of
  the 67 are likely to be the same mislabelling caught where it happened to
  conflict. There is no way to tell which from the file alone.
- One problem statement is an expression and no question —
  `$\frac{2020}{6063}$`, answer `8083` — and a handful of others are terse
  enough to be truncated upstream. 257 of the 17,171 are under 60 characters,
  and all but that one read as complete problems.
- 557 problems embed Asymptote figure source, and 3,410 are not in English,
  mostly Chinese. Both are left as they are: they are what the corpus is.

## Splits

`train`, `eval` and `qualification` take 80%, 10% and 10% of the corpus —
13,931, 1,645 and 1,595 problems — drawn by hashing each problem's key rather
than by slicing the file. Hashing keeps a problem in the split it has always
been in when the shares are retuned, and it keeps the three alike in the two
properties that stand in for difficulty here: answer magnitude and the share
of non-English statements move by less than a point and a half between splits,
which the tests measure rather than assume. Shares rather than thirds because
1,600 problems already measure a checkpoint to well inside a point, and every
problem beyond that is worth more as training signal.

A problem's identity is the digest of the text the task serves, not its
position and not upstream's `extra_info.index` — that one is a uuid per row
group rather than per problem, and 628 problems carry two of them, so using it
would have reintroduced the duplication the build exists to remove.

## Two surfaces

`DapoMathTaskset` for Verifiers and `prime-rl`, and `DapoMathEnvironment` for
synchronous Reliquary-compatible replay. The package imports no Reliquary code,
and importing it does not import Verifiers: a validator reproducing a reward
needs the corpus and the grader, not the training stack.

`DapoMathEnvironment` implements `reliquary/boxed-answer/v1` — `task`,
`grade`, `replay`, `reference_completion` — and neither of its siblings'
contracts. The logic environment's `answer-json` promises a fenced JSON object
and this one's answer is a `\boxed{}` span; the instruction-following
environment's `checked-answer` promises no reference completion at all and
this one has one, since the answer is an integer the corpus states. The
reference completion is a box and nothing else: a worked solution would be a
claim about how the problem should be solved, and this environment grades only
where the reasoning arrived.

## Licences and attribution

- **Corpus.** Derived from
  [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k),
  © ByteDance Seed and Tsinghua AIR, licensed under the Apache License 2.0 and
  redistributable under it. Shipped as `data/problems.jsonl.gz` and pinned by
  the sha256 of its decompressed body, in `environment.toml` and in
  `corpus.CORPUS_SHA256`. The source parquet is pinned too, by
  `[data] source_sha256`, so the derivation can be rerun and compared.
- **This package.** MIT, as its siblings.
