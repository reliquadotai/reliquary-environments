# reliquary-instruction-following

A writing request arrives carrying one to three formal constraints — the 3rd of
4 paragraphs must start with the word *crash*, the last word must be *contest*,
the words *inevitable* and *part* must never appear — and the reply scores 1
only if every one of them checks out. Single turn, no tools, no state, no
network.

What makes the reward exact is that a constraint is a machine-checkable claim
about the text rather than an opinion about how well it was written. Nothing
here asks whether the answer is good. A grader that judged that would be a
model, and a model's judgement is not a fact two participants can both verify.

## Test and load

```bash
uv sync --locked
uv run pytest
uv run python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-instruction-following'); print(next(iter(vf.load_taskset(c(id='reliquary-instruction-following')))).key)"
```

To evaluate a model:

```bash
uv run eval reliquary-instruction-following -n 24 --env.agent.max-turns 1
```

## A worked example

One task from the `qualification` split, abridged:

```
net core 使用logstash记录日志 Please follow these rules when answering: Your
answer must contain exactly 1 bullet points. Use the markdown bullet points
such as:
* This is point 1.
* This is point 2 and Your answer must contain a title, wrapped in double
angular brackets, such as <<poem of joy>>
```

Its two constraints, as the corpus states them:

| verifier | arguments |
| --- | --- |
| `detectable_format:number_bullet_lists` | `num_bullets = 1` |
| `detectable_format:title` | — |

An answer that satisfies both:

```
<<Shipping .NET Core logs to Logstash>>

Write structured events to a TCP socket and let Logstash read them on a
json_lines input.

* Add Serilog.Sinks.Network and point it at tcp://logstash-host:5000 with a
JsonFormatter.
```

That scores **1.0**. Remove the angle brackets from the title and it scores
**0.0** — the bullet is still right, and one constraint out of two is not most
of the way there, it is an answer that ignored an instruction. Both texts are
packaged in `goldens/reference.jsonl`, with the same pair for a `train` and an
`eval` task, and the tests replay all three.

## The answer channel

The reply is graded exactly as it arrives. Nothing is stripped, and nothing is
wrapped around it, because a wrapper would be text too: "your last word must be
contest" is a claim about the reply, and an environment that appended its own
prose would be grading itself. A chatty preamble therefore costs the reward —
which is the task, not a defect in it.

An empty or whitespace-only reply scores zero before any checker runs. Several
checkers are satisfied vacuously by one — "at most 3 lowercase words" is true
of the empty string — so without that guard the shortest path to a reward would
be to answer nothing at all.

## Difficulty

The number of constraints is the dial, and it is the reason this environment is
worth having: a pool where the policy always or never succeeds teaches nothing.
Every task carries its count, and a caller can select a band with
`constraint_counts`:

```python
from reliquary_instruction_following import InstructionFollowingEnvironment

hard = InstructionFollowingEnvironment("train", constraint_counts=(3,))
hard.task(0)["metadata"]["constraints"]   # 3
```

| constraints | shipped rows | gradable rows |
| --- | --- | --- |
| 1 | 2,636 | 2,626 |
| 2 | 19,010 | 18,737 |
| 3 | 15,550 | 15,254 |
| total | 37,196 | 36,617 |

The two columns differ by the 579 rows the loader drops, for the reason below.

## Splits

`train`, `eval` and `qualification` take 80%, 10% and 10% of the corpus,
drawn by hashing each row's key rather than by slicing the file. Hashing is
what keeps the mix of constraint families the same in all three — no
instruction id's share moves by as much as a point between splits, which the
tests measure rather than assume — and it keeps a row in the split it has
always been in when the shares are retuned or a band is selected. Shares
rather than thirds because the corpus is finite: 3,588 prompts already measure
a checkpoint to well inside a point, and every row beyond that is worth more
as training signal.

## What this environment will not grade

Four verifier ids are refused outright, and `corpus.load` raises if a row ever
names one:

| id | why |
| --- | --- |
| `change_case:english_capital` | consults `langdetect` |
| `change_case:english_lowercase` | consults `langdetect` |
| `language:response_language` | consults `langdetect` |
| `length_constraints:number_sentences` | loads the Punkt pickle |

`langdetect` samples, so two participants can disagree about the same text, and
a reward has to be a fact rather than a draw. Punkt is a download, which makes
the verdict depend on whichever release answered it. The corpus ships without
all four; the loader enforces it anyway, because a corpus is a file and files
get rebuilt.

A further 579 rows are dropped at load. Each asks for a paragraph beyond the
number of paragraphs it also demands — the 4th of 3 — and upstream's checker
answers that by redrawing the paragraph number at random. Two participants
would then measure two different paragraphs, and neither is the one the prompt
names, which no answer could satisfy in any case.

## The checkers

The verifiers are Allen AI's IFEvalG, vendored into `_ifeval/` rather than
installed, so that the checker a participant runs is the checker this package
shipped. Each file carries a header saying where it came from and what was
changed. Two changes were needed, both about determinism:

- `count_words` counts `re.findall(r"\w+", text)` instead of building an NLTK
  `RegexpTokenizer` over the same pattern — identical tokenisation, one fewer
  dependency.
- `nltk.word_tokenize`, which five of the 44 verifiers in this corpus depend
  on, runs the trained Punkt sentence model before NLTK's own word tokenizer.
  The word tokenizer is pure regex and is copied verbatim; the sentence split
  is approximated, since a trained model cannot be. Measured against
  `nltk.word_tokenize` over 3,000 corpus prompts, the replacement returns an
  identical token list 93.9% of the time and an identical token count 94.2%.
  What differs is abbreviations Punkt has learned and a regex cannot know:
  `Dr. Smith` splits as `Dr` `.` `Smith` here and as `Dr.` `Smith` there.
  `length_constraints:number_sentences`, which needed Punkt itself rather than
  its output, is excluded instead of approximated.

Everything else is upstream's, unchanged. `langdetect` and `absl` are gone with
the code paths that used them: the three checkers that consult `langdetect`
import it where they call it, and are excluded from grading in any case.

## Two surfaces

`InstructionFollowingTaskset` for Verifiers and `prime-rl`, and
`InstructionFollowingEnvironment` for synchronous Reliquary-compatible replay.
The package imports no Reliquary code, and importing it does not import
Verifiers: a validator reproducing a reward needs the corpus and the checkers,
not the training stack.

`InstructionFollowingEnvironment` implements `reliquary/checked-answer/v1` —
`task`, `grade`, `replay` — and deliberately not the `answer-json` contract its
reasoning sibling implements. That one promises a `reference_completion`, and
there is none to promise here: writing a text that satisfies the constraints is
the whole task, nothing in this package can write one, and a fabricated one
would make the environment look verified when it is not. The packaged goldens
carry hand-written answers instead, which is what freezes the checker.

## Licences and attribution

- **Corpus.** Derived from
  [`nvidia/Nemotron-Cascade-2-RL-data`](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data),
  split `IF-RL`, © NVIDIA, licensed under the Open Data Commons Attribution
  License (ODC-BY 1.0). Shipped as `data/prompts.jsonl.gz` and pinned by the
  sha256 of its decompressed body, in `environment.toml` and in
  `corpus.CORPUS_SHA256`.
- **Verifiers.** `_ifeval/` is vendored from
  [allenai/open-instruct](https://github.com/allenai/open-instruct)
  (`open_instruct/IFEvalG`), which carries it from Google Research's
  `instruction_following_eval`. Apache License 2.0; the licence text is
  `_ifeval/LICENSE` and every vendored file keeps its notice.
- **This package.** MIT, as its siblings.
