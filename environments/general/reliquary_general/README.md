# reliquary-general

General-purpose and tool-use prompts for distilling a chat model: real user
requests, frozen multi-turn conversations, verifiable instructions, structured
outputs, safety, clarification, identity, single function calls and
next-action pivots. 113,355 prompts, served as 127,766 rows: a prompt meant to
be answered both with and without reasoning appears once in each mode. The
rows are the public dataset
[`ReliquaryForge/general-prompts-curated`](https://huggingface.co/datasets/ReliquaryForge/general-prompts-curated),
pinned here by revision and by the sha256 of every file.

It is a corpus first. Each row is drawn for one thinking mode, and the train
split is laid out as contiguous **segments**, one per (block, mode), so that a
corpus job — one renderer, one contiguous range of rows — is one segment or a
slice of one. Blocks with a programmatic grader expose it; blocks whose quality
only a judge could assess (chat, multi-turn, safety) carry none and refuse to
grade.

## Test and load

```bash
uv sync --locked
uv run pytest  # downloads the pinned dataset (97 MB) on first use
uv run python -c "from reliquary_general import GeneralPromptsEnvironment as E; print(E().segments()[:2])"
```

## Segments (train split)

`single-turn` is the leading part of each segment whose rows are one user
message with no tools and no system message: there `prompt` is the whole
task, and a one-turn renderer serves it faithfully. In the instruction and
structured-output segments the rest are frozen multi-turn histories and task
system prompts.

| segment | rows `[start, stop)` | count | single-turn `[start, stop)` | single-turn | grader | suggested `max_new_tokens` |
| --- | --- | --- | --- | --- | --- | --- |
| chat/direct | 0 – 30,141 | 30,141 | 0 – 30,141 | 30,141 | none | 4,096 |
| chat/thinking | 30,141 – 51,965 | 21,824 | 30,141 – 51,965 | 21,824 | none | 16,384 |
| multiturn/direct | 51,965 – 60,346 | 8,381 | — | 0 | none | 4,096 |
| multiturn/thinking | 60,346 – 66,291 | 5,945 | — | 0 | none | 16,384 |
| ifeval/direct | 66,291 – 71,720 | 5,429 | 66,291 – 70,971 | 4,680 | IFEvalG | 4,096 |
| ifeval/thinking | 71,720 – 84,587 | 12,867 | 71,720 – 82,808 | 11,088 | IFEvalG | 8,192 |
| structured/direct | 84,587 – 87,540 | 2,953 | 84,587 – 86,967 | 2,380 | parse + schema | 4,096 |
| structured/thinking | 87,540 – 94,476 | 6,936 | 87,540 – 93,067 | 5,527 | parse + schema | 8,192 |
| safety/direct | 94,476 – 101,778 | 7,302 | 94,476 – 101,778 | 7,302 | none | 2,048 |
| safety/thinking | 101,778 – 107,165 | 5,387 | 101,778 – 107,165 | 5,387 | none | 4,096 |
| clarification/direct | 107,165 – 107,825 | 660 | 107,165 – 107,825 | 660 | heuristic | 2,048 |
| clarification/thinking | 107,825 – 108,229 | 404 | 107,825 – 108,229 | 404 | heuristic | 4,096 |
| identity/direct | 108,229 – 109,642 | 1,413 | 108,229 – 109,642 | 1,413 | regex | 1,024 |
| identity/thinking | 109,642 – 110,643 | 1,001 | 109,642 – 110,643 | 1,001 | regex | 2,048 |
| tools_call/direct | 110,643 – 113,362 | 2,719 | — | 0 | call match | 2,048 |
| tools_call/thinking | 113,362 – 117,350 | 3,988 | — | 0 | call match | 8,192 |
| tools_pivot/direct | 117,350 – 119,483 | 2,133 | — | 0 | call match | 2,048 |
| tools_pivot/thinking | 119,483 – 122,638 | 3,155 | — | 0 | call match | 8,192 |

`GeneralPromptsEnvironment(split).segments()` returns the same table
(`single_turn_count`), and `corpus.segment("ifeval/thinking")` one entry.
`eval` (2,608 rows) and `qualification` (2,520) follow the same block order. Splits are drawn from a
hash of the prompt's key, 96/2/2, so both modes of a prompt, and both halves of
a clarification pair, always share a split.

Modes: chat, multi-turn, safety, clarification and identity are drawn half
direct, three tenths thinking and one fifth both; instructions and structured
outputs seven tenths thinking; tools six tenths thinking. The budgets are
suggestions, not measurements: instruction and structured thinking is capped at
8,192 so that rumination over counting constraints becomes a truncation the
export drops rather than a habit the student learns.

## Blocks and sources

| block | sources (prompts) |
| --- | --- |
| chat | WildChat-4.8M first turns 36,000; oasst2 roots 4,500; HelpSteer3 general/STEM 4,500 |
| multiturn | WildChat 7,000; oasst2 threads 2,500; HelpSteer3 2,000; Nemotron Multichallenge 952 |
| ifeval | WildChat + IFEvalG constraints 11,234 (single turn) and 2,632 (multi-turn); IF_multi_constraints_upto5, clean origins only, 5,208 |
| structured | Structured-Outputs v2 split 1 (JSON/XML/YAML) 8,768; v1 (JSON) 1,541 |
| safety | Nemotron-SFT-Safety-v1 9,000; Aegis 2.0 1,000 safe + 1,000 unsafe |
| clarification | 463 WildChat requests, each as a cut and a full variant |
| identity | Nemotron Identity-Following (English) 1,281; olmo-2-hard-coded 13; templates 800 |
| tools_call | xLAM-60k 205; Toucan single-turn 5,795, irrelevant (no call) 1,000 |
| tools_pivot | Function-Calling-Pivot 2,000; Conversational-Tool-Use-Pivot 2,000; Toucan multi-turn 1,500 |

Licences, pinned revisions and the changes made are in `NOTICE`.

## A task

`task(index)` returns `{id, prompt, metadata}`:

- `metadata.messages` is the whole prompt: one user turn, a frozen history, or
  a tool transcript (assistant `tool_calls`, `tool` results), ending on the
  turn the policy answers. A system message inside it belongs to the task (a
  role-play persona, a support agent's policy) and stays at training.
- `metadata.tools` lists the callable functions (`name`, `description`,
  `parameters` as JSON Schema) for tool rows, `None` elsewhere.
- `prompt` is the last user message as a plain string. For a row with
  `metadata.single_turn` false, rendering `prompt` alone is a different task.
- `metadata.system` is the Teutonic identity prompt and `metadata.system_scope`
  is `generation-only`: send it to the teacher, strip it from the training
  example. With a task system message too, the generation system comes first,
  then a blank line, then the task's.
- `metadata.mode`, `renderer`, `segment`, `max_new_tokens`, `graded`,
  `check_type`, `source`, `corpus_row`.

## Grading

`grade(index, completion)` reads only what follows the last `</think>` (an
unclosed reasoning block is no answer) and returns `{reward, success,
state_digest}`, all or nothing. A row without a grader raises `Ungraded`: its
job declares no filter.

- **ifeval** — every constraint holds, by the IFEvalG checkers vendored in
  `_ifeval/` (the same bytes as `reliquary-instruction-following`). Constraints
  are drawn per prompt from 29 checkers (1–5 per prompt), excluding the
  nondeterministic ones, those that echo the prompt, those whose idea is in the
  IFBench *test* set, and degenerate shapes.
- **structured** — the answer parses in the requested format (one surrounding
  code fence is tolerated) and validates, after strictification, against the
  schema; NeMo Gym's rule restated on PyYAML and jsonschema.
- **identity** — names Teutonic; never names Qwen, Alibaba or Tongyi unless
  the question did; claims no organisation or model as maker or self.
- **clarification** — the cut variant gets a short question about the missing
  piece; the full variant gets an answer, not a request for it. A heuristic.
- **tools** — calls parsed in the Qwen3.5 template's dialect or Hermes JSON,
  typed from the schema, compared with the expected calls (unordered for
  parallel calls; extra arguments only at their defaults; long free-text
  arguments need only be present); `no_tool_call` rows pass with prose and no
  call.

## Decontamination

8-gram overlap (3 shared 8-grams, or the whole of a short item) against
IFEval, IFBench, Arena-Hard v2, MT-Bench, BFCL v4, tau2-bench, When2Call,
XSTest, MMLU-Pro and GPQA Diamond, on every user turn (on the base prompt only
for generated instructions) and on tool descriptions; function names matched
against tau2-bench everywhere, and against BFCL and When2Call for xLAM.
When2Call is built from xLAM's own API pool, so an xLAM row is dropped when any
of its tools carries a When2Call function name: 205 of 32,693 xLAM rows remain,
all kept, and Toucan refills the tool-call block. Exact and near
(MinHash) duplicates removed within and across sources, and against the prompts
of `reliquary-instruction-following`. Counts per filter and per evaluation are
in the build's `pool_stats.json`.

## Rebuilding

`scripts/build/` is the derivation, run in order with `GENERAL_PROMPTS_WORK`
pointing at a scratch directory: `evals_dl.py`, `wc_extract.py`, `extract.py
<source>…`, `pools.py`, `assemble.py`; then `scripts/pack_corpus.py
<work>/final <dataset dir>`, which writes the files to upload and the pins.
Upload, set `corpus.REVISION` to the commit, and run `scripts/write_goldens.py`.
Every choice is a hash of a key, so the same upstream revisions give the same
corpus. `RELIQUARY_GENERAL_DATA=<dataset dir>` serves a local copy instead of
the Hub; the pinned digests are checked either way.

## Size

The wheel is small: the corpus is not in it. The dataset is 97 MB of gzip,
most of it pasted chat context, long structured-output documents and tool
transcripts, one file per block, downloaded and decompressed only when a row
of that block is first asked for.
