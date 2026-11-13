# reliquary-automationbench

[AutomationBench](https://github.com/zapier/AutomationBench) (Zapier, MIT) 1.0.6 as a
deterministic, replayable episode environment. Each task seeds a simulated business
workspace (up to 47 SaaS services: Gmail, Salesforce, Sheets, Slack, Zendesk, …) and is
graded by deterministic assertions over the world the agent leaves behind. No judge
model, no network, no sandbox: the world is pure Python and runs on CPU.

The agent gets upstream's default **api** toolset, the setup Artificial Analysis reports:

| tool | what it does |
| --- | --- |
| `api_search(query, top_k)` | BM25 over the endpoint catalogue; returns URLs, parameters, bodies |
| `api_fetch(method, url, params, body)` | calls an endpoint; `params` and `body` are JSON strings |
| `base64_encode(text)` | base64url, for Gmail bodies |

The reward is upstream's strict pass, `task_completed_correctly` (every scored
assertion holds, assertions already true at the start excluded). `grade_report` also
returns upstream's `partial_credit` and the task's domain; the official score is the
mean pass over `eval`, and the per-domain score the mean over each 100-row block.

## Splits

| split | rows | rows |
| --- | --- | --- |
| `eval` | 600 | `[0,100)` sales, `[100,200)` marketing, `[200,300)` operations, `[300,400)` support, `[400,500)` finance, `[500,600)` hr |
| `qualification` | 200 | upstream's `simple` domain (one- and two-step tasks), never scored upstream |
| `sales` … `hr`, `simple` | 100 / 200 | one domain alone, for local runs |

There is no `train` split.

## What differs from upstream, and why

- **Deterministic world.** Upstream mints ids with `uuid4()`/`random` and stamps records
  with `datetime.now()`. `scripts/vendor_upstream.py` rewrites every such call to
  `_determinism`: a `random.Random` seeded from (task, episode seed, call position) and a
  clock pinned to the task's `meta.current_time` (+1 s per call, +1 µs per reading). The
  503 tasks that seed no scenario time run at `2026-08-04T21:42:13Z`, the pinned commit's
  time (upstream reads the host clock). Naive `.timestamp()` reads UTC, not the host zone;
  set-typed fields serialise sorted; Gmail's thread listing keeps first-appearance order
  (upstream cut it at `max_results` in hash order).
- **Offline.** The OpenAI key reads are `None`, so the `chatgpt` service always returns
  upstream's offline stub; the dead model client is removed.
- **One call per turn.** A Reliquary turn holds one tool call; upstream allows parallel
  calls within 50 responses. `max_turns` is 100.
- **Bounded observations.** A tool result is cut at 24 KiB (upstream does not cut).
- **Prompt.** Upstream's system prompt is prepended to the request in one user message;
  the harness's renderer owns the system turn (tool list, call format).

## Evidence

- `scripts/parity_upstream.py <checkout>`: the same action lists graded by upstream's own
  world, tools and `partial_credit` and by this port; per-assertion verdicts, partial
  credit and pass must agree on every task. Upstream ships no reference solutions, so the
  action lists are probes built from each task's assertions (`tests/probes.py`) and fuzz
  calls over the task's services' endpoints.
- `tests/`: replays repeat themselves; the world cache never changes a result; replays
  are byte-identical across `PYTHONHASHSEED` and `TZ`; no host-reading call is left in the
  vendored code; a digest of probe episodes is pinned.

## Regenerating

```bash
git clone https://github.com/zapier/AutomationBench && git -C AutomationBench checkout 4a8e1061254004d9dac807054eed33fad7d1ff14
python -I scripts/export_tasks.py AutomationBench reliquary_automationbench/data/tasks.jsonl.gz   # needs `datasets`
python -I scripts/vendor_upstream.py AutomationBench
python scripts/write_artifact.py
```
