# reliquary-telecom-solo

A support ticket arrives — *unable to send MMS since this morning, customer
John Smith, 555-123-2002, currently abroad in France* — and one actor works it
with 44 tools: thirteen that read and write the carrier's records, thirty that
poke the phone in the customer's hand, and `done`. The reward is a set of
predicates over the world left behind. Either the phone can send an MMS at the
end or it cannot, and no part of deciding that involves a model.

τ²-bench's telecom domain, ported in solo mode. 2,285 tickets, multi-turn, no
network, no clock, no conversation.

## Test and load

```bash
uv sync --locked
uv run pytest
uv run python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-telecom-solo'); print(next(iter(vf.load_taskset(c(id='reliquary-telecom-solo')))).key)"
```

To evaluate a model:

```bash
uv run eval reliquary-telecom-solo -n 24 --env.agent.max-turns 40
```

## Why solo mode

τ²-bench normally runs an LLM user simulator opposite the agent: the agent says
"please turn data roaming on" and a second model, playing the customer, decides
whether to do it. That is disqualifying here. A rollout has to be replayable
offline from its action list, and one of the two actors was a sampled model —
so the same actions would produce a different episode on the second run, and
there would be nothing for a validator to check.

Solo mode deletes the simulator. `LLMSoloAgent` with `DummyUser` gives the
agent **both** toolsets and a ticket instead of a conversation
(`src/tau2/agent/llm_agent.py:321`, `src/tau2/runner/build.py:109`). All 2,285
telecom tasks carry a non-empty `ticket`; no other τ² domain does, which is
why telecom is the domain that could be ported this way.

### The integrity argument

Solo mode is not only a convenience. In dual-control telecom, **2,539 of the
3,674 reward assertions read the user's device state** — `assert_can_send_mms`,
`assert_mobile_data_status`, `assert_service_status` — and the user simulator is
the actor that writes to that state. A rollout producer running both sides
could therefore have the "user" fix its own phone, collect the reward, and hand
in a transcript that replays faithfully: the cheat is in the trajectory, so
replay reproduces it rather than catching it. The only defence would be judging
whether the agent *caused* the fix, which is a question about intent and
therefore a model's opinion.

With one actor the attack does not exist. Every call is the agent's, the agent
is the thing being paid for, and "the phone works now" is the whole claim.

## A worked example

Task `[mms_issue]break_app_both_permissions|user_abroad_roaming_disabled_off`,
from the `qualification` split:

> The user has been unable to send MMS messages using their messaging app for
> the past few hours. Customer name: John Smith, phone number: 555-123-2002,
> current location: abroad in France. They will consider the issue resolved
> when an MMS message can be successfully sent.

Three things are wrong at once, arranged by the task's setup calls: the
messaging app has lost its `sms` and `storage` permissions, the line has
roaming disabled on the carrier's side, and the phone has data roaming switched
off — and the customer is in France, so the last two both matter.

One assertion decides the reward:

| side | predicate | wants |
| --- | --- | --- |
| user | `assert_can_send_mms` | `True` |

`_can_send_mms` is six checks in a row — mobile data working, not on 2G, Wi-Fi
calling not claiming MMS the carrier will not carry, an MMSC URL configured,
the messaging app installed, and that app holding both `storage` and `sms` —
and the first of those is six more, including whether the line is active on the
carrier's side and whether the user is abroad with roaming off. That is why a
ticket can be three-quarters fixed and score nothing.

The reference solution, four calls and the stop:

```
grant_app_permission(app_name="messaging", permission="sms")
grant_app_permission(app_name="messaging", permission="storage")
enable_roaming(customer_id="C1001", line_id="L1002")
toggle_roaming()
done()
```

Note the third and fourth: `enable_roaming` is the carrier permitting roaming
on the line and `toggle_roaming` is the handset setting. Neither alone lets the
data path come up abroad. That shape — a fault that spans both databases — is
most of what this environment teaches, and it is the shape dual-control τ²
splits across two actors.

Drop the last two calls and the reward is `0.0`. There is no partial credit:
three quarters of a working phone is a phone that cannot send an MMS.

## Reward

Binary, and the product of the components a task names in its reward basis.

| basis | tasks | what it checks |
| --- | --- | --- |
| `[ENV_ASSERTION]` | 2,253 | every predicate holds on the finished world |
| `[ENV_ASSERTION, ACTION]` | 32 | that, and every call in the reference solution was made |

The 32 are tickets that **cannot** be solved with the tools the agent has —
the SIM is PIN-locked, or the line was suspended when the contract ended — and
their required call is `transfer_to_human_agents`. The component exists because
their assertion is already true when the episode starts: without it, an agent
that did nothing at all would collect the reward for giving up correctly.

The predicates in use, over the whole corpus:

| predicate | side | assertions |
| --- | --- | --- |
| `assert_can_send_mms` | user | 1,984 |
| `assert_data_refueling_amount` | assistant | 1,120 |
| `assert_internet_speed` | user | 254 |
| `assert_mobile_data_status` | user | 254 |
| `assert_service_status` | user | 47 |
| `assert_no_overdue_bill` | assistant | 15 |

1,024 tickets carry one assertion, 1,133 carry two and 128 carry three.

None of them is a tool. They are ordinary undecorated methods on the two
toolkits, and `@is_tool` is what makes a method reachable by a tool call, so
the agent can neither read its own mark nor call the `break_*` helpers that set
the scenario up. `test_the_agent_cannot_call_its_own_marker` asserts both, and
also the asymmetry upstream built in deliberately: the agent gets
`grant_app_permission` and does not get `remove_app_permission`.

There is no LLM judge here and nowhere to put one. That is not a simplification
of τ²-bench — it is what telecom already is. Every other τ² domain scores on
`DB`, a hash compared against a reference trajectory the grader replays, or on
`COMMUNICATE`, which greps prose the agent said to a user who is not here.

## Episode length

| reference calls | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tickets | 45 | 61 | 168 | 312 | 424 | 449 | 382 | 257 | 130 | 46 | 10 | 1 |

Mean 5.78, median 6, 95th percentile 9, longest 12. With the stop call that is
a floor of 13 turns for an agent that already knows the answer, which none
does: the workflow policy prescribes a sequence of reads before any change, and
a wrong first guess costs a second pass. `max_turns` is **40** — room for three
diagnostic passes, and an end for an episode that is going nowhere. Upstream
allows 100, counting both sides of a conversation solo mode does not have.

The widest observation any tool can return is a customer's twelve bills with
their line items, 1,572 bytes; `max_observation_bytes` is 8 KiB.

## Determinism

The whole point of the port, and where the work went.

**The clock is frozen in upstream source.** `telecom/utils.py` defines
`get_now()` as `datetime(2025, 2, 25, 12, 8, 0)` and `get_today()` as
`date(2025, 2, 25)`, and every telecom tool goes through them. A bill period
computed during a rollout today is the same bill period computed by a validator
next month. That is inherited, not added.

**One real defect, fixed.** `telecom/tools.py:449` built a draft-bill id from
`uuid.uuid4().hex[:8]`. No assertion reads a bill id, so the reward never
noticed — but the id is handed back to the agent by `get_bills_for_customer`,
so the token transcript could not repeat, and a network that pays for a rollout
by replaying its transcript cannot price one whose tokens differ every run.
This package derives the id from a hash of the customer id and the billing
period instead, which is unique because a customer has at most one draft bill
per period.

The branch is reachable but never taken by a reference solution: every gold
`refuel_data` targets C1001, who already has a draft bill in the shipped
database, so the charge lands on the existing one. An agent refuelling a line
belonging to C1002 or C1003 creates a new one, which is exactly the sort of
thing an exploring policy does.

**A grep is the regression guard.** `test_nothing_reachable_is_non_determinis-
tic` parses every module in the package, strips comments and docstrings, and
fails on any remaining mention of `time.time`, `datetime.now`, `date.today`,
`uuid`, `random`, `requests`, `httpx`, `socket`, `urllib`, `getenv` or
`environ`. Comments are stripped so that the prose explaining why `uuid.uuid4`
is gone does not trip the check describing it; other string literals are kept,
because `__import__("uuid")` is precisely what is worth catching.

**One thing that looks non-deterministic and is not.** `_run_speed_test` reads
like a simulation with a random component. It is not: the speed is the midpoint
of a fixed range for the connected technology, scaled by fixed factors for
signal strength, VPN performance and data saver mode. `assert_internet_speed`,
which 254 tasks score on, therefore returns the same verdict in every process.
Nothing else in the vendored code samples.

## Splits

`train`, `eval` and `qualification` take 80%, 10% and 10% — drawn **inside each
task family**, by ranking the family's task ids on a hash and slicing.

| family | corpus | train | eval | qualification |
| --- | --- | --- | --- | --- |
| `mms_issue` | 1,984 | 1,587 | 198 | 199 |
| `mobile_data_issue` | 254 | 203 | 25 | 26 |
| `service_issue` | 47 | 37 | 5 | 5 |
| total | 2,285 | 1,827 | 228 | 230 |

Its sibling environments hash the row key over the whole corpus and let the mix
come out right in expectation. That does not work here: the families are
lopsided, 10% of 47 service tickets is 4.7, and where that lands decides
whether the eval split holds two of them or nine. Slicing each family
separately makes the mix exact — no family's share moves by as much as half a
point between splits, which `test_splits_preserve_the_family_mix` measures
rather than assumes. Hashing, rather than slicing the file, still keeps a task
in the split it has always been in whatever order the corpus arrives in.

## Two surfaces

`TelecomSoloTaskset` for Verifiers and `prime-rl`, and `TelecomSoloEnvironment`
for synchronous Reliquary-compatible replay. The package imports no Reliquary
code, and importing it does not import Verifiers: a validator reproducing a
reward needs the corpus and the vendored tools, not the training stack.

They are not two implementations. The Verifiers toolset carries the same 44
methods, and every one of them replays through `TelecomSoloEnvironment`: the
state that travels between tool calls is the list of calls, and the world is
what that list builds. Rebuilding the world on every call is not a workaround
for a state channel that carries JSON — it is the property this environment is
claiming, wired so that a bug in it is a failing test rather than a discrepancy
found at settlement.

Two things the surfaces do differ on, both cosmetic and both tested:

- The replay surface advertises the frozen schema, which carries upstream's
  per-parameter descriptions and titles. The MCP surface derives its parameter
  schema from the Python signature, so it has the same names and the same
  required set and not the same prose *about* each parameter. Tool names,
  tool descriptions and parameter names are identical, which
  `test_both_surfaces_advertise_the_same_tools` checks.
- `TelecomSoloEnvironment` ends an episode on `done`, on 10 failed calls or at
  turn 40. Under Verifiers the harness owns termination; `done` is still a tool,
  because the prompt tells the agent to call it.

## What was vendored, and what was left

`reliquary_telecom_solo/_tau2/` carries 2,941 lines derived from τ²-bench,
headers and all. Every file says which upstream file it came from and what
changed.

| file | from | lines |
| --- | --- | --- |
| `user_tools.py` | `domains/telecom/user_tools.py` | 1,170 |
| `tools.py` | `domains/telecom/tools.py` | 791 |
| `user_data_model.py` | `domains/telecom/user_data_model.py` | 346 |
| `data_model.py` | `domains/telecom/data_model.py` | 260 |
| `environment.py` | `environment/environment.py`, `domains/telecom/environment.py` | 200 |
| `toolkit.py` | `environment/toolkit.py`, `environment/db.py`, `utils/*` | 160 |
| `__init__.py` | — | 14 |

Vendored rather than depended on, for the reason the instruction-following
environment vendors its checkers: `tau2` is a research harness that moves, and
the code a participant runs has to be the code the package shipped, pinned by
digest.

What was deliberately left behind: the other five domains, the LLM agent
framework, the user simulator, the orchestrator, the CLI, and the evaluators
for `DB`, `COMMUNICATE` and `NL_ASSERTION`. Also:

- **`loguru`.** Five calls logged a suspension, a resumption, two roaming
  toggles and a refuel; nothing read them, and a logger is a configuration two
  participants could differ on.
- **`addict`, via `update_pydantic_model_with_dict`.** Its only callers were
  `ToolKitBase.update_db` and `TelecomUserDB.update_device`, both driven by a
  task's `initialization_data` — and no telecom task carries any. All 2,285 set
  their scenario up through `initialization_actions` instead. `corpus._parse`
  refuses a task that carries one rather than silently ignoring it.
- **`docstring_parser`.** The 44 tool schemas were generated once with
  upstream's own reflection code and are shipped as `data/tool_schemas.json`,
  pinned by digest. A tool description that changes with a library upgrade is a
  prompt that changes with a library upgrade.
  `test_tool_schemas_match_the_toolkits` is what keeps the frozen copy honest:
  it checks every name, every parameter and every required set against the live
  signatures.
- **Upstream's `set_state`.** It re-derives a world by replaying recorded tool
  calls and comparing each result against the recorded `ToolMessage`. This
  package owns the episode loop, so it never has to.

Two divergences in behaviour, both deliberate:

- **A failing assertion scores zero rather than raising.** Upstream's
  `run_env_assertion` raises if a predicate returns a non-bool, and
  `EnvironmentEvaluator` does not catch exceptions from inside one. Here a
  predicate that trips over the world it is handed scores zero: a raise would
  cost a whole window over one rollout. `test_assertions_are_predicates` says
  the path is unreachable rather than merely forgiven — every assertion any
  task names returns a bool.
- **The tech-support policy is the workflow variant, not the manual.**
  Upstream's `TELECOM_TECH_SUPPORT_POLICY_MANUAL_SOLO_PATH` points at
  `tech_support_manual.md`, the dual-control document — a copy-paste in its own
  path table — so the manual has no solo variant to ship.

## Licences and attribution

- **τ²-bench.** Code under `_tau2/` and the data under `data/` are from
  [`sierra-research/tau2-bench`](https://github.com/sierra-research/tau2-bench),
  MIT, © Sierra Research. The licence text is `_tau2/LICENSE`; every vendored
  file carries a header naming its source. The task file is `tasks_full.json`
  byte for byte, pinned by sha256 in `environment.toml` and in
  `corpus.TASKS_SHA256`; the two starting databases, the two policy documents
  and the tool schemas are pinned the same way.
- **This package.** MIT, as its siblings.
