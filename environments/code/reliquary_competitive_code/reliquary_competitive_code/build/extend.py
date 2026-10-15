"""v1.1: new problems appended to the published v1 dataset, all for SFT.

    python -m reliquary_competitive_code.build.extend STAGE --work DIR ...

Sources: open-r1/codeforces (rating 1200-2400, before 2024-08, stdio, no
checker, not interactive), Code-Contests-Plus 1x (stock checkers our
comparison reproduces), and the v1 rejects whose references are Python 2,
converted by 2to3. Each stage reads the previous one's output in `--work` and
can be rerun; `curate` skips problems already decided.

- `select` (statements only, network for Code-Contests-Plus metadata): the
  source filters; decontamination against every LiveCodeBench version and the
  v1 eval and qualification statements (13-gram overlap >= 0.10); the
  statement's own several-answers rule; exact and MinHash dedup against the
  21,294 DeepCoder rows of v1 and across the new sources; then the 13-gram
  containment guard against that pool and every problem kept before
  (`--containment`). Writes `selected.jsonl` and `select_report.json`.
- `fetch` (network): tests and references of the selected problems, one
  upstream file at a time (`--scratch`, deleted after reading), spilled to
  `work/problems/`. Tests over the byte budget alone are never kept, so they
  are not spilled.
- `curate` (no network: run it under `unshare -n`): runs the references,
  drops problems whose test contradicts an example, keeps a problem as
  `python_reference`, `py2to3` or `upstream_tests_only` (see
  `validate.Policy`). Writes `curated.jsonl`, `rejected.jsonl`,
  `curate_report.json`.
- `write`: the base dataset with the curated problems appended to train,
  `build_report.json` with the new SFT range.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import sys
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from reliquary_competitive_code.build import hub
from reliquary_competitive_code.build.dedup import CONTAINMENT_THRESHOLD, ContainmentGuard, deduplicate
from reliquary_competitive_code.build.filters import (
    CONTAMINATION_THRESHOLD,
    HeldOutIndex,
    example_disagrees,
    is_after_cutoff,
    is_multi_answer,
    load_lcb_statements,
)
from reliquary_competitive_code.build.io import append_dataset
from reliquary_competitive_code.build.validate import Curated, Policy, curate
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.layout import SPLITS, problems_file
from reliquary_competitive_code.sources import codecontests_plus as ccplus
from reliquary_competitive_code.sources import deepcoder, py2
from reliquary_competitive_code.sources import openr1_codeforces as openr1
from reliquary_competitive_code.sources.common import SourceRow, problem_id

BUILD = "v1.1"
GROUPS = ("openr1", "ccplus", "py2")
_GROUP_OF_SOURCE = {openr1.SOURCE: "openr1", ccplus.SOURCE: "ccplus"}

# The policy of v1.1 problems (documented in the README, "v1.1 sources").
TEST_BYTES = 4 << 20
MAX_TESTS = 32
PYTHON_TIME_FACTOR = 4.0
MAX_TIME_LIMIT_S = 10.0
# What `fetch` spills per problem at most (the budget then picks from it).
SPILL_BYTES = 32 << 20


def _group(source: str) -> str:
    return _GROUP_OF_SOURCE.get(source, "py2")


# ---------------------------------------------------------------- the base


def base_statements(base: Path) -> dict[str, list[tuple[str, str]]]:
    out = {}
    for split in SPLITS:
        table = pq.read_table(Path(base) / problems_file(split), columns=["problem_id", "statement"])
        out[split] = list(zip(table.column(0).to_pylist(), table.column(1).to_pylist()))
    return out


def replay_v1(root: Path, lcb: Path, report: Counter) -> tuple[list[SourceRow], list[SourceRow]]:
    """v1's stages before the references, on statements: every DeepCoder row
    (the dedup pool) and v1's dedup leads. Tests are replaced by
    placeholders of the same count (the lead rule reads only the count)."""
    v1_held = HeldOutIndex(load_lcb_statements(lcb))
    pool, passed, disagreeing = [], [], set()
    for row in deepcoder.rows(root):
        light = SourceRow(row.source, row.upstream_id, row.statement, (None,) * len(row.tests), row.references)
        pool.append(light)
        if is_after_cutoff(row) or v1_held.overlap(row.statement) >= CONTAMINATION_THRESHOLD or is_multi_answer(row):
            continue
        if example_disagrees(row):
            disagreeing.add(problem_id(row.statement))
        passed.append(light)
    leads = deduplicate([r for r in passed if problem_id(r.statement) not in disagreeing])
    report["py2:v1_rows"] = len(pool)
    report["py2:v1_leads"] = len(leads)
    return pool, leads


# ------------------------------------------------------------------ select


def _openr1_parquets(data_dir: Path | None) -> Iterator[tuple[str, pq.ParquetFile]]:
    if data_dir is not None:
        for path in sorted(Path(data_dir).glob("train-*.parquet")):
            yield path.name, pq.ParquetFile(path)
        return
    for name in hub.list_files(openr1.REPOSITORY, openr1.REVISION, openr1.DATA_PATTERN):
        yield name, hub.retry(lambda n=name: hub.remote_parquet(openr1.REPOSITORY, openr1.REVISION, n), name)


def openr1_metadata(data_dir: Path | None) -> Iterator[dict]:
    for _, parquet in _openr1_parquets(data_dir):
        yield from parquet.read(columns=openr1.METADATA_COLUMNS).to_pylist()


def ccplus_metadata() -> Iterator[dict]:
    for name in hub.list_files(ccplus.REPOSITORY, ccplus.REVISION, ccplus.DATA_PATTERN):
        parquet = hub.retry(lambda n=name: hub.remote_parquet(ccplus.REPOSITORY, ccplus.REVISION, n), name)
        for group in range(parquet.num_row_groups):
            table = hub.retry(lambda g=group: parquet.read_row_group(g, columns=ccplus.METADATA_COLUMNS), name)
            for record in table.to_pylist():
                record["_file"], record["_row_group"] = name, group
                yield record


def select(args: argparse.Namespace) -> None:
    work = _work(args)
    report: Counter[str] = Counter()
    base = base_statements(args.base)
    base_ids = {pid for rows in base.values() for pid, _ in rows}
    held_out = HeldOutIndex(
        load_lcb_statements(args.lcb, since=None)
        + [s for split in ("eval", "qualification") for _, s in base[split]]
    )

    def clean(group: str, row: SourceRow) -> bool:
        if is_after_cutoff(row):
            report[f"{group}:dropped_date"] += 1
        elif held_out.overlap(row.statement) >= CONTAMINATION_THRESHOLD:
            report[f"{group}:dropped_contaminated"] += 1
        elif is_multi_answer(row):
            report[f"{group}:dropped_multi_answer"] += 1
        else:
            return True
        return False

    pool, leads = replay_v1(args.deepcoder, args.lcb, report)
    print(f"replayed v1: {len(pool)} rows, {len(leads)} leads", flush=True)

    # open-r1 and Code-Contests-Plus, statements only.
    candidates: dict[str, SourceRow] = {}
    extra: dict[str, dict] = {}
    codeforces_meta: dict[str, dict] = {}
    for record in openr1_metadata(args.openr1_data):
        codeforces_meta[record["id"]] = {"date": openr1.contest_date(record), "rating": record.get("rating")}
        report["openr1:loaded"] += 1
        if reason := openr1.drop_reason(record):
            report[f"openr1:dropped_{reason}"] += 1
            continue
        row = openr1.candidate(record)
        if clean("openr1", row):
            key = f"{row.source}\t{row.upstream_id}"
            candidates[key] = row
            extra[key] = {"contest_id": record["contest_id"]}
    print(f"open-r1: {report['openr1:loaded']} rows", flush=True)
    for record in ccplus_metadata():
        report["ccplus:loaded"] += 1
        if reason := ccplus.drop_reason(record, codeforces_meta):
            report[f"ccplus:dropped_{reason}"] += 1
            continue
        row = ccplus.candidate(record, codeforces_meta)
        if clean("ccplus", row):
            key = f"{row.source}\t{row.upstream_id}"
            candidates[key] = row
            extra[key] = {"file": record["_file"], "row_group": record["_row_group"],
                          "checker": ccplus.checker_kind(record.get("checker")),
                          "codeforces_key": ccplus.codeforces_key(record)}
    print(f"ccplus: {report['ccplus:loaded']} rows", flush=True)
    for key, row in candidates.items():
        report[f"{_group(row.source)}:candidates"] += 1

    # Exact and MinHash dedup, against the v1 pool and across sources: the
    # tag of each row rides in `references`, which merging pools.
    tagged = [SourceRow(r.source, r.upstream_id, r.statement, (), ("pool",)) for r in pool]
    tagged += [SourceRow(r.source, r.upstream_id, r.statement, (), (key,)) for key, r in candidates.items()]
    clusters = []
    for merged in deduplicate(tagged):
        keys = [tag for tag in merged.references if tag != "pool"]
        if not keys:
            continue
        priority = sorted(keys, key=lambda k: (GROUPS.index(_group(candidates[k].source)), k))
        if "pool" in merged.references:
            for key in keys:
                report[f"{_group(candidates[key].source)}:dropped_dup_of_v1_pool"] += 1
            continue
        for key in priority[1:]:
            report[f"{_group(candidates[key].source)}:merged_into_other_source_copy"] += 1
        clusters.append(priority)

    # Python 2 recovery: v1 leads not curated, with a Python 2 reference.
    py2_rows = []
    for lead in leads:
        if problem_id(lead.statement) in base_ids:
            continue
        report["py2:v1_rejected_leads"] += 1
        if not any(py2.looks_python2(code) for code in lead.references):
            report["py2:dropped_no_python2_reference"] += 1
            continue
        report["py2:candidates"] += 1
        if clean("py2", lead):
            py2_rows.append(lead)

    # The containment guard, greedy: open-r1, then CC+, then Python 2, each
    # by problem id; open-r1 and CC+ against the v1 pool, the Python 2 leads
    # (themselves in that pool) against v1's curated problems.
    pool_guard = ContainmentGuard((r.statement for r in pool), args.containment)
    curated_guard = ContainmentGuard((s for rows in base.values() for _, s in rows), args.containment)
    kept_guard = ContainmentGuard((), args.containment)
    selected, seen = [], set(base_ids)

    def admit(group: str, row: SourceRow, against: ContainmentGuard) -> bool:
        pid = problem_id(row.statement)
        if pid in seen:
            report[f"{group}:dropped_same_id"] += 1
            return False
        if _union_share(row.statement, against, kept_guard) >= args.containment:
            report[f"{group}:dropped_contained_13gram"] += 1
            return False
        seen.add(pid)
        kept_guard.add(row.statement)
        report[f"{group}:selected"] += 1
        return True

    clusters.sort(key=lambda keys: (GROUPS.index(_group(candidates[keys[0]].source)),
                                    problem_id(candidates[keys[0]].statement)))
    for keys in clusters:
        lead = candidates[keys[0]]
        group = _group(lead.source)
        if admit(group, lead, pool_guard):
            selected.append(_entry(group, lead, [candidates[k] for k in keys], [extra[k] for k in keys]))
    for lead in sorted(py2_rows, key=lambda r: problem_id(r.statement)):
        if admit("py2", lead, curated_guard):
            entry = _entry("py2", lead, [lead], [{}])
            entry["references"] = list(lead.references)
            selected.append(entry)

    with (work / "selected_all.jsonl").open("w") as handle:  # before any sampling
        for entry in selected:
            handle.write(json.dumps({k: entry[k] for k in ("pid", "group", "source", "upstream_id", "rating")}) + "\n")
    selected = _sample(selected, args)
    report.update({f"{g}:sampled": sum(e["group"] == g for e in selected) for g in GROUPS})
    with (work / "selected.jsonl").open("w") as handle:
        for entry in selected:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _write_report(work / "select_report.json", report)


def _union_share(statement: str, *guards: ContainmentGuard) -> float:
    grams = ContainmentGuard._hashed(statement)
    if not grams:
        return 0.0
    return sum(1 for g in grams if any(g in guard._grams for guard in guards)) / len(grams)


def _entry(group: str, lead: SourceRow, members: list[SourceRow], extras: list[dict]) -> dict:
    return {
        "pid": problem_id(lead.statement),
        "group": group,
        "source": lead.source,
        "upstream_id": lead.upstream_id,
        "statement": lead.statement,
        "contest_date": lead.contest_date,
        "time_limit_s": lead.time_limit_s,
        "rating": next((m.rating for m in members if m.rating is not None), None),
        "upstream_tests": lead.upstream_tests,
        "members": [{"source": m.source, "upstream_id": m.upstream_id, **x} for m, x in zip(members, extras)],
    }


def _sample(selected: list[dict], args: argparse.Namespace) -> list[dict]:
    """A smoke subset: open-r1 whole contests (few generated-test files),
    CC+ from the fewest row groups (each costs a part download), Python 2 at
    random. A group with no `--sample-*` is kept whole."""
    wanted = {"openr1": args.sample_openr1, "ccplus": args.sample_ccplus, "py2": args.sample_py2}
    if all(n is None for n in wanted.values()):
        return selected
    rng = random.Random(args.seed)
    out = []
    for group in GROUPS:
        entries = [e for e in selected if e["group"] == group]
        n = wanted[group]
        if n is None:
            out += entries
            continue
        if group == "openr1":
            by_contest = defaultdict(list)
            for e in entries:
                by_contest[e["members"][0]["contest_id"]].append(e)
            contests = sorted(by_contest)
            rng.shuffle(contests)
            chosen = []
            for contest in contests:
                if len(chosen) >= n:
                    break
                chosen += by_contest[contest]
            out += chosen[:n]
        elif group == "ccplus":
            by_group = defaultdict(list)
            for e in entries:
                lead = e["members"][0]
                by_group[(lead["file"], lead["row_group"])].append(e)
            chosen = []
            for _, items in sorted(by_group.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                if len(chosen) >= n:
                    break
                chosen += items
            out += chosen[:n]
        else:
            out += rng.sample(entries, min(n, len(entries)))
    return out


# ------------------------------------------------------------------- fetch


class Spill:
    """Per problem, its tests as gzip JSON lines under work/problems/<pid>/,
    one file per upstream part, each capped so that a problem never spills
    more than SPILL_BYTES."""

    def __init__(self, root: Path, test_bytes: int) -> None:
        self.root, self.test_bytes = Path(root), test_bytes
        self.used: Counter[str] = Counter()

    def write(self, pid: str, part: str, tests: Iterable[TestCase]) -> int:
        directory = self.root / pid
        directory.mkdir(parents=True, exist_ok=True)
        written = 0
        with gzip.open(directory / f"{part}.jsonl.gz", "wt", encoding="utf-8") as handle:
            for test in tests:
                size = len(test.stdin.encode()) + len(test.stdout.encode())
                if size > self.test_bytes or self.used[pid] + size > SPILL_BYTES:
                    continue
                handle.write(json.dumps([test.stdin, test.stdout], ensure_ascii=False) + "\n")
                self.used[pid] += size
                written += 1
        return written

    def read(self, pid: str) -> tuple[TestCase, ...]:
        tests = []
        for path in sorted((self.root / pid).glob("*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                tests += [TestCase(*json.loads(line)) for line in handle]
        return tuple(dict.fromkeys(tests))


def fetch(args: argparse.Namespace) -> None:
    work = _work(args)
    selected = _selected(work)
    spill = Spill(work / "problems", args.test_bytes)
    references: dict[str, dict[str, list[str]]] = {e["pid"]: {"py3": [], "py2": []} for e in selected}
    report: Counter[str] = Counter()

    # Python 2 leads: the tests from DeepCoder, the references pooled at select.
    py2_entries = {e["upstream_id"]: e for e in selected if e["group"] == "py2"}
    for row in deepcoder.rows(args.deepcoder, wanted=set(py2_entries)):
        entry = py2_entries[row.upstream_id]
        report["py2:tests_spilled"] += spill.write(entry["pid"], "deepcoder", row.tests)
        references[entry["pid"]]["pooled"] = entry["references"]

    # open-r1 leads: official tests, then generated ones contest by contest.
    leads = {e["upstream_id"]: e["pid"] for e in selected if e["source"] == openr1.SOURCE}
    for name, parquet in _openr1_parquets(args.openr1_data):
        for group in range(parquet.num_row_groups):
            ids = parquet.read_row_group(group, columns=["id"]).column(0).to_pylist()
            if not leads.keys() & set(ids):
                continue
            records = parquet.read_row_group(group, columns=["id", "official_tests"]).to_pylist()
            for upstream_id, tests in openr1.official_tests(records, set(leads)):
                report["openr1:official_tests_spilled"] += spill.write(leads[upstream_id], "official", tests)
    contests = sorted({e["members"][0]["contest_id"] for e in selected if e["source"] == openr1.SOURCE}, key=int)
    available = set(hub.list_files(openr1.REPOSITORY, openr1.REVISION, "generated_tests/*.parquet")) if contests else set()
    for n, contest in enumerate(contests):
        path = openr1.generated_tests_file(contest)
        if path not in available:
            report["openr1:contests_without_generated_tests"] += 1
            continue
        with hub.downloaded(openr1.REPOSITORY, openr1.REVISION, path, args.scratch) as local:
            per_problem: dict[str, list[TestCase]] = defaultdict(list)
            parquet = pq.ParquetFile(local)
            for batch in parquet.iter_batches(batch_size=16, columns=["problem_id", "input", "output"]):
                for record in batch.to_pylist():
                    if record["problem_id"] in leads and (test := openr1.generated_test(record)):
                        size = len(test.stdin) + len(test.stdout)
                        if size <= args.test_bytes:
                            per_problem[record["problem_id"]].append(test)
            for upstream_id, tests in per_problem.items():
                report["openr1:generated_tests_spilled"] += spill.write(leads[upstream_id], "generated", tests)
        print(f"generated tests: contest {contest} ({n + 1}/{len(contests)})", flush=True)

    # Codeforces submissions for every Codeforces member (open-r1 ids, and
    # CC+'s Codeforces problems under open-r1's id).
    by_key = defaultdict(set)
    for e in selected:
        for m in e["members"]:
            key = m["upstream_id"] if m["source"] == openr1.SOURCE else m.get("codeforces_key")
            if key:
                by_key[key].add(e["pid"])
    if by_key:
        for key, codes in _codeforces_submissions(set(by_key), args.openr1_submissions).items():
            for pid in by_key[key]:
                for kind in ("py3", "py2"):
                    references[pid][kind] += codes[kind]

    # Code-Contests-Plus: one part at a time, its needed row groups only.
    wanted = defaultdict(lambda: defaultdict(dict))  # file -> row group -> id -> (pid, is_lead)
    for e in selected:
        for i, m in enumerate(e["members"]):
            if m["source"] == ccplus.SOURCE:
                wanted[m["file"]][m["row_group"]][m["upstream_id"].split("/", 1)[1]] = (e["pid"], i == 0)
    for name in sorted(wanted):
        with hub.downloaded(ccplus.REPOSITORY, ccplus.REVISION, name, args.scratch) as local:
            parquet = pq.ParquetFile(local)
            for group, ids in sorted(wanted[name].items()):
                for batch in parquet.iter_batches(batch_size=2, row_groups=[group], columns=ccplus.TEST_COLUMNS):
                    for record in batch.to_pylist():
                        if record["id"] not in ids:
                            continue
                        pid, is_lead = ids[record["id"]]
                        py3_codes, py2_codes = ccplus.submissions(record)
                        references[pid]["py3"] += py3_codes[: ccplus.PER_PROBLEM]
                        references[pid]["py2"] += py2_codes[: ccplus.PER_PROBLEM]
                        if is_lead:
                            report["ccplus:tests_spilled"] += spill.write(pid, "ccplus", ccplus.tests_of(record))
        print(f"ccplus: {name} read and deleted", flush=True)

    with (work / "references.json").open("w") as handle:
        json.dump(references, handle)
    report["problems_with_python3"] = sum(bool(r["py3"]) for r in references.values())
    report["problems_with_python2"] = sum(bool(r["py2"]) for r in references.values())
    _write_report(work / "fetch_report.json", report)


def _codeforces_submissions(keys: set[str], local: Path | None = None) -> dict[str, dict[str, list[str]]]:
    """Accepted Python submissions per problem, by row-group statistics on
    `problem_id`: only the row groups that may hold a wanted problem are read,
    from `local` (build.fetch --v11) or over HTTP."""
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"py3": [], "py2": []})
    lock = threading.Lock()
    if local is not None:
        names = [str(path) for path in sorted(Path(local).glob("train-*.parquet"))]
    else:
        names = hub.list_files(openr1.SUBMISSIONS_REPOSITORY, openr1.SUBMISSIONS_REVISION, openr1.SUBMISSIONS_PATTERN)
    low, high = min(keys), max(keys)

    def open_file(name: str) -> pq.ParquetFile:
        if local is not None:
            return pq.ParquetFile(name)
        return hub.remote_parquet(openr1.SUBMISSIONS_REPOSITORY, openr1.SUBMISSIONS_REVISION, name)

    def scan(name: str) -> None:
        parquet = hub.retry(lambda: open_file(name), name)
        meta = parquet.metadata
        column = meta.schema.names.index("problem_id")
        for group in range(meta.num_row_groups):
            stats = meta.row_group(group).column(column).statistics
            if stats is not None and stats.has_min_max and (stats.max < low or stats.min > high
                                                            or not any(stats.min <= k <= stats.max for k in keys)):
                continue
            table = hub.retry(lambda g=group: parquet.read_row_group(
                g, columns=["problem_id", "programmingLanguage", "verdict", "source"]), name)
            for key, language, verdict, code in zip(*(table.column(c).to_pylist() for c in table.column_names)):
                kind = openr1.submission_language(language)
                if key in keys and verdict == "OK" and kind and code:
                    with lock:
                        if len(out[key][kind]) < openr1.PER_PROBLEM:
                            out[key][kind].append(code)
        print(f"submissions: {name}", flush=True)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(scan, names))
    return out


# ------------------------------------------------------------------ curate


def policy_of(args: argparse.Namespace) -> Policy:
    return Policy(
        test_bytes=args.test_bytes,
        max_tests=args.max_tests,
        python_time_factor=args.python_time_factor,
        max_time_limit_s=args.max_time_limit,
    )


def row_of(entry: dict, tests: tuple[TestCase, ...], refs: dict) -> SourceRow:
    if entry["group"] == "py2":
        references, converted = py2.upgrade(tuple(refs.get("pooled", ())))
    else:
        upgraded, converted = py2.upgrade(tuple(refs["py2"]))
        references = tuple(dict.fromkeys([*refs["py3"], *upgraded]))
    return SourceRow(
        source=entry["source"],
        upstream_id=entry["upstream_id"],
        statement=entry["statement"],
        tests=tests,
        references=references,
        contest_date=entry["contest_date"],
        upstream_tests=entry["upstream_tests"],
        time_limit_s=entry["time_limit_s"],
        rating=entry["rating"],
        converted=converted,
    )


def decide(entry: dict, row: SourceRow, policy: Policy) -> Curated | str:
    """A curated problem, or the reason it is not one."""
    if entry["members"][0].get("checker") == "nyesno" and not ccplus.yes_no_only(list(row.tests)):
        return "yesno_checker_other_tokens"
    if example_disagrees(row):
        return "example_mismatch"
    outcome = curate(row, policy)
    return outcome if isinstance(outcome, Curated) else outcome.reason


def curate_stage(args: argparse.Namespace) -> None:
    work = _work(args)
    selected = _selected(work)
    references = json.loads((work / "references.json").read_text())
    spill = Spill(work / "problems", args.test_bytes)
    policy = policy_of(args)
    done = set()
    for name in ("curated.jsonl", "rejected.jsonl"):
        if (work / name).exists():
            done |= {json.loads(line)["pid"] for line in (work / name).open()}
    todo = [e for e in selected if e["pid"] not in done]
    print(f"curate: {len(todo)} to decide, {len(done)} already decided", flush=True)
    lock = threading.Lock()
    curated_out = (work / "curated.jsonl").open("a")
    rejected_out = (work / "rejected.jsonl").open("a")

    def one(entry: dict) -> None:
        row = row_of(entry, spill.read(entry["pid"]), references[entry["pid"]])
        outcome = decide(entry, row, policy)
        with lock:
            if isinstance(outcome, Curated):
                curated_out.write(json.dumps({
                    "pid": entry["pid"], "group": entry["group"], "statement": outcome.statement,
                    "tests": [[t.stdin, t.stdout] for t in outcome.tests],
                    "time_limit_s": outcome.time_limit_s, "reference": outcome.reference,
                    "origin": origin_of(entry, outcome), "reference_cpu_s": outcome.reference_cpu_s,
                }, ensure_ascii=False) + "\n")
                curated_out.flush()
            else:
                rejected_out.write(json.dumps({"pid": entry["pid"], "group": entry["group"], "reason": outcome}) + "\n")
                rejected_out.flush()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    curated_out.close()
    rejected_out.close()
    report = curate_report(work, selected)
    _write_report(work / "curate_report.json", report)
    overloaded = sum(v for k, v in report.items() if k.endswith(":dropped_harness_overload"))
    if overloaded:
        raise SystemExit(
            f"{overloaded} problem(s) dropped as harness_overload: delete their lines from "
            "rejected.jsonl and rerun curate on a quieter box (fewer --workers)."
        )


def origin_of(entry: dict, curated: Curated) -> dict:
    return {
        "source": entry["source"],
        "upstream_id": entry["upstream_id"],
        "use": "sft",
        "build": BUILD,
        "reference_kind": curated.reference_kind,
        "rating": entry["rating"],
        "contest_date": entry["contest_date"],
        "upstream_time_limit_s": entry["time_limit_s"],
    }


def curate_report(work: Path, selected: list[dict]) -> dict:
    report: Counter[str] = Counter()
    for e in selected:
        report[f"{e['group']}:selected"] += 1
    ratios = []
    for line in (work / "curated.jsonl").open() if (work / "curated.jsonl").exists() else ():
        c = json.loads(line)
        report[f"{c['group']}:curated"] += 1
        report[f"{c['group']}:kind_{c['origin']['reference_kind']}"] += 1
        upstream = c["origin"]["upstream_time_limit_s"]
        if c["reference_cpu_s"] is not None and upstream:
            ratios.append(c["reference_cpu_s"] / upstream)
    for line in (work / "rejected.jsonl").open() if (work / "rejected.jsonl").exists() else ():
        r = json.loads(line)
        report[f"{r['group']}:dropped_{r['reason']}"] += 1
    out: dict = dict(report)
    if ratios:
        q = statistics.quantiles(ratios, n=20) if len(ratios) > 1 else ratios * 19
        out["python_cpu_over_upstream_limit"] = {
            "n": len(ratios), "median": round(statistics.median(ratios), 3),
            "p90": round(q[17], 3), "p95": round(q[18], 3), "max": round(max(ratios), 3),
        }
    return out


# ------------------------------------------------------------------- write


def write(args: argparse.Namespace) -> None:
    work = _work(args)
    curated = []
    for line in (work / "curated.jsonl").open():
        c = json.loads(line)
        curated.append(Curated(
            problem_id=c["pid"], split="train", statement=c["statement"],
            tests=tuple(TestCase(a, b) for a, b in c["tests"]), time_limit_s=c["time_limit_s"],
            reference=c["reference"], origin=c["origin"], reference_kind=c["origin"]["reference_kind"],
        ))
    files = append_dataset(args.base, curated, args.out)
    start = pq.ParquetFile(Path(args.base) / problems_file("train")).metadata.num_rows
    report = {
        "build": BUILD,
        "select": json.loads((work / "select_report.json").read_text()),
        "curate": json.loads((work / "curate_report.json").read_text()),
        "appended": len(curated),
        "train_sft_v11_start": start,
        "train_sft_v11_count": len(curated),
        "policy": {"test_bytes": args.test_bytes, "max_tests": args.max_tests,
                   "python_time_factor": args.python_time_factor, "max_time_limit_s": args.max_time_limit,
                   "containment": args.containment},
    }
    (Path(args.out) / "build_report.json").write_text(
        json.dumps({"report": report, "files": files}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"appended": len(curated), "train_sft_v11": [start, len(curated)], "files": files}, indent=1))


# -------------------------------------------------------------------- misc


def _work(args: argparse.Namespace) -> Path:
    Path(args.work).mkdir(parents=True, exist_ok=True)
    return Path(args.work)


def _selected(work: Path) -> list[dict]:
    return [json.loads(line) for line in (work / "selected.jsonl").open()]


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(dict(sorted(report.items())), indent=1) + "\n")
    print(path.read_text(), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m reliquary_competitive_code.build.extend")
    parser.add_argument("stage", choices=["select", "fetch", "curate", "write"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--base", type=Path, help="the published v1 dataset (four parquet files)")
    parser.add_argument("--lcb", type=Path, help="LiveCodeBench test*.jsonl (build.fetch)")
    parser.add_argument("--deepcoder", type=Path, help="DeepCoder taco + primeintellect (build.fetch)")
    parser.add_argument("--openr1-data", type=Path, default=None,
                        help="local open-r1/codeforces data/ (build.fetch --v11); read over HTTP if omitted")
    parser.add_argument("--openr1-submissions", type=Path, default=None,
                        help="local open-r1/codeforces-submissions data/ (build.fetch --v11); HTTP if omitted")
    parser.add_argument("--scratch", type=Path, help="where one upstream file at a time is downloaded")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--containment", type=float, default=CONTAINMENT_THRESHOLD)
    parser.add_argument("--test-bytes", type=int, default=TEST_BYTES, help="kept test bytes per problem")
    parser.add_argument("--max-tests", type=int, default=MAX_TESTS)
    parser.add_argument("--python-time-factor", type=float, default=PYTHON_TIME_FACTOR)
    parser.add_argument("--max-time-limit", type=float, default=MAX_TIME_LIMIT_S)
    parser.add_argument("--sample-openr1", type=int, default=None)
    parser.add_argument("--sample-ccplus", type=int, default=None)
    parser.add_argument("--sample-py2", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args(argv)
    required = {"select": ["base", "lcb", "deepcoder"], "fetch": ["deepcoder", "scratch"],
                "curate": [], "write": ["base", "out"]}[args.stage]
    if missing := [name for name in required if getattr(args, name) is None]:
        parser.error(f"{args.stage} needs --{', --'.join(m.replace('_', '-') for m in missing)}")
    {"select": select, "fetch": fetch, "curate": curate_stage, "write": write}[args.stage](args)


if __name__ == "__main__":
    main(sys.argv[1:])
