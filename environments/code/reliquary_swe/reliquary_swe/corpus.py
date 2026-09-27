"""SWE-bench and SWE-smith rows, and nothing that needs a container.

Deliberately free of Docker and of verifiers runtimes: corpus questions — how
many instances, which repositories, what a task id looks like — must be
answerable on a machine that cannot host images.

SWE-smith's schema forced `SweRow` to grow three optional fields rather than
gain a sibling dataclass (see the design spec's section 8 correction):
`base_commit`, `version` and `test_patch` are all empty for a SWE-smith row,
and `image` -- unset for SWE-bench Verified, which derives its image instead
(`swe_adapter.image_for`) -- is given directly, because SWE-smith's own
`image_name` column is authoritative and a from-scratch derivation
(`swesmith.profiles.RepoProfile.image_name`) was checked and found to compute
a *different*, wrong Docker Hub namespace (`swebench/...` rather than the
dataset's actual `jyangballin/...`). "The image is given, not derived" is not
a simplification here, it is the only correct option.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass

from datasets import load_dataset

# The splits `load_rows` (the SWE-bench-shaped loader) accepts. SWE-smith's
# "train" split is a different schema entirely and goes through
# `load_swesmith_rows` instead -- `_SOURCES` below also has a "train" entry
# (both loaders share the same pinned-revision convention), but `load_rows`
# itself must never accept it: SWE-smith rows have no `base_commit`,
# `version` or `test_patch` columns, so `load_rows("train")` would not fail
# fast with a clear message, it would fail deep inside row construction with
# a confusing `KeyError`.
SPLITS = ("eval",)

# Pinned by revision, as every other environment here pins its data. An
# unpinned corpus makes two runs incomparable for a reason that never shows up
# in the metrics.
_SOURCES = {
    "eval": (
        "princeton-nlp/SWE-bench_Verified",
        "test",
        "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
    ),
    # SWE-smith's own HF card text ("50137 task instances from 128
    # repositories") is stale -- it was not updated when the dataset grew;
    # the datasets-server API for this exact pinned revision reports 59,136
    # rows, which is what actually loads below and matches the task brief.
    "train": (
        "SWE-bench/SWE-smith",
        "train",
        "ea6d7173829c7ec8fa16c22055699ff2e9188091",
    ),
}

# The measured sweet spot (task brief, confirmed against the pinned revision
# below): 20 images -> 23,844 tasks. The task brief's own "~45GB" cites
# compressed download size; this package's own five pulled images (see the
# implementation report) averaged ~1.4GB compressed but ~3.9GB *on disk*
# once decompressed, so 20 images is closer to ~78GB on disk, not ~45GB --
# restated here from direct measurement rather than left standing on the
# inherited figure. Configurable via `SweTasksetConfig.num_images`; declared,
# never auto-sized from local free disk -- see corpus.load_swesmith_rows.
DEFAULT_SWESMITH_IMAGES = 20

# A named, documented value to reach for, NOT the active default (see
# `load_swesmith_rows`, which defaults `max_test_count` to `None`). The
# second, orthogonal selection dimension alongside `DEFAULT_SWESMITH_IMAGES`
# when a caller does pass it: a declared cap on fail-to-pass + pass-to-pass
# test count, because SWE-smith's per-instance test count is heavily
# right-skewed (measured on the full pinned corpus: p50=415, p75=1151,
# p90=2657, p95=5060, max=22028) and that skew IS a real per-instance cost
# difference for the subset of repos whose profile sets `min_testing=True`
# (their test command narrows to exactly these tests' files -- see
# swesmith_adapter.py).
#
# Not the default, because seven real container gradings across the size
# range (the implementation report has the transcripts) found this
# corpus's actual cost a non-problem: the single most expensive instance in
# the entire 59,136-row corpus (pandas-dev/pandas.pr_59615, 22,028
# fail+pass-to-pass entries, 23,724 tests *actually* run) graded in 40
# seconds -- two orders of magnitude below a naive per-test estimate
# carried over from a different benchmark (SWE-bench Verified) and a
# different, per-test-heavier repo (sympy). Discarding 632 real training
# rows (2.65% of the top 20 images' 23,844 pre-exclusion rows) by default
# to bound a cost this package's own measurement shows is not a problem
# would be the wrong trade. Kept as a named constant, at the measured p95,
# for the day a repository this package has not measured turns out as
# per-test-expensive as sympy did -- pass it explicitly as
# `max_test_count` when that day comes.
#
# For the larger share of the corpus regardless -- repos whose profile
# leaves `min_testing` at its default `False` -- this cap would not bound
# cost at all even if applied: their test command always runs the whole
# suite regardless of any one instance's fail-to-pass/pass-to-pass counts,
# so cost there is a per-repository constant, not a per-instance one.
# Measured directly (same report): whole-suite runs for four such
# repositories in the top 20 images took 3-34 seconds, cheap enough that no
# bound is needed for them either.
DEFAULT_SWESMITH_MAX_TEST_COUNT = 5060


@dataclass(frozen=True, slots=True)
class SweRow:
    instance_id: str
    repo: str
    problem_statement: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    gold_patch: str
    # A git revision `taskset.py` and `grading.py` can hand to `git checkout`
    # and `git checkout <rev> -- <path>` without caring which corpus a row
    # came from. SWE-bench Verified: a 40-hex base commit SHA. SWE-smith: no
    # such SHA exists in the row at all -- each instance is its own branch,
    # committed on top of the image's clean `main` (see swesmith_adapter's
    # module docstring) -- so this is instead the *expression*
    # `f"origin/{instance_id}~1"`: SWE-smith's own harness comments this
    # exact resolution as "HEAD~1 ... effectively brings the tests back into
    # the codebase" (the branch's tip additionally removes the fail-to-pass
    # test files; `~1` is the commit before that removal, bug present, tests
    # intact). Verified empirically: `git checkout -q` and `git reset --hard`
    # both resolve it cold, with no prior fetch, against a freshly pulled
    # image (see the implementation report).
    base_commit: str = ""
    # Keys `MAP_REPO_VERSION_TO_SPECS`; empty for SWE-smith, which resolves
    # its test command through `swesmith.profiles.registry` instead (keyed by
    # `repo`, not `(repo, version)` -- see swesmith_adapter.py).
    version: str = ""
    # Empty for SWE-smith: it injects its bug into source, never into tests,
    # so there is no test-restoring patch to reapply (spec section 8's
    # correction). `grading._restore_strategy_for` reads this field's
    # truthiness to choose a restoration strategy.
    test_patch: str = ""
    # Given directly for SWE-smith (`row["image_name"]`, authoritative -- see
    # the module docstring). `None` for SWE-bench Verified, whose image is
    # derived instead (`swe_adapter.image_for`).
    image: str | None = None
    # The shell command that runs this row's tests, when the corpus gives one
    # rather than having it derived from a per-repository registry (SWE-bench
    # Verified: `swe_adapter.test_command`; SWE-smith:
    # `swesmith_adapter.test_command`). Polyglot only; empty elsewhere.
    test_command: str = ""
    # Where the repository lives inside the image. `/testbed` for SWE-bench
    # Verified and SWE-smith; the polyglot corpus ships images of both
    # `/testbed` and `/workspace/repo` shapes and says which per row.
    workdir: str = "/testbed"


def _tests(raw: object) -> tuple[str, ...]:
    """SWE-bench stores the two test lists as JSON-encoded strings."""
    if isinstance(raw, str):
        return tuple(json.loads(raw))
    return tuple(raw or ())


@functools.lru_cache(maxsize=None)
def load_rows(split: str = "eval") -> tuple[SweRow, ...]:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r}; expected one of {SPLITS}")
    name, hf_split, revision = _SOURCES[split]
    dataset = load_dataset(name, split=hf_split, revision=revision or None)
    return tuple(
        SweRow(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            version=row["version"],
            problem_statement=row["problem_statement"],
            fail_to_pass=_tests(row["FAIL_TO_PASS"]),
            pass_to_pass=_tests(row["PASS_TO_PASS"]),
            gold_patch=row["patch"],
            test_patch=row["test_patch"],
        )
        for row in dataset
    )


# A unified-diff hunk header: `@@ -oldstart,oldlen +newstart,newlen @@ ...`.
# The length term is optional (git omits it for a 1-line hunk).
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _reverse_unified_diff(patch: str) -> str:
    """The syntactic reverse of a unified diff that only modifies existing
    file content -- swap each hunk's old/new line counts, and each hunk
    body's `+`/`-` line prefixes, leaving context lines and file headers
    alone.

    Why this function exists at all: SWE-smith's `patch` column is the diff
    that *introduces* its bug, not the fix -- confirmed three ways (see the
    implementation report): the diff text itself reads as a regression
    (`reversed(scope)`, a dropped `str()` cast, `None`->`[]`), a real
    container run showed FAIL_TO_PASS tests passing before `patch` and
    failing after it, and SWE-smith's own `_apply_patch` in
    `swesmith/harness/utils.py` applies gold predictions with
    `git apply --reverse`, commented plainly: "Because gold patches = bug
    patches, so fix = revert." `SweRow.gold_patch` must mean the same thing
    for every row regardless of source corpus -- "apply forward to fix" --
    since `grading.grade` always applies its `patch` argument forward, and
    `tests/test_goldens.py` calls `grade` with `data.gold_patch` directly
    with no per-corpus branch. Normalizing here, once, at load time, is what
    keeps that call site corpus-agnostic.

    Deliberately narrow: raises if the diff touches file existence (add,
    delete, rename, copy), since a correct reversal there also needs to swap
    the file-header lines (`--- /dev/null` <-> a real path, `new file mode`
    <-> `deleted file mode`), which this function does not implement.
    Checked against the full pinned corpus below (see the implementation
    report) -- none of it hits this, because SWE-smith's bug generation
    rewrites existing source, it never creates or removes a file.

    Line-order note, checked by hand and by container: swapping each
    hunk-body line's `+`/`-` character *in place* (never reordering lines)
    is sufficient. `git apply` matches `-`/context lines against the current
    file in the order they appear in the hunk and inserts `+` lines at that
    position; reordering an edit's deletions before or after its insertions
    changes nothing about the resulting file content as long as each line's
    relative order among same-direction lines is preserved, which an
    in-place swap always does.
    """
    if re.search(
        r"^(new file mode|deleted file mode|--- /dev/null|\+\+\+ /dev/null|"
        r"rename (from|to)|copy (from|to))",
        patch,
        re.MULTILINE,
    ):
        raise ValueError(
            "_reverse_unified_diff: file add/delete/rename/copy is not supported"
        )
    out = []
    for line in patch.split("\n"):
        match = _HUNK_HEADER.match(line)
        if match:
            old_start, old_len, new_start, new_len, rest = match.groups()
            out.append(
                f"@@ -{new_start},{new_len or '1'} "
                f"+{old_start},{old_len or '1'} @@{rest}"
            )
        elif line.startswith("+") and not line.startswith("+++"):
            out.append("-" + line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            out.append("+" + line[1:])
        else:
            out.append(line)
    return "\n".join(out)


def swesmith_image_rank() -> tuple[str, ...]:
    """Every SWE-smith image, sorted by task count descending, ties broken by
    image name ascending -- the declared, machine-independent ordering
    corpus selection is built on (spec section 8: "N alone determines the
    exact task set" -- now "N and max_test_count together", see
    `load_swesmith_rows`). Loads the full 59,136-row pinned corpus; cheap
    relative to that load, not relative to nothing, so callers that only need
    a small N should still expect one full dataset fetch.
    """
    name, hf_split, revision = _SOURCES["train"]
    dataset = load_dataset(name, split=hf_split, revision=revision or None)
    counts = Counter(dataset["image_name"])
    return tuple(sorted(counts, key=lambda image: (-counts[image], image)))


def _swesmith_row_defect(
    fail_to_pass: tuple[str, ...],
    pass_to_pass: tuple[str, ...],
    max_test_count: int | None,
) -> str | None:
    """Which unconditional or cap-based exclusion, if any, a SWE-smith row
    hits -- factored out of `load_swesmith_rows`'s loop so each reason is a
    pure, directly testable function of the row's own test lists, not
    something only exercisable by finding a real dataset row that happens
    to trigger it (none does, for the empty-FAIL_TO_PASS case, in the
    pinned revision -- see `load_swesmith_rows`'s own docstring).
    """
    if not fail_to_pass:
        return "empty_fail_to_pass"
    if max_test_count is not None and len(fail_to_pass) + len(pass_to_pass) > max_test_count:
        return "too_costly"
    return None


@functools.lru_cache(maxsize=None)
def load_swesmith_rows(
    num_images: int = DEFAULT_SWESMITH_IMAGES,
    max_test_count: int | None = None,
) -> tuple[SweRow, ...]:
    """The task set for the top `num_images` SWE-smith images by task count,
    optionally excluding any instance whose fail-to-pass + pass-to-pass test
    count exceeds `max_test_count`. Uncapped by default: real container
    measurements found the shipped corpus's cost a non-problem (see
    `DEFAULT_SWESMITH_MAX_TEST_COUNT`'s own docstring for the numbers), so
    the cap discards real training tasks by default only if a caller asks
    for it -- pass `DEFAULT_SWESMITH_MAX_TEST_COUNT` (its measured p95)
    explicitly if a repository this package has not measured turns out to
    need it.

    Deterministic given (pinned revision, num_images, max_test_count) alone:
    `swesmith_image_rank` fixes which images are in, the cap is a pure
    per-row arithmetic filter, and surviving rows are kept in the pinned
    dataset's own native order (never re-sorted) -- that native order is
    itself fixed by the pinned revision, so a second party with the same
    three inputs recomputes the identical, identically ordered task set with
    no coordination (spec section 8). Both parameters participate in what a
    task id means: two configs that differ in either one are different task
    sets, never a subset/superset relationship a shared index could paper
    over.

    Two further, small, unconditional exclusions, always applied regardless
    of `max_test_count`:

    - Empty `FAIL_TO_PASS`. Not observed in the pinned revision (checked
      against all 59,136 rows, not sampled), but upstream's own
      `gather.py` skips exactly this case when building the dataset
      (`n_f2p == 0`) except for a `.pr_*` exception this package already
      excludes for an unrelated reason (below) -- so a future revision
      could carry one, and an instance with no fail-to-pass test cannot
      express failure, only ever a vacuous, unconditional pass. Filtering
      here is a second, independent line of defense alongside the
      "Bug Patch" branch-shape assertion `taskset._TRAIN_GUARD_AND_REROOT`
      makes inside the container (see that constant's own docstring) --
      that one guards the branch upstream actually built; this one guards
      the row upstream published, which is the more direct fix for this
      specific failure mode and does not need a container to check.
    - A patch `_reverse_unified_diff` cannot reverse: it adds, deletes,
      renames or copies a file (see that function's own docstring), and
      measured against the top 20 images, 41 of 23,844 rows (0.17%) do
      exactly that -- every one of them a `.pr_<N>` instance, the "PR
      Mirroring" construction path `swe_adapter.get_test_cmd`'s own
      upstream docstring calls out as different from the synthetic
      bug-generation strategies (`func_basic`, `combine_*`, `lm_rewrite`)
      that make up the rest of the corpus.

    Both are excluded rather than silently mis-scored; logged once with
    their counts so the exclusion is visible rather than assumed away.
    """
    if num_images < 1:
        raise ValueError(f"num_images must be >= 1, got {num_images}")
    if max_test_count is not None and max_test_count < 1:
        raise ValueError(f"max_test_count must be >= 1 or None, got {max_test_count}")
    name, hf_split, revision = _SOURCES["train"]
    dataset = load_dataset(name, split=hf_split, revision=revision or None)
    selected = set(swesmith_image_rank()[:num_images])
    rows = []
    skipped_empty_f2p = 0
    skipped_unreversible = 0
    skipped_too_costly = 0
    for row in dataset:
        if row["image_name"] not in selected:
            continue
        fail_to_pass = _tests(row["FAIL_TO_PASS"])
        pass_to_pass = _tests(row["PASS_TO_PASS"])
        reason = _swesmith_row_defect(fail_to_pass, pass_to_pass, max_test_count)
        if reason == "empty_fail_to_pass":
            skipped_empty_f2p += 1
            continue
        if reason == "too_costly":
            skipped_too_costly += 1
            continue
        try:
            gold_patch = _reverse_unified_diff(row["patch"])
        except ValueError:
            skipped_unreversible += 1
            continue
        rows.append(
            SweRow(
                instance_id=row["instance_id"],
                repo=row["repo"],
                problem_statement=row["problem_statement"],
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                gold_patch=gold_patch,
                base_commit=f"origin/{row['instance_id']}~1",
                image=row["image_name"],
            )
        )
    total = skipped_empty_f2p + skipped_unreversible + skipped_too_costly + len(rows)
    if skipped_empty_f2p or skipped_unreversible or skipped_too_costly:
        logging.getLogger(__name__).warning(
            "load_swesmith_rows(num_images=%d, max_test_count=%s): excluded "
            "%d/%d rows with empty FAIL_TO_PASS, %d/%d rows whose patch "
            "adds/deletes/renames/copies a file, and %d/%d rows over the "
            "test-count cap",
            num_images,
            max_test_count,
            skipped_empty_f2p,
            total,
            skipped_unreversible,
            total,
            skipped_too_costly,
            total,
        )
    return tuple(rows)


# MiMo-V2.6-RL-oss's `code` config: 2,698 tasks across Python, Go, JS/TS,
# Ruby, PHP, Java, C++ and Rust, one Docker image per task. Pinned like every
# other source here.
_POLYGLOT_SOURCE = (
    "XiaomiMiMo/MiMo-V2.6-RL-oss",
    "code",
    "639865fd3374018d6cb29b9fb82dd531406fcf5f",
)

# The dataset's own `docker_image` column names a local tag
# (`format-code-task-001457:latest`); what is published and pullable is this
# repository, tagged with the instance id. Checked against Docker Hub: every
# instance id in the pinned revision has a tag of its own name there.
POLYGLOT_IMAGE_REPOSITORY = "xiaomimimo/mimo-v2.6-rl-oss"


@functools.lru_cache(maxsize=None)
def load_polyglot_rows() -> tuple[SweRow, ...]:
    """Every row of the pinned polyglot corpus, in the dataset's own order.

    What upstream publishes differs from both other corpora, and each
    difference is a field left empty here rather than filled with a guess:

    - No fix. There is no `patch` column at all: each task is a feature
      request or bug report plus *hidden* tests, and `gold_patch` stays
      empty. Where a reference exists, it lives inside the image instead
      (see `grading.py`'s polyglot notes and `tests/test_polyglot_goldens.py`).
    - No FAIL_TO_PASS/PASS_TO_PASS. The row's `test_command` runs
      `mimo_test_command.sh`, a script `test_patch` itself adds, and its exit
      status is the verdict: 0 passes, anything else fails.
    - No base commit. Each image already sits at the state the agent must
      start from, so `base_commit` is `"HEAD"`, resolved inside the box.
    - No repository name. The row carries none, and nothing grades by it.

    Checked against every row of the pinned revision, not sampled: every
    `test_patch` is non-empty and adds `mimo_test_command.sh`, and every
    instance id is unique.
    """
    name, config, revision = _POLYGLOT_SOURCE
    dataset = load_dataset(name, config, split="train", revision=revision)
    rows = []
    for row in dataset:
        instance = json.loads(row["extra_info"]["instance_json"])
        rows.append(
            SweRow(
                instance_id=instance["instance_id"],
                repo="",
                problem_statement=instance["problem_statement"],
                fail_to_pass=(),
                pass_to_pass=(),
                gold_patch="",
                base_commit="HEAD",
                test_patch=instance["test_patch"],
                image=f"{POLYGLOT_IMAGE_REPOSITORY}:{instance['instance_id']}",
                test_command=instance["test_command"],
                workdir=instance["cwd"],
            )
        )
    return tuple(rows)
