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

SPLITS = ("eval", "train")

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
# below): 20 images -> 23,844 tasks, ~45GB of images on disk. Configurable via
# `SweTasksetConfig.num_images`; declared, never auto-sized from local free
# disk -- see corpus.load_swesmith_rows.
DEFAULT_SWESMITH_IMAGES = 20

# The second, orthogonal selection dimension, alongside `DEFAULT_SWESMITH_IMAGES`:
# a declared cap on fail-to-pass + pass-to-pass test count, because SWE-smith's
# per-instance test count is heavily right-skewed (measured on the full pinned
# corpus: p50=415, p75=1151, p90=2657, p95=5060, max=22028) and that skew is a
# real per-instance cost difference for the subset of repos whose profile sets
# `min_testing=True` (their test command narrows to exactly these tests'
# files -- see swesmith_adapter.py). Set at the measured p95 rather than a more
# aggressive cut: seven real container gradings across the size range (the
# implementation report has the transcripts), including the single most
# expensive instance in the entire 59,136-row corpus
# (pandas-dev/pandas.pr_59615, 22,028 fail+pass-to-pass entries, 23,724 tests
# *actually* run), measured 3-40 seconds wall clock -- two orders of magnitude
# below a naive per-test estimate carried over from a different benchmark
# (SWE-bench Verified) and a different, per-test-heavier repo (sympy). Nothing
# in this package's own measurements justifies a cut as aggressive as p75; p95
# is kept anyway as a cheap backstop against a repository this package has not
# measured turning out as per-test-expensive as that one did. Excludes 631 of
# the top 20 images' 23,844 pre-exclusion rows (2.65%).
#
# For the other, larger share of the corpus -- repos whose profile leaves
# `min_testing` at its default `False` -- this cap does not bound cost at all:
# their test command always runs the whole suite regardless of any one
# instance's fail-to-pass/pass-to-pass counts, so cost there is a per-repository
# constant, not a per-instance one. Measured directly (same report): whole-suite
# runs for four such repositories in the top 20 images took 3-34 seconds, cheap
# enough that no additional bound was added for them.
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


@functools.lru_cache(maxsize=None)
def load_swesmith_rows(
    num_images: int = DEFAULT_SWESMITH_IMAGES,
    max_test_count: int | None = DEFAULT_SWESMITH_MAX_TEST_COUNT,
) -> tuple[SweRow, ...]:
    """The task set for the top `num_images` SWE-smith images by task count,
    additionally excluding any instance whose fail-to-pass + pass-to-pass
    test count exceeds `max_test_count` (see `DEFAULT_SWESMITH_MAX_TEST_COUNT`
    for why, and for the measurements behind the default). `None` disables
    the cap.

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

    A further, small, unconditional exclusion: `_reverse_unified_diff` cannot
    reverse a patch that adds, deletes, renames or copies a file (see its own
    docstring), and measured against the top 20 images, 41 of 23,844 rows
    (0.17%) do exactly that -- every one of them a `.pr_<N>` instance, the
    "PR Mirroring" construction path `swe_adapter.get_test_cmd`'s own upstream
    docstring calls out as different from the synthetic bug-generation
    strategies (`func_basic`, `combine_*`, `lm_rewrite`) that make up the
    rest of the corpus. Excluded rather than silently mis-scored; logged once
    with the count so the exclusion is visible rather than assumed away.
    """
    if num_images < 1:
        raise ValueError(f"num_images must be >= 1, got {num_images}")
    if max_test_count is not None and max_test_count < 1:
        raise ValueError(f"max_test_count must be >= 1 or None, got {max_test_count}")
    name, hf_split, revision = _SOURCES["train"]
    dataset = load_dataset(name, split=hf_split, revision=revision or None)
    selected = set(swesmith_image_rank()[:num_images])
    rows = []
    skipped_unreversible = 0
    skipped_too_costly = 0
    for row in dataset:
        if row["image_name"] not in selected:
            continue
        fail_to_pass = _tests(row["FAIL_TO_PASS"])
        pass_to_pass = _tests(row["PASS_TO_PASS"])
        if max_test_count is not None and len(fail_to_pass) + len(pass_to_pass) > max_test_count:
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
    total = skipped_unreversible + skipped_too_costly + len(rows)
    if skipped_unreversible or skipped_too_costly:
        logging.getLogger(__name__).warning(
            "load_swesmith_rows(num_images=%d, max_test_count=%s): excluded "
            "%d/%d rows whose patch adds/deletes/renames/copies a file, and "
            "%d/%d rows over the test-count cap",
            num_images,
            max_test_count,
            skipped_unreversible,
            total,
            skipped_too_costly,
            total,
        )
    return tuple(rows)
