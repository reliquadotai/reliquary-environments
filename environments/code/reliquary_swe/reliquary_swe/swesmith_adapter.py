"""The only module here that imports `swesmith`.

SWE-smith ships its own per-repository knowledge -- what command runs a
repository's tests, and how to parse that command's output -- as
`swesmith.profiles.registry`, a global map from repo/mirror name to a
`RepoProfile` subclass. Reusing it is the point: the task brief that started
this module says so directly ("the swesmith package may ship per-repo
profiles, and reusing theirs beats inventing ours"), and duplicating a few
hundred hand-written test commands would be exactly the kind of invention
this repository's other environments avoid.

Ground truth this module leans on, verified against real containers rather
than assumed (see the implementation report for the transcripts):

- Every SWE-smith image checks out, by default, a clean `main` branch. Each
  instance is its own branch, built as `main` plus two commits: "Bug Patch"
  (the dataset's own `patch` column, applied to source only) then "Remove
  F2P Tests" (deletes the fail-to-pass tests' own files outright). `~1` from
  the branch tip is therefore the state a rollout must start from: bug
  present, fail-to-pass *and* pass-to-pass test files both still pristine.
  This is `corpus.SweRow.base_commit` for a SWE-smith row
  (`f"origin/{instance_id}~1"`) -- computed in corpus.py, not here, since it
  is pure string formatting and corpus.py must stay container-free.
- `registry.get_from_inst({"repo": row.repo, ...})` resolves a `RepoProfile`
  keyed on `repo` alone (SWE-smith's `repo` column, e.g.
  `"swesmith/oauthlib__oauthlib.1fd52536"`, already matches the registry's
  own mirror-name key format byte for byte -- no transformation needed).
  This is this module's answer to the design spec's "no version -> a second
  resolution path is needed": SWE-smith keys by `repo` where SWE-bench keys
  by `(repo, version)`.
- `RepoProfile.image_name` is NOT `row.image`: it computes the org as
  `swesmith.constants.ORG_NAME_DH` ("swebench"), while the dataset's own
  `image_name` column -- what is actually published and pullable -- uses
  `jyangballin`. Confirmed by direct comparison. This module never calls
  `.image_name`; `corpus.SweRow.image` carries the dataset's own value
  instead, which is the concrete evidence behind the design spec's "the
  image is given, not derived."
- Not every SWE-smith repository is Python (92 of the pinned corpus's 222
  images are Go, PHP, Java or Rust -- checked against every repo in the
  pinned revision, not sampled). This module only knows how to run Python
  profiles. `ensure_python_profile` is the fail-loud guard `taskset.py` calls
  once per distinct repo when a taskset loads, so an `N` that reaches a
  non-Python image raises at load time with a clear message instead of
  silently mis-scoring every rollout for that repo. Checked and NOT
  triggered by any of this package's default `N=20` (nor by `N` up to 30);
  first triggered at `N=31` (`caddyserver/caddy`, Go).
"""

from __future__ import annotations

from swesmith.profiles import registry
from swesmith.profiles.python import PythonProfile

from reliquary_swe.corpus import SweRow
from reliquary_swe.swe_adapter import FAIL_TO_PASS, PASS_TO_PASS, TestStatus

# `conda activate testbed` never needs a separate locale export the way
# swe_adapter's `_ACTIVATE_TESTBED` does for SWE-bench Verified: checked by
# hand inside a pulled SWE-smith image, `python -c "import locale; print(...)"`
# already reports UTF-8 with no `LC_ALL` set, unlike the SWE-bench Verified
# images the sibling module's own comment documents. Not re-derived here in
# case a different repository's image disagrees -- see the implementation
# report's "residual risk" list.


def _instance(row: SweRow) -> dict[str, object]:
    """The dict shape every `swesmith.profiles` call in this module expects:
    the subset of SWE-smith's own instance schema those calls actually read.

    Deliberately never includes `KEY_PATCH` ("patch"). `get_test_cmd`'s own
    branching (`swesmith/profiles/base.py`) checks
    `self.min_testing and FAIL_TO_PASS in instance` *before* it ever checks
    for a patch key, and this dict always sets `FAIL_TO_PASS` (as a key,
    even when the value is an empty list), so that branch always returns
    first for every `min_testing=True` profile this package runs today --
    the patch-requiring branch below it is unreachable from here. It stays
    unreachable only as long as `KEY_PATCH` stays out of this dict: adding
    it (e.g. to "help" by passing `row.gold_patch`) would route a
    `min_testing=True` profile with no `FAIL_TO_PASS` files derived into
    `_get_cached_test_paths()` -> `self.clone()` -- a live GitHub clone
    attempted *at grading time*, inside a box whose whole `network_allow=[]`
    point is that no such thing can succeed -- and even where it could
    reach the network, `row.gold_patch` is the *fix* where upstream's own
    code expects the bug-introducing patch, a second, independent way the
    same one-line change would be wrong. `test_ensure_no_patch_key_in_
    swesmith_instance_dict` in `tests/test_swesmith_adapter.py` pins this.
    """
    return {
        "instance_id": row.instance_id,
        "repo": row.repo,
        FAIL_TO_PASS: list(row.fail_to_pass),
        PASS_TO_PASS: list(row.pass_to_pass),
    }


def ensure_python_profile(repo: str) -> None:
    """Raise with a clear message if `repo`'s profile is not Python.

    Called once per distinct repo when a "train" taskset loads (`taskset.py`),
    not per row: `registry.get` is a cheap in-memory lookup, but there is no
    reason to repeat it thousands of times for one repo's worth of rows.
    """
    profile = registry.get(repo)
    if not isinstance(profile, PythonProfile):
        raise ValueError(
            f"reliquary_swe.swesmith_adapter only supports Python repositories; "
            f"{repo!r} resolved to {type(profile).__name__}, "
            f"a {type(profile).__mro__[1].__name__}"
        )


def test_command(row: SweRow) -> list[str]:
    """An argv running `row`'s own fail-to-pass and pass-to-pass tests (or,
    for a profile that has not opted into `min_testing`, its whole suite --
    upstream's own default; see this module's docstring on reusing theirs).

    Deliberately calls `get_test_cmd` with `f2p_only=False`: grading needs
    pass-to-pass verified too, exactly like the SWE-bench Verified path.
    """
    profile = registry.get(row.repo)
    cmd, _ = profile.get_test_cmd(_instance(row), f2p_only=False)
    return ["bash", "-c", cmd]


def test_files(row: SweRow) -> list[str]:
    """Every file `row`'s fail-to-pass or pass-to-pass tests live in --
    `grading.py`'s restoration strategy for a corpus with no `test_patch`:
    make these files be what they are at `base_commit`, regardless of what
    the agent's patch did to them. Matches SWE-smith's own anti-tamper step
    in `run_patch_in_container` exactly (`git checkout -- {f2p+p2p files}`),
    just executed one path at a time (see `grading._checkout`'s own docstring
    for why a batched checkout is the wrong call here too).
    """
    profile = registry.get(row.repo)
    f2p_files, p2p_files = profile.get_test_files(_instance(row))
    seen: dict[str, None] = {}
    for path in (*f2p_files, *p2p_files):
        seen[str(path)] = None
    return list(seen)


def parse_results(row: SweRow, stdout: str) -> dict[str, str]:
    """Map each reported test name to PASSED / FAILED / ERROR / SKIPPED.

    Never raises -- unparseable output returns an empty map, the same
    "cannot trust it, so no reward" contract `swe_adapter.parse_results`
    documents. No sentinel slicing here: unlike SWE-bench's `get_logs_eval`,
    SWE-smith's own `log_parser` implementations (`swesmith/profiles/python.py`)
    scan every line of the raw log directly with no start/end markers, so
    there is nothing upstream to reproduce.
    """
    profile = registry.get(row.repo)
    try:
        status_map = profile.log_parser(stdout)
    except Exception:
        return {}
    # Same XFAIL normalization as swe_adapter.parse_results, and for the
    # same reason: swebench.harness.grading.test_passed treats XFAIL as a
    # pass, and swesmith's own log_parser (which iterates every TestStatus
    # member, XFAIL included) can emit it verbatim.
    return {
        name: "PASSED" if status == TestStatus.XFAIL.value else status
        for name, status in status_map.items()
    }


__all__ = [
    "ensure_python_profile",
    "parse_results",
    "test_command",
    "test_files",
]
