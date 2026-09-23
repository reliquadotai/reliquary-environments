"""The only module here that imports `swebench`.

Per-repository test output has no generic shape: each repository uses a
different framework and prints a different format. `swebench` already carries a
parser per repository and the naming scheme for the prebuilt evaluation images.
Wrapping both here means a version bump touches one file, and the rest of this
package depends on names we control.

Resolved swebench imports for the pinned version (`swebench==3.0.17`; see the
implementation plan, Task 2 Step 1, for how these were found):
    swebench.harness.constants.MAP_REPO_VERSION_TO_SPECS -- per (repo, version)
        install/test spec; ["test_cmd"] is the shell command that runs a
        repository's tests.
    swebench.harness.constants.USE_X86 -- instance ids that must still build for
        x86_64 even on an arm64 host (a handful of repos have no arm64 image).
    swebench.harness.constants.LATEST -- the tag prebuilt images are published
        under ("latest").
    swebench.harness.constants.START_TEST_OUTPUT / END_TEST_OUTPUT -- the
        sentinel lines a real eval run echoes around the test command's own
        output; grading (`swebench.harness.grading.get_logs_eval`) slices on
        them before handing the middle to a parser, and this module does the
        same.
    swebench.harness.log_parsers.MAP_REPO_TO_PARSER -- repo -> the function that
        turns one repository's raw test output into a {test name: status} map.
    swebench.harness.constants.TestStatus -- its `XFAIL` member is what an
        `@pytest.mark.xfail` test parses to; see `parse_results` for why this
        module normalizes it away rather than passing it through.
    swebench.harness.test_spec.test_spec.TestSpec -- `instance_image_key`
        reads only `instance_id`/`arch`/`namespace`/`instance_image_tag`; see
        `image_for` for why this module still constructs one directly rather
        than going through the upstream `make_test_spec` factory.
    swebench.harness.test_spec.python.get_test_directives -- derives the file
        paths (module labels, for django/django) a test command should run
        from an instance's `test_patch`; see `test_command` for why this
        module calls it rather than running FAIL_TO_PASS/PASS_TO_PASS entries
        directly, which is not what the official harness does either.
    swebench.harness.utils.get_modified_files -- the pre-patch ("a/" side)
        path of every file a diff touches, filtered to exclude "/dev/null"
        (a file the diff adds, which therefore does not exist yet at any
        commit to check out); see `get_modified_files` here and
        `grading._restore_from_test_patch` for why grading calls this rather
        than reading the "+++ b/" side of `test_patch` itself.

3.0.17 was chosen deliberately over the 4.x/5.x line: starting at 4.0.0,
swebench moved to a "task repo" model where `image`, `log_parser` and
`eval_script` are pre-computed fields the dataset itself must carry (see
`swebench.task.repo` in 5.0.2). `SweRow` -- and the raw
`princeton-nlp/SWE-bench_Verified` rows it wraps -- is shaped for the older,
still-published (repo, version) constants API that 3.0.17 is the last major
line to expose, so that is the line this module is coupled to.
"""

from __future__ import annotations

import platform
import re
import shlex

from swebench.harness.constants import (
    END_TEST_OUTPUT,
    FAIL_TO_PASS,
    LATEST,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_PASS,
    START_TEST_OUTPUT,
    TestStatus,
    USE_X86,
)
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.test_spec.test_spec import TestSpec
from swebench.harness.utils import get_modified_files as _get_modified_files

from reliquary_swe.corpus import SweRow

# The Docker Hub namespace SWE-bench itself publishes prebuilt evaluation
# images under -- the default of swebench.harness.run_evaluation's own
# `--namespace` flag, not a guess.
_NAMESPACE = "swebench"


def get_modified_files(patch: str) -> list[str]:
    """The path of every file `patch` touches that already existed before it
    -- upstream's own `get_modified_files`, re-exported so a caller never has
    to parse a diff's "+++ b/" side itself (see this module's docstring for
    why that side is the wrong one to restore-from-base with: it names paths
    a diff *adds*, which do not exist at any earlier commit to check out).
    """
    return _get_modified_files(patch)


def _arch(instance_id: str) -> str:
    """The architecture `instance_id`'s prebuilt image was published for.

    Mirrors `TestSpec`'s own `make_test_spec` factory exactly (see
    `swebench.harness.test_spec.test_spec`): every instance is x86_64 unless
    the host itself is arm64, in which case a named few (`USE_X86`) still have
    no arm64 image and stay on x86_64. Duplicated rather than called because
    `make_test_spec` reaches this decision only after building the repo's
    install/eval scripts first -- see `image_for` for why this module never
    calls that factory at all.
    """
    if platform.machine() not in {"aarch64", "arm64"}:
        return "x86_64"
    return "x86_64" if instance_id in USE_X86 else "arm64"


def image_for(row: SweRow) -> str:
    """The prebuilt evaluation image holding this instance's repository.

    SWE-bench Verified publishes one already-built image per instance; its
    name and tag are `TestSpec.instance_image_key`, which reads only
    `instance_id`, `arch`, `namespace` and the tag -- nothing version- or
    script-specific. Going through the normal factory (`make_test_spec`) to
    reach that same property would build the repo's install/env/eval scripts
    too, and building those makes real HTTP calls out to GitHub
    (`swebench.harness.test_spec.python.get_requirements`/
    `get_environment_yml`, both `requests.get` against
    raw.githubusercontent.com at a pinned commit) to fetch a
    requirements/environment file this function has no use for. Naming an
    already-built image should not depend on GitHub being reachable, so this
    builds the `TestSpec` directly instead, leaving every field
    `instance_image_key` does not read (the script lists, `language`,
    `docker_specs`) at an empty placeholder. `version` is filled in for
    accuracy now that `SweRow` carries it, even though the property itself
    ignores it.
    """
    spec = TestSpec(
        instance_id=row.instance_id,
        repo=row.repo,
        version=row.version,
        repo_script_list=[],
        eval_script_list=[],
        env_script_list=[],
        arch=_arch(row.instance_id),
        FAIL_TO_PASS=list(row.fail_to_pass),
        PASS_TO_PASS=list(row.pass_to_pass),
        language="python",
        docker_specs={},
        namespace=_NAMESPACE,
        instance_image_tag=LATEST,
    )
    return spec.instance_image_key


def _test_cmd_for(repo: str, version: str) -> str:
    """The shell test-runner invocation SWE-bench uses for `(repo, version)`.

    A direct `MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]` lookup --
    upstream keys the command by version because it genuinely varies by
    version for some repos (e.g. django/django's oldest entries drop
    `--parallel`). Guessing across versions would be silently wrong on
    exactly the instances that motivate keying by version in the first
    place, so a missing pair raises instead of falling back to a nearby one.
    """
    try:
        spec = MAP_REPO_VERSION_TO_SPECS[repo][version]
    except KeyError:
        raise KeyError(
            f"swebench has no test spec for (repo={repo!r}, version={version!r})"
        ) from None
    cmd = spec["test_cmd"]
    # A few non-Python specs store a fallback list; grading itself takes the
    # last entry as the one that actually produced the log (see
    # swebench.harness.grading.get_logs_eval).
    return cmd[-1] if isinstance(cmd, list) else cmd


# A leading `NAME=value` shell-assignment prefix on a test_cmd (sympy's
# `PYTHONWARNINGS='...' bin/test ...`) is not the command; skip past it to
# find the token that actually runs.
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def test_entrypoint(row: SweRow) -> str | None:
    """The repo-relative script `test_cmd` itself executes, if any.

    Checked across every (repo, version) pair the current corpus uses:
    django's `./tests/runtests.py` and sympy's `bin/test` are real, in-repo
    files a patch can rewrite; astropy/pylint/pytest/etc.'s bare `pytest` and
    sphinx's bare `tox` are not paths at all -- they resolve off PATH inside
    the testbed conda env, and `tox`'s own in-repo config (`tox.ini`) is a
    separate restoration concern, not this one.

    This is the sharper version of the conftest.py vector `grading.py`
    already closes: none of `test_patch`, FAIL_TO_PASS/PASS_TO_PASS, or the
    filename `conftest.py` names this file, so nothing restores it unless a
    caller asks for it by name specifically -- and on django/sympy, this is
    the file that actually runs the tests, not merely one pytest loads.
    """
    tokens = shlex.split(_test_cmd_for(row.repo, row.version))
    for token in tokens:
        if _ASSIGNMENT.match(token):
            continue
        if "/" not in token or token.startswith("/"):
            return None
        return token.removeprefix("./")
    return None


# django's `--settings=NAME` is the one argument (beyond the entry point
# itself) this corpus's test commands read that also names a real, in-repo
# file. Checked across all 20 django versions MAP_REPO_VERSION_TO_SPECS
# carries (9 of them are actually used by the corpus): every one that has
# this flag at all names the identical bare label `test_sqlite`, never a
# dotted one.
_SETTINGS_FLAG = re.compile(r"--settings=([\w.]+)")


def test_command_argument_paths(row: SweRow) -> list[str]:
    """Repo-relative paths `test_cmd`'s own arguments name, resolved
    relative to the entry point's directory -- currently just django's
    `--settings=` flag.

    The real rule this approximates: any repo-relative path or module label
    a test command's own arguments name is infrastructure the grader
    executes and must restore, exactly like its entry point.
    `--settings=test_sqlite` is `runtests.py`'s `DJANGO_SETTINGS_MODULE`,
    imported *inside the grading process itself* before a single test runs
    -- confirmed real, not hypothetical: a settings module rewritten to
    print a fake status line per name and call `os._exit(0)` at import is
    the identical bypass as rewriting `runtests.py` directly, on the same
    231 instances, and is named by no `test_patch` in the corpus, no
    `conftest.py`, and nothing `test_entrypoint` returns.

    Deliberately targeted, not general: this recognizes exactly one flag
    shape and resolves a bare module label (`test_sqlite` -> `test_sqlite.py`
    next to the entry point) -- never observed with a dot in this corpus, so
    a multi-segment label (`a.b` -> `a/b.py`) is not handled. A different
    runner's own equivalent flag, or an argument naming a directory rather
    than a single module, would slip through this silently; nothing else in
    the current 12-repo corpus needs it (checked: only django's `test_cmd`
    contains `--settings=` at all).
    """
    entrypoint = test_entrypoint(row)
    if entrypoint is None:
        return []
    match = _SETTINGS_FLAG.search(_test_cmd_for(row.repo, row.version))
    if match is None:
        return []
    directory = entrypoint.rsplit("/", 1)[0] if "/" in entrypoint else ""
    relative = match.group(1).replace(".", "/") + ".py"
    return [f"{directory}/{relative}" if directory else relative]


def pytest_reporting_fixup(row: SweRow) -> list[str] | None:
    """A one-time command grading must run before the test command, when
    the prebuilt image does not already produce output `parse_results` can
    read -- currently just sphinx/sphinx's own `tox.ini`.

    sphinx's `test_cmd` (`tox --current-env -epy39 -v --`) reads its actual
    pytest invocation from `tox.ini`'s own `[testenv] commands=`, which is a
    bare `pytest`, no `-rA` and no equivalent. Confirmed by hand: with no
    named PASSED/FAILED lines anywhere in its output, `parse_results`
    returns `{}` regardless of outcome -- even the *gold* patch scores 0 on
    sphinx-doc__sphinx-8595 without this fixup, which is the same silent-zero
    shape as this module's other defects, just upstream of restoration
    rather than inside it.

    Upstream's own image builder papers over exactly this with a
    `pre_install` step (`MAP_REPO_VERSION_TO_SPECS[repo][version]["pre_install"]`)
    that `sed`s `-rA` into `tox.ini` while building the evaluation image.
    That step is not baked into the prebuilt image on Docker Hub (confirmed:
    `tox.ini` ships with plain `pytest`, no `-rA`) -- staler than this
    package's `pre_install` reads, or never applied to the published image in
    the first place. Grading reproduces only this one, specific,
    load-bearing line rather than replaying every `pre_install` step
    generically: the rest are dependency/environment setup the prebuilt
    image has already done, and blindly rerunning them (several need
    network, which grading's box does not have) would be a new defect, not
    a fix for this one.
    """
    for step in MAP_REPO_VERSION_TO_SPECS.get(row.repo, {}).get(row.version, {}).get(
        "pre_install", []
    ):
        if "-rA" in step and "tox.ini" in step:
            return shlex.split(step)
    return None


# `swebench`'s own eval images never put a repository's test runner on PATH
# for the base interpreter -- only the `testbed` conda environment has it.
# Verified by hand in two images with unrelated test runners: `pytest` is
# missing from PATH on `swebench/sweb.eval.x86_64.astropy_1776_astropy-12907`
# until `conda activate testbed`, and the same holds for
# `swebench/sweb.eval.x86_64.django_1776_django-10097` (whose test_cmd is its
# own `tests/runtests.py`, not pytest). "testbed" is not a per-repo guess: it
# is a literal, unconditional constant in the pinned swebench's own image
# builder (`swebench.harness.test_spec.test_spec.make_test_spec`, which every
# instance goes through regardless of repo or language), and the same
# activation line is what its generated eval scripts use
# (`swebench.harness.test_spec.python`, e.g. `make_eval_script_list`).
#
# `export LC_ALL=C.UTF-8` is prepended for the same reason `conda activate`
# is: this runs as `bash -c "( ... ) 2>&1"`, a non-login, non-interactive
# shell that never sources `/etc/profile.d/01-locale-fix.sh` -- the image
# ships a correct locale fixup, but nothing here was invoking it. Without
# it, the container's default POSIX/C locale makes any non-ASCII byte a
# repo's own tooling writes to stdout/stderr (django's management commands
# print "Creating tables..." -- yes, an actual U+2026 ellipsis) raise
# UnicodeEncodeError and abort the whole test run before any test result is
# ever printed. `parse_results` then finds neither sentinel and returns
# `{}`, which grades identically to "every test failed": reward 0, silently,
# for gold and empty patches alike -- the same silent-zero shape as this
# module's other defects. Set here, not in grading.py, because this is the
# one place that owns the shell string `test_command` hands to `runtime.run`;
# grading.py only executes an argv it does not construct.
_ACTIVATE_TESTBED = (
    "export LC_ALL=C.UTF-8 && "
    "source /opt/miniconda3/bin/activate && conda activate testbed"
)


def test_command(row: SweRow) -> list[str]:
    """An argv running every test file (module label, for django/django)
    that `row.test_patch` touches, inside the instance's image.

    Deliberately takes no test-name argument. FAIL_TO_PASS and PASS_TO_PASS
    are lookup keys for scoring the already-parsed {test name: status} map,
    never command-line arguments -- that is what the official harness itself
    does (`swebench.harness.test_spec.python.make_eval_script_list_py`:
    `test_command = " ".join([test_cmd, *get_test_directives(instance)])`,
    with the individual FAIL_TO_PASS/PASS_TO_PASS names looked up afterward
    in grading, not passed here). An earlier version of this function ran
    FAIL_TO_PASS/PASS_TO_PASS entries directly and hit two problems that
    both trace back to that one divergence from upstream, not to each
    other: django/django's entries are unittest `str()` reprs its own
    runner will not accept as a command-line label at all, and enough
    entries corpus-wide are malformed or environment-drifted that direct
    invocation would score differently -- and therefore incomparably --
    from every published SWE-bench Verified number. Running directives
    instead sidesteps both by construction and matches upstream exactly,
    because this calls upstream's own `get_test_directives` rather than
    reimplementing its regex and its django-specific path transform.

    A handful of file paths also never approach the 128 KiB single
    argv-element cap that inlining thousands of test names did on the
    corpus's largest instances -- a side effect of matching upstream, not a
    mechanism this function has to maintain itself (see
    `test_no_argv_element_approaches_the_kernel_argument_length_cap` in
    test_adapter.py, which still pins it).

    The whole thing is wrapped in `( ... ) 2>&1`: `parse_results` (see its
    own docstring) treats "the output" as one stream, mirroring upstream's
    `get_logs_eval`, which slices a single combined log the outer harness
    captured -- upstream never keeps stdout and stderr apart either.
    Several repos' test runners genuinely need this merge to be usable at
    all: django's own `unittest`-based runner (confirmed by hand, running
    this exact command) prints every per-test result line, and its final
    "Ran N tests" summary, to stderr -- a caller that reads only `.stdout`
    from a runtime's `ProgramResult` would see nothing but database setup
    and teardown noise and parse an empty, silently-wrong result.
    """
    directives = get_test_directives({"repo": row.repo, "test_patch": row.test_patch})
    argv = [*shlex.split(_test_cmd_for(row.repo, row.version)), *directives]
    return ["bash", "-c", f"( {_ACTIVATE_TESTBED} && {shlex.join(argv)} ) 2>&1"]


def wrap_test_output(stdout: str) -> str:
    """Bound raw `test_command` output in the sentinels `parse_results` looks
    for, the way a real swebench-generated eval script echoes them around its
    own test command -- `test_command` here runs only the bare invocation, not
    that full generated script, so nothing else produces the boundary
    `parse_results` requires. A caller that skips this gets an empty map back
    from every real run, silently -- the same silent-zero shape as the conda
    and argv defects this module's docstring records.

    Not a trust boundary against the process under test. Upstream's own
    sentinels mean something because upstream's *eval script* echoes them,
    never the code being tested -- content the code under test prints cannot
    land between them. This function wraps `test_command`'s *entire* captured
    stream unconditionally, every byte of it, including anything a
    monkeypatched test process printed to imitate a real status line. So the
    sentinel check `parse_results` does is satisfied by construction here; it
    excludes nothing a submission wrote, and must not be read as doing so.
    The actual defense against a forged test process is restoring, before
    this ever runs, every file that could install one (see
    `grading._restore_from_test_patch`) -- this function has no part in that.
    """
    return f"{START_TEST_OUTPUT}\n{stdout}\n{END_TEST_OUTPUT}\n"


def parse_results(row: SweRow, stdout: str) -> dict[str, str]:
    """Map each reported test name to PASSED / FAILED / ERROR / SKIPPED.

    Never raises. Unparseable output returns an empty map, which grading reads
    as "no test reported a pass" -- the correct reward for a run whose output we
    cannot trust.

    Mirrors `swebench.harness.grading.get_logs_eval`'s slicing contract:
    read only between the `START_TEST_OUTPUT`/`END_TEST_OUTPUT` sentinel
    lines. Upstream's own sentinel check is a real trust boundary, because
    upstream's eval script -- never the code under test -- is what echoes
    them. Called through `swe_adapter.wrap_test_output` (see its docstring),
    it is NOT one here: those sentinels are injected unconditionally around
    the whole captured stream, so this reproduces upstream's parsing
    contract without reproducing what made it trustworthy there. It cannot
    distinguish real test-runner output from anything the process under test
    printed to imitate it; the sole real defense against that is not letting
    a forged test process exist in the first place (restoring every path
    that could install one before this ever runs). Missing either sentinel
    still means no output was ever produced to parse at all.
    """
    parser = MAP_REPO_TO_PARSER.get(row.repo)
    if parser is None or START_TEST_OUTPUT not in stdout or END_TEST_OUTPUT not in stdout:
        return {}
    content = stdout.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    try:
        status_map = dict(parser(content, None))
    except Exception:
        # A parser that cannot make sense of this content is exactly the
        # "cannot trust it" case this function's contract exists for, not a
        # bug to propagate.
        return {}
    # swebench's own resolution semantics treat XFAIL as a pass, not a fifth
    # status: `test_passed` (swebench/harness/grading.py:27-28) checks
    # `sm[case] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]`. A
    # pytest-family parser emits "XFAIL" verbatim for an
    # `@pytest.mark.xfail` test, several Verified repos use that marker, and
    # without this normalization a genuinely resolved xfail test would fail
    # grading's exact `== "PASSED"` check and score a resolved instance as a
    # failure -- silently, on real corpus data. Normalizing here keeps the
    # four-value contract this function documents rather than adding XFAIL
    # as a fifth value grading would then also have to know about.
    return {
        name: "PASSED" if status == TestStatus.XFAIL.value else status
        for name, status in status_map.items()
    }
