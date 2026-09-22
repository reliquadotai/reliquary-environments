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
import shlex

from swebench.harness.constants import (
    END_TEST_OUTPUT,
    LATEST,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
    TestStatus,
    USE_X86,
)
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.test_spec import TestSpec

from reliquary_swe.corpus import SweRow

# The Docker Hub namespace SWE-bench itself publishes prebuilt evaluation
# images under -- the default of swebench.harness.run_evaluation's own
# `--namespace` flag, not a guess.
_NAMESPACE = "swebench"


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
_ACTIVATE_TESTBED = "source /opt/miniconda3/bin/activate && conda activate testbed"


# Linux's per-argv-element cap (`MAX_ARG_STRLEN`, fs/exec.c), independent of
# the much larger total `ARG_MAX`. It binds any single argument of any
# process a runtime execs -- including `docker` itself on the host, since
# `verifiers.v1.runtimes.docker.DockerRuntime.run` passes each argv element
# straight to `asyncio.create_subprocess_exec("docker", ...)` with no
# handling for the `OSError` an argument this large raises. Measured
# directly: folding django__django-10097's combined FAIL_TO_PASS +
# PASS_TO_PASS list (1,870 tests, ~134 KB joined) into one argument and
# exec'ing it raises exactly that error before the container is ever
# reached. matplotlib__matplotlib-25122's list (2,488 tests) is ~265 KB --
# over twice this cap -- and it is not the only instance in the corpus this
# large.
_MAX_ARG_STRLEN = 128 * 1024


def tests_file_contents(tests: tuple[str, ...]) -> bytes:
    """The newline-separated bytes `test_command`'s argv reads back (via
    `xargs -d '\\n'`) from the path the caller writes them to.

    Owning the format here, next to the command that reads it, keeps the
    delimiter defined once rather than implicitly duplicated between a
    writer (grading, elsewhere) and this reader. An empty `tests` encodes to
    empty bytes on purpose: `test_command`'s `test -s` guard then refuses to
    run anything for a file grading forgot to populate, the same way it
    refuses a missing file.
    """
    return ("\n".join(tests) + "\n").encode() if tests else b""


def test_command(row: SweRow, tests_path: str) -> list[str]:
    """An argv running exactly the tests listed at `tests_path` inside the
    instance's image -- one per line, in the format `tests_file_contents`
    writes and this reads back.

    Test names travel through that file rather than through the command
    line: see `_MAX_ARG_STRLEN`'s note for why an inlined list is not safe
    for every instance in the corpus. `xargs` turns the file's lines back
    into individually short arguments to the real test runner, so no argv
    element this function returns grows with the size of the test list --
    only `tests_path` does, and that stays a small, fixed-size string no
    matter how many tests it names. Also wrapped in the same conda
    activation the image's own eval scripts use (see `_ACTIVATE_TESTBED`):
    the base interpreter's PATH has no `pytest`/`./tests/runtests.py` at all.

    Two failure modes are refused outright rather than left to whatever the
    runner does by default with no target tests: `test -s` refuses a
    missing or empty file, and `xargs -r` refuses to invoke the runner at
    all if it still ends up with zero arguments. Both make the command exit
    non-zero instead of silently grading the wrong thing -- calling a
    runner like django's `tests/runtests.py` with no test arguments runs
    its *entire* suite, not zero tests, which is the failure mode this
    guards against, not merely a symmetrical edge case.
    """
    prefix = shlex.join(shlex.split(_test_cmd_for(row.repo, row.version)))
    quoted_path = shlex.quote(tests_path)
    script = (
        f"{_ACTIVATE_TESTBED} && test -s {quoted_path} && "
        f'exec xargs -r -d "\\n" -a {quoted_path} -- {prefix}'
    )
    return ["bash", "-c", script]


def parse_results(row: SweRow, stdout: str) -> dict[str, str]:
    """Map each reported test name to PASSED / FAILED / ERROR / SKIPPED.

    Never raises. Unparseable output returns an empty map, which grading reads
    as "no test reported a pass" -- the correct reward for a run whose output we
    cannot trust.

    Mirrors `swebench.harness.grading.get_logs_eval`: a genuine eval run's
    output is only trustworthy between the `START_TEST_OUTPUT` /
    `END_TEST_OUTPUT` sentinel lines the eval script itself echoes (content
    outside them can be shell tracing, install noise, or -- if a submission
    prints its own fake status lines -- a forged result). Missing either
    sentinel means the run never produced trustworthy output at all.
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
    # status: `test_passed` (swebench/harness/grading.py:26-27) checks
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
