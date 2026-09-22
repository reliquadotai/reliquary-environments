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
    swebench.harness.test_spec.test_spec.TestSpec -- its `instance_image_key`
        property is the actual image-naming logic; see `image_for` for why this
        module constructs one directly instead of going through the upstream
        `make_test_spec` factory.

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
from collections import Counter

from swebench.harness.constants import (
    END_TEST_OUTPUT,
    LATEST,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
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
    `make_test_spec` only reaches this decision after building the (repo,
    version) install scripts this module has no `version` to look up (see
    `image_for`).
    """
    if platform.machine() not in {"aarch64", "arm64"}:
        return "x86_64"
    return "x86_64" if instance_id in USE_X86 else "arm64"


def image_for(row: SweRow) -> str:
    """The prebuilt evaluation image holding this instance's repository.

    SWE-bench Verified publishes one already-built image per instance; its
    name and tag are `TestSpec.instance_image_key`. Going through the normal
    factory (`make_test_spec`) to reach that property would additionally
    require the repo's per-version install/eval scripts, which need a
    `version` -- a field `SweRow` deliberately does not carry (Task 1: corpus
    questions must be answerable without it). `instance_image_key` itself only
    reads `instance_id`, `arch`, `namespace` and the tag, so this builds the
    `TestSpec` directly with the rest left empty, reaching the real upstream
    naming code without inventing data this row does not have.
    """
    spec = TestSpec(
        instance_id=row.instance_id,
        repo=row.repo,
        version="",
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


def _test_cmd_for_repo(repo: str) -> str:
    """The shell test-runner invocation SWE-bench uses for `repo`.

    Upstream keys this by (repo, version) -- `MAP_REPO_VERSION_TO_SPECS[repo]
    [version]["test_cmd"]` -- but `SweRow` carries no `version`. Almost every
    repo uses one command across every version it has (the known exception is
    django/django's oldest entry, which drops `--parallel`); voting for the
    command that the repo's versions agree on most often is correct for every
    instance except that handful, and does not require picking one version
    arbitrarily.
    """
    commands = []
    for spec in MAP_REPO_VERSION_TO_SPECS[repo].values():
        cmd = spec["test_cmd"]
        # A few non-Python specs store a fallback list; grading itself takes
        # the last entry as the one that actually produced the log (see
        # swebench.harness.grading.get_logs_eval).
        commands.append(cmd[-1] if isinstance(cmd, list) else cmd)
    ((command, _count),) = Counter(commands).most_common(1)
    return command


def test_command(row: SweRow, tests: tuple[str, ...]) -> list[str]:
    """An argv running exactly `tests` inside the instance's image."""
    return [*shlex.split(_test_cmd_for_repo(row.repo)), *tests]


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
        return dict(parser(content, None))
    except Exception:
        # A parser that cannot make sense of this content is exactly the
        # "cannot trust it" case this function's contract exists for, not a
        # bug to propagate.
        return {}
