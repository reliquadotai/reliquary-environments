"""The static selection stage, on synthetic inputs. No container, no network."""

from __future__ import annotations

import json

import pytest
from test_tmax import DEF, make_task

from reliquary_terminal import tmax, tmax_select

PACKAGES = """\
Package: python3
Version: 3.10.6-1~22.04

Package: python3-pip
Version: 22.0.2+dfsg-1ubuntu0.4

Package: ca-certificates
Version: 20230311ubuntu0.22.04.1

Package: git
Version: 1:2.34.1-1ubuntu1.11

Package: tesseract-ocr
Version: 4.1.1-2.1build1

Package: postfix
Version: 3.6.4-1ubuntu1.3
Provides: mail-transport-agent
Conflicts: mail-transport-agent

Package: exim4-daemon-light
Version: 4.95-4ubuntu2.5
Provides: mail-transport-agent
Conflicts: mail-transport-agent

Package: libfoo2
Version: 2.0-1
Breaks: libfoo-tools (<< 1.5)

Package: libfoo-tools
Version: 1.6-1

Package: netcat-openbsd
Version: 1.218-4ubuntu1
Provides: netcat

Package: libgl1-mesa-glx
Version: 23.0.4-0ubuntu1~22.04.1

Package: python3-yaml
Version: 5.4.1-1ubuntu1
"""


@pytest.fixture
def index():
    return tmax_select.AptIndex.parse([PACKAGES])


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("1.0", "1.0", 0),
        ("1.0~rc1", "1.0", -1),
        ("1:1.0", "2.0", 1),
        ("1.0-1", "1.0-1ubuntu1", -1),
        ("2.34.1-1ubuntu1.11", "2.34.1-1ubuntu1.9", 1),
        ("1.0a", "1.0", 1),
        ("1.0+dfsg", "1.0", 1),
    ],
)
def test_debian_version_order(a, b, expected):
    assert tmax_select.debian_compare(a, b) == expected
    assert tmax_select.debian_compare(b, a) == -expected


def test_apt_names_resolve_like_apt_get(index):
    assert index.resolve("git") == "git"
    assert index.resolve("netcat") == "netcat-openbsd"  # one provider
    assert index.resolve("mail-transport-agent") is None  # several providers
    assert index.resolve("no-such-package") is None


def test_apt_conflicts_direct_through_provides_and_versioned(index):
    assert index.conflict("postfix", "exim4-daemon-light")
    assert not index.conflict("libfoo2", "libfoo-tools")  # 1.6 is not << 1.5
    assert not index.conflict("git", "tesseract-ocr")


def test_plan_apt_drops_the_minority_side_of_a_conflict(index):
    plan = tmax_select.plan_apt(
        {
            "t1": ["postfix", "git"],
            "t2": ["postfix"],
            "t3": ["exim4-daemon-light"],
            "t4": ["no-such-package", "git"],
        },
        index,
    )
    assert plan.unknown == {"t4": "no-such-package"}
    assert set(plan.conflicts) == {"t3"}
    assert plan.left_out == ["exim4-daemon-light"]
    assert "postfix" in plan.packages and "git" in plan.packages
    assert {"python3", "python3-pip", "ca-certificates"} <= set(plan.packages)


def test_plan_pip_takes_the_majority_pin():
    plan = tmax_select.plan_pip(
        {
            "a": ["numpy==1.26.4", "pandas"],
            "b": ["numpy==1.26.4"],
            "c": ["numpy==2.0.0"],
            "d": ["numpy<2", "requests[socks]>=2"],
            "e": ["Flask==2.0.1"],
            "f": ["flask==3.0.0"],
        }
    )
    assert "numpy==1.26.4" in plan.requirements
    assert set(plan.conflicts) == {"c", "e"}  # flask tie: the higher pin wins
    assert "flask==3.0.0" in plan.requirements
    assert "pandas" in plan.requirements
    assert "requests[socks]>=2" in plan.requirements
    assert "pytest" in plan.requirements and "pytest-json-ctrf" in plan.requirements


def _words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_decontamination_rules():
    S = tmax_select
    code = _words("code", 30)  # 18 shingles with tb-a
    few = _words("few", 14)  # 2 shingles: too few for code overlap
    shared = _words("common", 13)  # in 3 TB tasks: boilerplate
    idiom = _words("idiom", 30)  # in tb-a, but also in 10 TMax tasks
    prose = _words("prose", 13)
    contamination = S.Contamination(
        {
            "tb-a": S.shingles(code) | S.shingles(few) | S.shingles(shared) | S.shingles(idiom),
            "tb-b": S.shingles(shared),
            "tb-c": S.shingles(shared),
        },
        {"tb-a": S.shingles(prose)},
    )
    texts = {
        "copied-code": ("x", f"a {code} b"),
        "few": ("x", few),
        "boilerplate": ("x", shared),
        "prose": (f"Please {prose}.", ""),
        "canary": ("x", f"# {S.CANARY}"),
        "clean": ("x", _words("other", 40)),
    }
    texts.update({f"idiom-{i}": ("x", idiom) for i in range(S.TMAX_BOILERPLATE_MIN_TASKS)})
    flagged = S.decontaminate(contamination, texts)
    assert set(flagged) == {"copied-code", "prose", "canary"}
    assert flagged["copied-code"] == "18 code shingles shared with tb-a"
    assert flagged["prose"] == "instruction overlaps tb-a"


def test_contamination_reads_terminal_bench_task_dirs(tmp_path):
    task = tmp_path / "tb" / "some-task"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text("")
    words = _words("unique", 30)
    (task / "instruction.md").write_text(words)
    (task / "tests" / "blob.bin").write_bytes(b"\0" + _words("binary", 30).encode())
    contamination = tmax_select.Contamination.from_dirs([tmp_path / "tb"])
    assert set(contamination.shared(words).values()) == {"some-task"}
    assert contamination.instruction_match(words) == "some-task"
    assert contamination.shared(_words("binary", 30)) == {}


def test_order_key_is_sha256_of_the_id():
    import hashlib

    assert tmax_select.order_key("task_1") == hashlib.sha256(b"task_1").hexdigest()


def _manifest(tmp_path, validation=None, contaminate=False):
    src = tmp_path / "src"
    make_task(src, "task_000001_aaaaaaaa")
    make_task(src, "task_000002_bbbbbbbb", success=False)
    make_task(src, "task_000003_cccccccc", definition=DEF.replace("mkdir -p /home/user", "git clone https://x.org/r.git"))
    make_task(src, "task_000004_dddddddd", definition=DEF.replace("tesseract-ocr", "no-such-package"))
    make_task(src, "task_000005_eeeeeeee", definition=DEF + "\n%startscript\n    nginx\n")
    source = tmax.Source(src)
    index = tmax_select.AptIndex.parse([PACKAGES])
    tb = {}
    if contaminate:
        tb["tb-task"] = tmax_select.shingles(tmax_select.tmax_text(source, "task_000001_aaaaaaaa"))
    return tmax_select.build_manifest(source, index, tmax_select.Contamination(tb), validation)


def test_manifest_records_every_task_and_its_reasons(tmp_path):
    manifest, base = _manifest(tmp_path)
    tasks = manifest["tasks"]
    assert len(tasks) == 5
    assert tasks["task_000001_aaaaaaaa"] == {"status": "pending"}
    assert tasks["task_000002_bbbbbbbb"]["reasons"] == ["no_successful_run"]
    assert tasks["task_000003_cccccccc"]["reasons"] == ["setup_network"]
    assert tasks["task_000004_dddddddd"]["reasons"] == ["apt_unknown"]
    assert tasks["task_000005_eeeeeeee"]["reasons"] == ["startscript"]
    assert list(tasks) == sorted(tasks, key=tmax_select.order_key)
    assert manifest["counts"]["status"] == {"excluded": 4, "pending": 1}
    assert manifest["base_image"] is None
    # The base carries what the one surviving task needs, plus the runner.
    assert "tesseract-ocr" in base["apt.txt"].split()
    assert "no-such-package" not in base["apt.txt"]
    assert "numpy==1.26.4" in base["requirements.txt"].split()
    assert "pytest-json-ctrf" in base["requirements.txt"].split()
    assert "FROM ubuntu:22.04" in base["Dockerfile"]
    # The dump is valid JSON that round-trips.
    assert json.loads(tmax_select.dump_manifest(manifest)) == manifest


def test_decontamination_excludes(tmp_path):
    manifest, _ = _manifest(tmp_path, contaminate=True)
    assert manifest["tasks"]["task_000001_aaaaaaaa"]["reasons"] == ["decontaminated"]


def test_validation_turns_pending_into_kept_or_excluded(tmp_path):
    image = "registry.example/tmax-base@sha256:" + "0" * 64
    kept = {"reasons": [], "run": 0, "protected": ["/home/user/.truth.json"], "hidden": [], "base_image": image}
    manifest, _ = _manifest(tmp_path, validation={"task_000001_aaaaaaaa": kept})
    assert manifest["tasks"]["task_000001_aaaaaaaa"] == {
        "status": "kept",
        "run": 0,
        "protected": ["/home/user/.truth.json"],
        "hidden": [],
    }
    assert manifest["base_image"] == image
    assert [tid for tid, _ in tmax_select.kept_tasks(manifest)] == ["task_000001_aaaaaaaa"]
    rejected = {"reasons": ["mutation_passes"], "base_image": image}
    manifest, _ = _manifest(tmp_path / "again", validation={"task_000001_aaaaaaaa": rejected})
    assert manifest["tasks"]["task_000001_aaaaaaaa"] == {"status": "excluded", "reasons": ["mutation_passes"]}
