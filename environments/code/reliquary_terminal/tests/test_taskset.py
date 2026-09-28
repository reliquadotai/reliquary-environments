"""Loading, without a container."""

from __future__ import annotations

import pytest
import verifiers.v1 as vf

from reliquary_terminal import taskset


def _config(**kwargs):
    return vf.taskset_config_type("reliquary-terminal")(id="reliquary-terminal", **kwargs)


def test_the_split_has_no_default():
    with pytest.raises(ValueError, match="split is required"):
        list(vf.load_taskset(_config()))


def test_eval_is_terminal_bench_pinned_by_digest():
    assert _config(split="eval").dataset == taskset.EVAL_DATASET
    assert "@sha256:" in taskset.EVAL_DATASET


def test_eval_reads_the_images_own_workdir_where_task_toml_declares_none():
    tasks = {t.data.name: t for t in vf.load_taskset(_config(split="eval"))}
    assert len(tasks) == 89
    moved = {name: t.data.workdir for name, t in tasks.items() if t.data.workdir != "/app"}
    assert moved == {
        "terminal-bench/fix-git": "/app/personal-site",
        "terminal-bench/prove-plus-comm": "/workspace",
        "terminal-bench/sanitize-git-repo": "/app/dclm",
    }
    # Graded as it ships: in the agent's own box.
    assert all(t.data.verifier is None for t in tasks.values())


def test_train_is_the_64_terminal_tasks_graded_in_a_separate_box():
    tasks = list(vf.load_taskset(_config(split="train")))
    assert len(tasks) == 64
    for t in tasks:
        assert t.data.image.startswith(f"{taskset.TRAIN_IMAGE_REPOSITORY}:general-agent-env-")
        assert t.data.verifier is not None
        assert t.data.verifier.workdir == "/"
        assert t.data.verifier.network_allow == []
        assert t.data.network_allow == []
        assert [a.source for a in t.data.artifacts] == ["/app"]


def test_train_tests_never_ship_in_the_prompt():
    for t in vf.load_taskset(_config(split="train")):
        assert "anti_hack_guard" not in t.data.prompt


def test_materialize_is_idempotent(tmp_path):
    row = taskset.load_train_rows()[0]
    first = taskset.materialize_tests(row, tmp_path)
    before = sorted(p.relative_to(first) for p in first.rglob("*"))
    second = taskset.materialize_tests(row, tmp_path)
    assert first == second
    assert sorted(p.relative_to(second) for p in second.rglob("*")) == before
    assert (first / "tests" / "test.sh").is_file()


def test_image_workdir_takes_the_last_declaration(tmp_path):
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "Dockerfile").write_text(
        "FROM x\nWORKDIR /app\nRUN true\nWORKDIR /app/sub\n"
    )
    assert taskset.image_workdir(tmp_path) == "/app/sub"
    assert taskset.image_workdir(tmp_path / "missing") is None
