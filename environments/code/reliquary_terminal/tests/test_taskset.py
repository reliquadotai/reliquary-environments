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


def _guard(tmp_path):
    import importlib.util

    task_dir = taskset.materialize_tests(taskset.load_train_rows()[0], tmp_path / "cache")
    spec = importlib.util.spec_from_file_location("guard", task_dir / "tests" / taskset.GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard


@pytest.mark.parametrize("target", ["real", "/tmp/elsewhere/conftest.py", "../hooks/x.py", "dir"])
def test_the_guard_refuses_dangerous_names_that_are_symlinks(tmp_path, target):
    """The shipped guard skipped symlinks: `conftest.py -> ../hooks/x.py` planted a
    pytest hook it never saw. Dangling or not, to a file or not, a planted link with
    a dangerous name is refused like a planted file."""
    guard = _guard(tmp_path)
    app = tmp_path / "app"
    (app / "sub").mkdir(parents=True)
    (app / "real").write_text("import os\n")
    (app / "dir").mkdir()
    (app / "sub" / "conftest.py").symlink_to(app / target if target in ("real", "dir") else target)
    assert guard.scan(None, app=app) == "planted_interpreter_hook:sub/conftest.py"
    (app / "sub" / "conftest.py").unlink()
    assert guard.scan(None, app=app) is None


def test_the_guard_still_refuses_planted_files_and_accepts_listed_ones(tmp_path):
    guard = _guard(tmp_path)
    app = tmp_path / "app"
    app.mkdir()
    (app / "conftest.py").write_text("x = 1\n")
    assert guard.scan(None, app=app) == "planted_interpreter_hook:conftest.py"
    digest = guard.sha256(app / "conftest.py")
    assert guard.scan({"files": {"conftest.py": digest}}, app=app) is None


def test_materialize_replaces_a_cache_dir_that_disagrees_with_the_row(tmp_path):
    row = taskset.load_train_rows()[0]
    task_dir = taskset.materialize_tests(row, tmp_path)
    test_sh = task_dir / "tests" / "test.sh"
    pristine = test_sh.read_bytes()
    test_sh.write_bytes(b"printf '1\\n' > /logs/verifier/reward.txt\n")
    (task_dir / "tests" / "extra.py").write_text("planted\n")
    assert taskset.materialize_tests(row, tmp_path) == task_dir
    assert test_sh.read_bytes() == pristine
    assert not (task_dir / "tests" / "extra.py").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [row["instance_id"]]


def test_materialize_is_safe_when_processes_race(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    row = taskset.load_train_rows()[0]
    with ThreadPoolExecutor(8) as pool:
        dirs = list(pool.map(lambda _: taskset.materialize_tests(row, tmp_path), range(16)))
    assert set(dirs) == {tmp_path / row["instance_id"]}
    assert sorted(p.name for p in tmp_path.iterdir()) == [row["instance_id"]]
    assert (dirs[0] / "tests" / "test.sh").is_file()
