"""scripts/sandbox_images.py with a fake docker (nothing is run)."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parent.parent / "sandbox_images.py"
spec = importlib.util.spec_from_file_location("sandbox_images", PATH)
si = importlib.util.module_from_spec(spec)
spec.loader.exec_module(si)

A = "repo/a@sha256:" + "a" * 64
B = "repo/b@sha256:" + "b" * 64


class FakeDocker:
    def __init__(self, present=(), failing=(), missing_tools=()):
        self.present = set(present)
        self.failing = set(failing)
        self.missing_tools = set(missing_tools)
        self.commands = []

    def __call__(self, argv, **kwargs):
        self.commands.append(argv)
        image = argv[-1] if argv[-2] in ("pull", "inspect") else None
        if "inspect" in argv:
            return SimpleNamespace(returncode=0 if image in self.present else 1, stdout="", stderr="")
        if "pull" in argv:
            if image in self.failing:
                return SimpleNamespace(returncode=1, stdout="", stderr="denied")
            self.present.add(image)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "run" in argv:
            image = argv[argv.index("--entrypoint") + 2]
            return SimpleNamespace(returncode=3 if image in self.missing_tools else 0,
                                   stdout="missing: bash\n" if image in self.missing_tools else "",
                                   stderr="")
        raise AssertionError(argv)


def manifest(tmp_path, images, name="m.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"env": "e", "split": "s", "images": images}))
    return path


def test_load_images_merges_manifests_and_refuses_tags(tmp_path):
    assert si.load_images([manifest(tmp_path, [A, B]), manifest(tmp_path, [B], "n.json")]) == [A, B]
    with pytest.raises(ValueError, match="digest"):
        si.load_images([manifest(tmp_path, ["repo/a:latest"])])


def test_pull_skips_present_images_and_reports_failures():
    docker = FakeDocker(present={A}, failing={B})
    failed = si.pull([A, B], docker=["docker", "-H", "unix:///x.sock"], jobs=1, run=docker)
    assert failed == [B]
    pulls = [c for c in docker.commands if "pull" in c]
    assert pulls == [["docker", "-H", "unix:///x.sock", "pull", B]]


def test_check_runs_each_image_without_network_under_the_runtime():
    docker = FakeDocker(missing_tools={B})
    results = si.check([A, B], docker=["docker"], runtime="runsc", require=["git"], run=docker)
    assert [r["ok"] for r in results] == [True, False] and "bash" in results[1]["detail"]
    run = next(c for c in docker.commands if "run" in c)
    assert run[:6] == ["docker", "run", "--rm", "--network", "none", "--runtime"]
    assert "command -v git" in run[-1]


def test_approve_prints_the_gateway_setting():
    line = si.approve_line([B, A, A])
    assert line == "RELIQUARY_SANDBOX_EPISODE_IMAGES=" + json.dumps([A, B])


def test_check_records_each_images_last_result(tmp_path):
    record = tmp_path / "checked.json"
    m = manifest(tmp_path, [A, B])
    code = si.main(["check", str(m), "--record", str(record)], run=FakeDocker(missing_tools={B}))
    assert code == 1
    first = json.loads(record.read_text())["images"]
    assert first[A]["ok"] is True and first[B]["ok"] is False and first[A]["runtime"] == "runsc"
    # a later check of one image replaces only its entry
    assert si.main(["check", str(manifest(tmp_path, [B], "n.json")), "--record", str(record)],
                   run=FakeDocker()) == 0
    assert json.loads(record.read_text())["images"][B]["ok"] is True
    assert json.loads(record.read_text())["images"][A] == first[A]


def test_approve_needs_a_passing_last_check_for_every_image(tmp_path, capsys):
    record = tmp_path / "checked.json"
    m = manifest(tmp_path, [A, B])
    assert si.main(["approve", str(m), "--record", str(record)]) == 1  # never checked
    assert "check" in capsys.readouterr().err
    si.main(["check", str(manifest(tmp_path, [A], "a.json")), "--record", str(record)],
            run=FakeDocker())
    assert si.main(["approve", str(m), "--record", str(record)]) == 1  # B never checked
    assert B in capsys.readouterr().err
    si.main(["check", str(m), "--record", str(record)], run=FakeDocker(missing_tools={B}))
    assert si.main(["approve", str(m), "--record", str(record)]) == 1  # B's last check failed
    out = capsys.readouterr()
    assert B in out.err and "RELIQUARY_SANDBOX_EPISODE_IMAGES" not in out.out
    si.main(["check", str(m), "--record", str(record)], run=FakeDocker())
    capsys.readouterr()
    assert si.main(["approve", str(m), "--record", str(record)]) == 0
    assert capsys.readouterr().out.strip() == si.approve_line([A, B])
    si.main(["check", str(m), "--record", str(record)], run=FakeDocker(missing_tools={A}))
    assert si.main(["approve", str(m), "--record", str(record)]) == 1  # a newer failure wins


def test_build_tmax_pushes_to_the_given_registry(tmp_path):
    command = si.build_tmax_command("registry.example:5000/team", tmp_path)
    assert command[-2:] == ["--repository", "registry.example:5000/team/reliquary-tmax-base"]
    assert "tmax_validate.py" in " ".join(command)


def test_build_tmax_is_a_dry_run_without_push(tmp_path, capsys):
    calls = []
    code = si.main(["build-tmax", "--registry", "r.example/team", "--terminal-package", str(tmp_path)],
                   run=lambda *a, **k: calls.append(a))
    assert code == 0 and calls == []
    assert "reliquary-tmax-base" in capsys.readouterr().out


def test_build_tmax_pushes_only_with_registry_and_push(tmp_path):
    calls = []
    si.main(["build-tmax", "--registry", "r.example/team", "--push", "--terminal-package", str(tmp_path)],
            run=lambda argv, **k: calls.append(argv) or SimpleNamespace(returncode=0))
    assert len(calls) == 1 and calls[0][-2:] == ["--repository", "r.example/team/reliquary-tmax-base"]
    with pytest.raises(SystemExit):
        si.main(["build-tmax", "--push"])
