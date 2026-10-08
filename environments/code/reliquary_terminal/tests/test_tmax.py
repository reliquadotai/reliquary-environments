"""The TMax converter, on synthetic tasks shaped like the real ones. No
container, no network."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from reliquary_terminal import tmax

DEF = """Bootstrap: docker
From: ubuntu:22.04

%files
    /gpfs/x/tasks_5k/{tid}/fixtures/image.png /app/fixtures/image.png

%post
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y python3 python3-pip git \\
    tesseract-ocr && rm -rf /var/lib/apt/lists/*
pip3 install pytest "numpy==1.26.4" || true
useradd -m -s /bin/bash user || true
mkdir -p /home/user
cat << 'EOF' > /home/user/notes.txt
apt-get install -y not-a-command-here
EOF
echo 42 > /home/user/.truth.json
chmod -R 777 /home/user

%environment
export LC_ALL=C.UTF-8
export PATH=/opt/tool/bin:$PATH
"""


def make_task(root: Path, tid: str = "task_000001_abcdef12", definition: str = DEF, success: bool = True) -> str:
    d = root / tid
    (d / "fixtures").mkdir(parents=True)
    (d / "fixtures" / "image.png").write_bytes(b"\x89PNG")
    (d / "container.def").write_text(definition.format(tid=tid))
    (d / "task.json").write_text(json.dumps({"name": tid, "description": "  Put 42 in /home/user/out.txt.\n", "domain": "debugging"}))
    (d / "test_final_state.py").write_text(
        "def test_out():\n    assert open('/home/user/out.txt').read().strip() == open('/home/user/.truth.json').read().strip()\n"
    )
    (d / "test_initial_state.py").write_text("def test_nothing():\n    pass\n")
    (d / "solutions").mkdir()
    (d / "solutions" / "summary.json").write_text("{}")
    calls = [
        {"function": {"name": "bash", "arguments": json.dumps({"command": "cd /home/user"})}},
        {"function": {"name": "bash", "arguments": json.dumps({"command": "cat .truth.json > out.txt"})}},
        {"function": {"name": "bash", "arguments": json.dumps({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})}},
    ]
    results = [
        {"success": False, "messages": [{"role": "assistant", "tool_calls": calls[:1]}]},
        {"success": success, "messages": [{"role": "assistant", "tool_calls": calls}]},
    ]
    (d / "solutions" / "gemini_gemini-3-flash-preview_summary.json").write_text(
        json.dumps({"num_runs": 2, "num_success": int(success), "results": results})
    )
    return tid


def test_definition_sections():
    d = tmax.parse_definition(DEF.format(tid="t"))
    assert d.files == [("/gpfs/x/tasks_5k/t/fixtures/image.png", "/app/fixtures/image.png")]
    assert "useradd" in d.post and "%environment" not in d.post
    assert "LC_ALL" in d.environment
    assert d.startscript == "" and d.other_sections == []


def test_install_half_moves_to_the_base_and_the_data_half_stays():
    split = tmax.split_post(tmax.parse_definition(DEF.format(tid="t")).post)
    assert split.problems == {}
    assert split.apt == ["python3", "python3-pip", "git", "tesseract-ocr"]
    assert split.pip == ["pytest", "numpy==1.26.4"]
    data = split.data
    assert "useradd -m -s /bin/bash user || true" in data
    assert "echo 42 > /home/user/.truth.json" in data
    # Heredoc bodies are file content, never installs, and kept verbatim.
    assert "apt-get install -y not-a-command-here" in data
    executable = [line for line in data.splitlines() if not line.startswith("#")]
    assert not any(line.startswith(("apt-get update", "pip3 install")) for line in executable)


@pytest.mark.parametrize(
    "line,code",
    [
        ("cd /app/x && pip3 install -e .", "install_unparsed"),
        ("/home/user/venv/bin/pip install requests", "install_unparsed"),
        ("pip3 install -r requirements.txt", "install_unparsed"),
        ("for p in a b; do pip3 install $p; done", "install_unparsed"),
        ("apt-get install -y $PKGS", "install_unparsed"),
        ("yes | apt-get install foo", "install_unparsed"),
        ("curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y", "setup_network"),
        ("wget -q https://example.org/x.tar.gz", "setup_network"),
        ("git clone https://github.com/a/b.git /app/b", "setup_network"),
        ("go get github.com/mattn/go-sqlite3", "setup_network"),
        ("cd /app && npm install", "setup_network"),
        ("cat << 'EOF' > /.singularity.d/env/99-x.sh", "apptainer_only_path"),
    ],
)
def test_lines_the_converter_refuses(line, code):
    split = tmax.split_post(line + "\nEOF\n" if "EOF" in line else line + "\n")
    assert code in split.problems


@pytest.mark.parametrize(
    "line",
    [
        "curl -s http://localhost:8080/health",
        "git clone /home/user/repo /home/user/copy",
        "echo 'pip3 install x' > /home/user/README",
        "apt-get remove -y gcc",
        "pip3 uninstall -y requests",
        "cargo build --release",
    ],
)
def test_lines_that_stay_in_the_data_half(line):
    split = tmax.split_post(line + "\n")
    assert split.problems == {} and split.apt == [] and split.pip == []
    assert line in split.data


def test_pip_options_the_base_reproduces():
    split = tmax.split_post(
        "pip3 install --no-cache-dir --default-timeout=100 --retries 5 -U pytest flask\n"
        "pip3 install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "python3 -m pip install 'pandas>=2' || pip install pandas\n"
    )
    assert split.problems == {}
    assert split.pip == ["pytest", "flask", "torch", "pandas>=2"]
    assert split.torch_cpu_index


def test_environment_literals_and_path_prepend_only():
    env = tmax.parse_environment('export LC_ALL=C.UTF-8\nexport PATH="/opt/x/bin:$PATH"\nFOO=bar\n')
    assert env == {"LC_ALL": "C.UTF-8", "PATH": "/opt/x/bin:" + tmax.BASE_PATH, "FOO": "bar"}
    assert tmax.parse_environment("/usr/sbin/sshd || true\n") is None
    assert tmax.parse_environment("export HOME_DIR=$HOME\n") is None
    assert tmax.parse_environment("") == {}


def test_files_resolve_inside_the_task_or_not_at_all():
    files = [("/gpfs/a/t1/fixtures/x.png", "/app/x.png")]
    assert tmax.resolve_files("t1", files, ["fixtures/x.png"]) == [("fixtures/x.png", "/app/x.png")]
    assert tmax.resolve_files("t1", files, []) is None
    assert tmax.resolve_files("t1", [("/elsewhere/x", "/x")], ["x"]) is None


def test_successful_runs_and_the_replay_script(tmp_path):
    tid = make_task(tmp_path)
    source = tmax.Source(tmp_path)
    runs = tmax.successful_runs(source, tid)
    assert [(r.index, r.commands) for r in runs] == [(1, ("cd /home/user", "cat .truth.json > out.txt"))]
    script = tmax.solve_script(runs[0])
    assert script.startswith("#!/bin/bash") and "cd /home/user\n" in script
    assert tmax.SUBMIT_MARKER not in script


def test_convert_writes_a_runnable_task_dir(tmp_path):
    tid = make_task(tmp_path / "src")
    converted = tmax.convert(
        tmax.Source(tmp_path / "src"), tid, protected=["/home/user/.truth.json"], hidden=["/home/user/.truth.json"]
    )
    assert converted.instruction == "Put 42 in /home/user/out.txt."
    assert converted.env["LC_ALL"] == "C.UTF-8"
    task_dir = tmax.materialize(converted, tmp_path / "cache")
    names = sorted(p.relative_to(task_dir).as_posix() for p in task_dir.rglob("*") if p.is_file())
    assert names == [
        ".content",
        "checks/test_initial_state.py",
        "instruction.md",
        "setup/files/fixtures/image.png",
        "setup/inputs.json",
        "setup/post.sh",
        "setup/setup.sh",
        "setup/tmax_box.py",
        "solution/solve.sh",
        "tests/pytest.ini",
        "tests/test.sh",
        "tests/test_final_state.py",
        "tests/tmax_box.py",
    ]
    assert json.loads((task_dir / "setup/inputs.json").read_text())["hidden"] == ["/home/user/.truth.json"]
    for script in ("setup/setup.sh", "setup/post.sh", "tests/test.sh", "solution/solve.sh"):
        assert subprocess.run(["bash", "-n", str(task_dir / script)], check=False).returncode == 0, script
    setup = (task_dir / "setup/setup.sh").read_text()
    assert "cp -a \"$here\"/files/fixtures/image.png /app/fixtures/image.png" in setup
    assert "GIT_COMMITTER_DATE" in setup
    # Idempotent, and rewritten when the content changes.
    assert tmax.materialize(converted, tmp_path / "cache") == task_dir
    again = tmax.convert(tmax.Source(tmp_path / "src"), tid)
    tmax.materialize(again, tmp_path / "cache")
    assert json.loads((task_dir / "setup/inputs.json").read_text())["hidden"] == []


def test_convert_refuses_what_the_static_stage_excludes(tmp_path):
    tid = make_task(tmp_path, definition=DEF.replace("mkdir -p /home/user", "wget https://example.org/x"))
    with pytest.raises(ValueError, match="setup_network"):
        tmax.convert(tmax.Source(tmp_path), tid)


def test_the_test_script_runs_pytest_from_tests_only():
    assert "cd /tests" in tmax.TEST_SH
    assert "--confcutdir=/tests" in tmax.TEST_SH and "-c /tests/pytest.ini" in tmax.TEST_SH
    assert "--ctrf /logs/verifier/ctrf.json" in tmax.TEST_SH
    # The relay runs before pytest, and its failure scores 0.
    assert tmax.TEST_SH.index("tmax_box.py relay") < tmax.TEST_SH.index("pytest")
    assert tmax.TEST_SH.index("echo 0 >") < tmax.TEST_SH.index("tmax_box.py relay")


def test_zip_and_directory_sources_agree(tmp_path):
    import zipfile

    tid = make_task(tmp_path / "dir")
    archive = tmp_path / "tasks.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for path in sorted((tmp_path / "dir").rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(tmp_path / "dir").as_posix())
    a, b = tmax.Source(tmp_path / "dir"), tmax.Source(archive)
    assert a.task_ids() == b.task_ids() == [tid]
    assert a.files(tid) == b.files(tid)
    assert tmax.convert(a, tid).files == tmax.convert(b, tid).files


def test_the_replay_keeps_shell_state_and_survives_a_broken_command(tmp_path):
    run = tmax.Run(
        "solutions/x_summary.json",
        0,
        (
            f"mkdir -p {tmp_path}/w && cd {tmp_path}/w",
            "export GREETING=hello",
            "echo 'unterminated",
            "cat > out.txt << 'EOF'\n$GREETING stays literal\nRELIQUARY_TMAX_COMMAND_4\nEOF",
            'echo "$GREETING" >> out.txt',
        ),
    )
    script = tmax.solve_script(run).replace(f"cd {tmax.WORKDIR}\n", f"cd {tmp_path}\n", 1)
    (tmp_path / "solve.sh").write_text(script)
    subprocess.run(["bash", str(tmp_path / "solve.sh")], stdin=subprocess.DEVNULL, capture_output=True, check=False)
    assert (tmp_path / "w" / "out.txt").read_text() == "$GREETING stays literal\nRELIQUARY_TMAX_COMMAND_4\nhello\n"
