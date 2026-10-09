"""TMax-15K's Apptainer tasks, converted to tasks this package can run.

TMax ships no images and no reference solutions (docs/tmax.md). Each task is
an Apptainer definition whose `%post` both installs packages and generates the
task's data, a pytest file for the final state, and up to 8 recorded Gemini
runs. This module turns one task into:

- the **install half** of its `%post` (`apt-get install`, `pip install`),
  which goes into the one shared base image (`tmax_select.base_files`);
- a **setup bundle**: the data half of `%post`, its `%files`, and the in-box
  helper (`tmax_box.py`). `TerminalTask.setup` runs it in every box at start,
  with no network;
- `tests/`: `test.sh`, the final-state test, `pytest.ini`, and the helper,
  which puts the protected inputs back before pytest runs;
- `solution/solve.sh`: a successful Gemini run's commands, replayed in one
  bash process from `/home/user`. The 2026-10-04 spike replayed 160/169.

Everything here is pure and offline: the same pinned zip always converts to
the same bytes.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import re
import shlex
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

SOURCE_REPO = "allenai/TMax-15K"
SOURCE_REVISION = "e3ded940596cc611c9815684ba51a7e8ee0cffc9"
SOURCE_FILE = "tasks.zip"
# The LFS sha256 of `tasks.zip` at SOURCE_REVISION, checked after download.
SOURCE_SHA256 = "59e2efd8785d68214f4f8f61c383ef806dff9778d965af5d13cf45f02e3dc106"

CACHE = Path.home() / ".cache" / "reliquary-terminal" / f"tmax-{SOURCE_REVISION[:12]}"

WORKDIR = "/home/user"
ARTIFACT_ROOTS = ("/app", "/home/user")
# Where setup unpacks its bundle in the box. Removed after setup, and outside
# both artifact roots, so nothing in it reaches the grading box or stays
# visible to the agent.
SETUP_DIR = "/tmp/reliquary-tmax-setup"
# ubuntu:22.04's default PATH, which `$PATH` in `%environment` expands to.
BASE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Variables that choose what a root command executes or loads. A grading box runs its
# root commands with each unset (PATH: BASE_PATH) unless the task's own value stays
# outside the artifact roots (`root_argv`); the static stage excludes a task whose
# value does not (`environment_in_artifact_roots`).
GUARDED_ENV = ("PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "BASH_ENV", "ENV", "PYTHONHOME",
               "PYTHONSTARTUP", "PYTHONPATH")
_ABSOLUTE = {"sh": "/bin/sh", "bash": "/bin/bash", "rm": "/bin/rm"}
# Fixed so that a repository the setup creates gets the same commit hashes in
# the agent's box and the grading box (the spike saw replays fail on hashes
# that differ per build).
FIXED_GIT_DATE = "2026-01-01T00:00:00+0000"
FINAL_TEST = "test_final_state.py"
INITIAL_TEST = "test_initial_state.py"
# The command that ends a recorded run. It is not a shell command.
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

_SECTION = re.compile(r"^%(setup|files|environment|post|runscript|startscript|test|labels|help)\b")


# --------------------------------------------------------------------------
# Source: the pinned zip, or the same tree unpacked.
# --------------------------------------------------------------------------


class Source:
    """TMax's tasks, read from `tasks.zip` or a directory of the same layout
    (`<task_id>/container.def`, ...)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        self._index: dict[str, list[str]] | None = None

    def _build_index(self) -> dict[str, list[str]]:
        if self._index is None:
            index: dict[str, list[str]] = {}
            if self._zip is not None:
                for name in self._zip.namelist():
                    if name.endswith("/"):
                        continue
                    task, _, rel = name.partition("/")
                    index.setdefault(task, []).append(rel)
            else:
                for task_dir in self.path.iterdir():
                    if task_dir.is_dir():
                        index[task_dir.name] = [
                            p.relative_to(task_dir).as_posix()
                            for p in task_dir.rglob("*")
                            if p.is_file()
                        ]
            self._index = {k: sorted(v) for k, v in sorted(index.items())}
        return self._index

    def task_ids(self) -> list[str]:
        return list(self._build_index())

    def files(self, task_id: str) -> list[str]:
        return self._build_index().get(task_id, [])

    def read(self, task_id: str, rel: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(f"{task_id}/{rel}")
        return (self.path / task_id / rel).read_bytes()

    def text(self, task_id: str, rel: str) -> str:
        return self.read(task_id, rel).decode("utf-8", errors="replace")

    def has(self, task_id: str, rel: str) -> bool:
        return rel in self.files(task_id)


def download_source() -> Path:
    """`tasks.zip` at the pinned revision, via the Hugging Face cache, checked
    against its LFS sha256."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            SOURCE_REPO, SOURCE_FILE, repo_type="dataset", revision=SOURCE_REVISION
        )
    )
    check_source(path)
    return path


def check_source(path: Path) -> None:
    """Raise unless the zip at `path` is the pinned `tasks.zip`."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != SOURCE_SHA256:
        raise ValueError(f"{path}: sha256 {digest.hexdigest()} is not the pinned {SOURCE_SHA256}")


# --------------------------------------------------------------------------
# The definition file.
# --------------------------------------------------------------------------


@dataclass
class Definition:
    header: str
    files: list[tuple[str, str]]
    post: str
    environment: str = ""
    startscript: str = ""
    other_sections: list[str] = field(default_factory=list)


def parse_definition(text: str) -> Definition:
    """Split an Apptainer definition into its sections. Section headers are the
    known `%name` keywords at the start of a line, as Apptainer reads them."""
    sections: dict[str, list[str]] = {}
    header: list[str] = []
    current: str | None = None
    for line in text.splitlines():
        match = _SECTION.match(line)
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
            continue
        (sections[current] if current else header).append(line)
    files = []
    for line in sections.get("files", []):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        files.append((parts[0], parts[1] if len(parts) > 1 else parts[0]))
    known = {"files", "post", "environment", "startscript"}
    return Definition(
        header="\n".join(header),
        files=files,
        post="\n".join(sections.get("post", [])),
        environment="\n".join(sections.get("environment", [])),
        startscript="\n".join(sections.get("startscript", [])),
        other_sections=sorted(k for k, v in sections.items() if k not in known and any(s.strip() for s in v)),
    )


def resolve_files(task_id: str, files: list[tuple[str, str]], available: list[str]) -> list[tuple[str, str]] | None:
    """Each `%files` source as a path inside the task directory, or None when
    one cannot be found. The sources are absolute paths on the authors'
    cluster (`/gpfs/.../<task_id>/fixtures/x`); what follows the task id is
    the task-relative path."""
    resolved = []
    for src, dst in files:
        marker = f"/{task_id}/"
        if marker not in src:
            return None
        rel = src.split(marker, 1)[1].rstrip("/")
        if rel not in available and not any(a.startswith(rel + "/") for a in available):
            return None
        resolved.append((rel, dst))
    return resolved


_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PATH_REF = re.compile(r"\$\{PATH\}|\$PATH\b")


def parse_environment(text: str) -> dict[str, str] | None:
    """`%environment` as a literal environment, or None when it is more than
    `[export] NAME=value` lines. `$PATH` expands to the base image's PATH;
    any other expansion, command or control flow is refused."""
    env: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            return None
        if len(tokens) != 1 or not _ASSIGN.match(tokens[0]):
            return None
        name, _, value = tokens[0].partition("=")
        # shlex has already removed the quotes; single-quoted text never
        # expands, but telling it apart from double-quoted text is not worth
        # it: none of the measured definitions single-quote a `$`.
        value = _PATH_REF.sub(env.get("PATH", BASE_PATH), value)
        if "$" in value or "`" in value:
            return None
        env[name] = value
    return env


def _under_roots(path: str) -> bool:
    path = posixpath.normpath(path)
    return any(path == root or path.startswith(root + "/") for root in ARTIFACT_ROOTS)


def _guarded_value_unsafe(value: str) -> bool:
    """A guarded variable's value names an artifact root, or a relative or empty entry
    (resolved against a directory the grading command did not choose)."""
    entries = [e for part in value.split(":") for e in (part.split() or [""])]
    return any(not e.startswith("/") or _under_roots(e) for e in entries)


def env_in_artifact_roots(env: dict[str, str]) -> list[str]:
    """The names of `env` that point a command at the agent's files: a guarded variable
    (`GUARDED_ENV`) whose value is unsafe, or any other `LD_*` / `PYTHON*` variable with
    an absolute entry under an artifact root."""
    found = []
    for name, value in sorted(env.items()):
        if name in GUARDED_ENV:
            if _guarded_value_unsafe(value):
                found.append(name)
        elif name.startswith(("LD_", "PYTHON")):
            if any(e.startswith("/") and _under_roots(e) for e in re.split(r"[:\s]+", value)):
                found.append(name)
    return found


def root_argv(argv: list[str], env: dict[str, str]) -> list[str]:
    """`argv` as a grading box runs it as root: through `/usr/bin/env`, which unsets
    every guarded variable whose value in `env` (what the box applies) is unsafe or
    absent, sets PATH (the task's own when safe, else BASE_PATH), and runs `sh`, `bash`
    and `rm` by absolute path."""
    unset, assign = [], []
    for name in GUARDED_ENV:
        value = env.get(name)
        safe = value is not None and not _guarded_value_unsafe(value)
        if name == "PATH":
            assign.append(f"PATH={value if safe else BASE_PATH}")
        elif safe:
            assign.append(f"{name}={value}")
        else:
            unset += ["-u", name]
    head = _ABSOLUTE.get(argv[0], argv[0])
    return ["/usr/bin/env", *unset, *assign, head, *argv[1:]]


# --------------------------------------------------------------------------
# %post: the install half and the data half.
# --------------------------------------------------------------------------

_HEREDOC = re.compile(r"(?<!<)<<(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
_SEPARATORS = {"&&", "||", ";", "|", "&", "(", ")", ";;", "|&"}
_KEYWORDS = {"if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac", "{", "}", "!", "in"}
_PREFIXES = {"sudo", "time", "nohup", "command", "exec", "env"}
_PIP_COMMANDS = {"pip", "pip3", "pip3.10"}
_PYTHONS = {"python", "python3", "python3.10"}
# pip options that change nothing the base image would do differently.
_PIP_FLAGS = {
    "-q", "-qq", "-qqq", "--quiet", "-v", "--verbose", "-U", "--upgrade", "--no-cache-dir",
    "--break-system-packages", "--ignore-installed", "-I", "--no-deps", "--user",
    "--force-reinstall", "--no-warn-script-location", "--disable-pip-version-check",
    "--prefer-binary", "--no-input", "--root-user-action=ignore", "--pre",
    "--no-build-isolation", "--upgrade-strategy=only-if-needed",
}
# pip options that take a value and change nothing the base would do
# differently (network patience).
_PIP_VALUE_OPTIONS = {"--default-timeout", "--timeout", "--retries", "--progress-bar"}
# The CPU wheel index for torch, which the base image uses for torch anyway.
_PIP_INDEX_OPTIONS = {"-i", "--index-url", "--extra-index-url"}
_TORCH_INDEX = re.compile(r"^https://download\.pytorch\.org/whl/cpu/?$")
_APT_FLAGS = {
    "-y", "--yes", "--assume-yes", "-q", "-qq", "-yq", "-qy", "-yqq", "--quiet",
    "--no-install-recommends", "--fix-missing", "--allow-unauthenticated",
    "--allow-downgrades", "-f", "--fix-broken", "--no-upgrade", "--force-yes",
}
_APT_NAME = re.compile(r"^[a-z0-9][a-z0-9+.\-]+(=[A-Za-z0-9.+:~\-]+)?$")
_APT_INSTALLS = {"install", "update", "upgrade", "dist-upgrade", "full-upgrade"}
# Segments that may share a line with an install and are harmless to drop:
# what an install line usually ends with.
_APT_CLEANUP = re.compile(r"^(apt-get|apt)\s+(clean|autoclean|autoremove(\s+-y)?)$|^rm\s+-rf\s+/var/lib/apt/lists/\*?$|^(true|:)$")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "[::1]"}
_URL = re.compile(r"^(?:https?|ftp)://([^/:?#]+)")


@dataclass
class PostSplit:
    apt: list[str] = field(default_factory=list)
    pip: list[str] = field(default_factory=list)
    torch_cpu_index: bool = False
    data: str = ""
    # Reason codes this %post earns on its own (`install_unparsed`,
    # `setup_network`, `apptainer_only_path`), with the offending lines.
    problems: dict[str, list[str]] = field(default_factory=dict)

    def flag(self, code: str, line: str) -> None:
        self.problems.setdefault(code, []).append(line.strip()[:300])


@dataclass
class _Logical:
    text: str  # continuation lines joined
    raw: list[str]
    heredoc_body: bool = False
    executed_body: bool = False  # the body of `bash <<EOF`, which runs


def _logical_lines(script: str) -> Iterator[_Logical]:
    lines = script.splitlines()
    i = 0
    while i < len(lines):
        raw = [lines[i]]
        text = lines[i]
        while text.endswith("\\") and i + 1 < len(lines):
            i += 1
            raw.append(lines[i])
            text = text[:-1] + " " + lines[i]
        i += 1
        yield _Logical(text=text, raw=raw)
        markers = [(m.group(1) == "-", m.group(3)) for m in _HEREDOC.finditer(text)]
        executed = bool(markers) and _command_word(_first_segment(text)) in {"bash", "sh"}
        for strip_tabs, marker in markers:
            body: list[str] = []
            while i < len(lines):
                line = lines[i]
                i += 1
                if (line.lstrip("\t") if strip_tabs else line) == marker or line.strip() == marker:
                    yield _Logical(text="\n".join(body), raw=body + [line], heredoc_body=True, executed_body=executed)
                    break
                body.append(line)
            else:
                yield _Logical(text="\n".join(body), raw=body, heredoc_body=True, executed_body=executed)
                yield _Logical(text=f"<<{marker}", raw=[], heredoc_body=True)  # unterminated


def _tokens(text: str) -> list[str] | None:
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        return list(lexer)
    except ValueError:
        return None


def _segments(tokens: list[str]) -> list[tuple[str, list[str]]]:
    """(separator before, tokens) for each simple command of a line."""
    out: list[tuple[str, list[str]]] = []
    sep = ""
    current: list[str] = []
    for token in tokens:
        if token in _SEPARATORS:
            if current:
                out.append((sep, current))
            current = []
            sep = token
        else:
            current.append(token)
    if current:
        out.append((sep, current))
    return out


def _first_segment(text: str) -> list[str]:
    tokens = _tokens(text) or []
    segments = _segments(tokens)
    return segments[0][1] if segments else []


def _strip_prefixes(words: list[str]) -> tuple[list[str], bool]:
    """The command and its arguments, without leading assignments and
    wrappers (`sudo`, `env`, ...); and whether a shell keyword came first,
    which puts the command inside a compound statement."""
    compound = False
    i = 0
    while i < len(words):
        word = words[i]
        if word in _KEYWORDS:
            compound = True
        elif not (_ASSIGN.match(word) or word in _PREFIXES or (word.startswith("-") and i and words[i - 1] == "env")):
            break
        i += 1
    return words[i:], compound


def _command_word(words: list[str]) -> str:
    rest, _ = _strip_prefixes(words)
    return PurePosixPath(rest[0]).name if rest else ""


def _apt_subcommand(args: list[str]) -> str | None:
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg == "-o":
            skip = True
        elif not arg.startswith("-"):
            return arg
    return None


_SYSTEM_BIN = ("/usr/bin/", "/usr/local/bin/", "/bin/")


def _install_kind(words: list[str]) -> str | None:
    """"apt" or "pip" for an install the base image can take over; "venv"
    for a pip install into some other interpreter (a virtualenv's pip),
    which it cannot; None for anything else."""
    rest, _ = _strip_prefixes(words)
    if not rest:
        return None
    program = rest[0]
    cmd = PurePosixPath(program).name
    elsewhere = "/" in program and not program.startswith(_SYSTEM_BIN)
    args = rest[1:]
    if cmd in {"apt-get", "apt"} and _apt_subcommand(args) in _APT_INSTALLS:
        return "apt"
    if (cmd in _PIP_COMMANDS and "install" in args[:2]) or (
        cmd in _PYTHONS and args[:3] == ["-m", "pip", "install"]
    ):
        return "venv" if elsewhere else "pip"
    return None


def _apt_packages(words: list[str]) -> list[str] | None:
    rest, _ = _strip_prefixes(words)
    args = rest[1:]
    sub = _apt_subcommand(args)
    if sub != "install":
        return []  # update / upgrade: the base image is built current
    packages = []
    skip = False
    seen_sub = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == "-o":
            skip = True
            continue
        if arg.startswith(("-o", "--option")):
            continue
        if arg.startswith("-"):
            if arg not in _APT_FLAGS:
                return None
            continue
        if not seen_sub:
            seen_sub = True
            continue
        if not _APT_NAME.match(arg):
            return None
        packages.append(arg)
    return packages


def _pip_requirements(words: list[str]) -> tuple[list[str], bool] | None:
    """The requirement strings of a pip install, and whether it named the
    torch CPU index; None for anything this converter does not model."""
    from packaging.requirements import InvalidRequirement, Requirement

    rest, _ = _strip_prefixes(words)
    args = rest[1:]
    if PurePosixPath(rest[0]).name in _PYTHONS:
        args = args[2:]  # -m pip
    args = args[args.index("install") + 1:]
    requirements: list[str] = []
    torch_index = False
    skip_value = False
    for i, arg in enumerate(args):
        if skip_value:
            skip_value = False
            continue
        if arg in _PIP_INDEX_OPTIONS:
            value = args[i + 1] if i + 1 < len(args) else ""
            if not _TORCH_INDEX.match(value):
                return None
            torch_index = True
            skip_value = True
            continue
        if any(arg.startswith(opt + "=") for opt in _PIP_INDEX_OPTIONS):
            if not _TORCH_INDEX.match(arg.split("=", 1)[1]):
                return None
            torch_index = True
            continue
        if arg in _PIP_VALUE_OPTIONS:
            skip_value = True
            continue
        if arg.startswith("-"):
            if arg not in _PIP_FLAGS and arg.split("=", 1)[0] not in _PIP_VALUE_OPTIONS:
                return None
            continue
        if "$" in arg or "/" in arg or arg.startswith(".") or "://" in arg or arg.endswith((".whl", ".tar.gz", ".zip")):
            return None
        try:
            Requirement(arg)
        except InvalidRequirement:
            return None
        requirements.append(arg)
    return requirements, torch_index


def _network_reason(words: list[str]) -> str | None:
    rest, _ = _strip_prefixes(words)
    if not rest:
        return None
    cmd = PurePosixPath(rest[0]).name
    args = rest[1:]
    sub = next((a for a in args if not a.startswith("-")), "")

    def remote_url() -> bool:
        for arg in args:
            m = _URL.match(arg)
            if m and m.group(1) not in _LOCAL_HOSTS:
                return True
        return False

    if cmd in {"curl", "wget"} and remote_url():
        return cmd
    if cmd == "git" and sub in {"clone", "fetch", "pull", "ls-remote", "submodule"} and (
        remote_url() or any(a.startswith("git@") for a in args)
    ):
        return f"git {sub}"
    if cmd in {"rustup", "rustup-init", "npx", "snap", "conda", "mamba", "add-apt-repository"}:
        return cmd
    if cmd == "go" and (sub in {"get", "install"} or args[:2] == ["mod", "download"]):
        return f"go {sub}"
    if cmd in {"npm", "yarn", "pnpm"} and sub in {"install", "i", "ci", "add", "update", "upgrade", ""}:
        return f"{cmd} {sub}".strip()
    if cmd == "cargo" and sub in {"install", "fetch", "add", "update", "search"}:
        return f"cargo {sub}"
    if cmd in _PIP_COMMANDS and sub == "download":
        return "pip download"
    if cmd in _PYTHONS and args[:3] == ["-m", "pip", "download"]:
        return "pip download"
    if cmd == "gem" and sub == "install":
        return "gem install"
    if cmd == "apt-key" and any(a in {"adv", "--keyserver"} for a in args):
        return "apt-key"
    if cmd == "docker" and sub in {"pull", "run", "build"}:
        return f"docker {sub}"
    return None


_SINGULARITY = re.compile(r"/\.singularity\.d\b")


def split_post(post: str) -> PostSplit:
    """Separate `%post` into what the base image installs and what each box
    runs at setup. A line whose every command is an install (or its usual
    cleanup: `apt-get clean`, `rm -rf /var/lib/apt/lists/*`, `|| true`) moves
    to the base. An install anywhere else -- in a compound statement, in a
    pipeline, chained with real work, with options or arguments the base
    cannot reproduce -- marks the task `install_unparsed` rather than being
    guessed at."""
    split = PostSplit()
    kept: list[str] = []
    for logical in _logical_lines(post):
        if logical.heredoc_body:
            kept.extend(logical.raw)
            if not logical.raw and logical.text.startswith("<<"):
                split.flag("install_unparsed", f"unterminated heredoc {logical.text}")
            if logical.executed_body:
                for line in logical.text.splitlines():
                    tokens = _tokens(line) or []
                    for _, words in _segments(tokens):
                        if _install_kind(words):
                            split.flag("install_unparsed", line)
                        reason = _network_reason(words)
                        if reason:
                            split.flag("setup_network", line)
            continue
        text = logical.text
        if _SINGULARITY.search(text):
            split.flag("apptainer_only_path", text)
        tokens = _tokens(text)
        if tokens is None:
            if re.search(r"\b(apt-get|apt|pip3?)\b", text):
                split.flag("install_unparsed", text)
            if re.search(r"\b(curl|wget|git clone)\b.*://", text):
                split.flag("setup_network", text)
            kept.extend(logical.raw)
            continue
        segments = _segments(tokens)
        kinds = [_install_kind(words) for _, words in segments]
        if any(kinds):
            movable = True
            apt: list[str] = []
            pip: list[str] = []
            torch_index = False
            after_install_or = False
            for (sep, words), kind in zip(segments, kinds):
                rest, compound = _strip_prefixes(words)
                if compound or sep in {"|", "(", ")", "&", "|&"}:
                    movable = False
                    break
                if after_install_or and sep == "||":
                    continue  # the fallback of an install the base already did
                after_install_or = False
                if kind == "apt":
                    packages = _apt_packages(words)
                    if packages is None:
                        movable = False
                        break
                    apt += packages
                    after_install_or = True
                elif kind == "pip":
                    parsed = _pip_requirements(words)
                    if parsed is None:
                        movable = False
                        break
                    pip += parsed[0]
                    torch_index |= parsed[1]
                    after_install_or = True
                elif kind == "venv":
                    movable = False
                    break
                elif _APT_CLEANUP.match(" ".join(rest)):
                    continue
                else:
                    movable = False
                    break
            if movable:
                split.apt += apt
                split.pip += pip
                split.torch_cpu_index |= torch_index
                # `true`, not a bare comment: the line may be the only command of
                # an `if` or loop body, which the shell refuses empty.
                kept.append("true  # reliquary-tmax: in the base image: " + " ".join(r.strip() for r in logical.raw))
                continue
            split.flag("install_unparsed", text)
            kept.extend(logical.raw)
            continue
        for _, words in segments:
            if _network_reason(words):
                split.flag("setup_network", text)
                break
        kept.extend(logical.raw)
    split.data = "\n".join(kept).strip("\n") + "\n"
    return split


# --------------------------------------------------------------------------
# Reference solutions.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    file: str  # e.g. solutions/gemini_gemini-3-flash-preview_summary.json
    index: int
    commands: tuple[str, ...]


def successful_runs(source: Source, task_id: str) -> list[Run]:
    """Every successful recorded run, in file then run order, as the bash
    commands it executed (the submit marker removed)."""
    runs = []
    for rel in source.files(task_id):
        if not (rel.startswith("solutions/") and rel.endswith("_summary.json")):
            continue
        try:
            summary = json.loads(source.read(task_id, rel))
        except ValueError:
            continue
        for index, result in enumerate(summary.get("results") or []):
            if not result.get("success"):
                continue
            commands = []
            for message in result.get("messages") or []:
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    if function.get("name") != "bash":
                        continue
                    try:
                        command = json.loads(function.get("arguments") or "{}").get("command", "")
                    except ValueError:
                        continue
                    if command and SUBMIT_MARKER not in command:
                        commands.append(command)
            if commands:
                runs.append(Run(rel, index, tuple(commands)))
    return runs


def solve_script(run: Run) -> str:
    """The run's commands in one bash process from /home/user: the recorded
    agent had a persistent shell, so `cd` and `export` carry over. Each
    command goes through its own `eval`, as each was its own submission: one
    that does not parse (6 of the 6,061 runs the static stage keeps) fails
    alone instead of taking every later command with it."""
    lines = [
        "#!/bin/bash",
        f"# reliquary-tmax reference: {run.file} run {run.index}, replayed in one shell.",
        f"cd {WORKDIR}",
    ]
    for i, command in enumerate(run.commands):
        marker = f"RELIQUARY_TMAX_COMMAND_{i}"
        while any(line.strip() == marker for line in command.splitlines()):
            marker += "_"
        lines += [f"eval \"$(cat <<'{marker}'", command, marker, ')"']
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# The converted task directory.
# --------------------------------------------------------------------------


def setup_script(task_id: str, files: list[tuple[str, str]]) -> str:
    """Run in every box at start, as root, from `/`: `%files`, then the data
    half of `%post` with `/bin/sh -e` as Apptainer runs it, then the helper
    (stash the protected inputs in the grading box, delete the hidden ones in
    the agent's)."""
    lines = [
        "#!/bin/bash",
        f"# reliquary-tmax setup for {task_id}. Usage: setup.sh agent|grade",
        "set -euo pipefail",
        'role="$1"',
        'here="$(cd "$(dirname "$0")" && pwd)"',
        "export DEBIAN_FRONTEND=noninteractive",
        f"export GIT_AUTHOR_DATE='{FIXED_GIT_DATE}' GIT_COMMITTER_DATE='{FIXED_GIT_DATE}'",
        "cd /",
    ]
    for rel, dst in files:
        q_src = shlex.quote(f"files/{rel}")
        q_dst = shlex.quote(dst)
        lines.append(f'mkdir -p "$(dirname {q_dst})" && cp -a "$here"/{q_src} {q_dst}')
    lines += [
        '(cd / && /bin/sh -e "$here/post.sh")',
        'python3 "$here/tmax_box.py" setup "$role" "$here/inputs.json"',
    ]
    return "\n".join(lines) + "\n"


TEST_SH = f"""#!/bin/bash
# reliquary-tmax grading. Puts the protected inputs back over the agent's
# artifacts, then runs the final-state test from /tests, never from a
# directory the agent wrote.
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
if ! python3 -I /tests/tmax_box.py relay; then
    echo "reliquary-tmax: protected inputs could not be restored; scoring 0"
    exit 1
fi
cd /tests
python3 -s -m pytest -q -p no:cacheprovider -c /tests/pytest.ini --rootdir=/tests \\
    --confcutdir=/tests --ctrf /logs/verifier/ctrf.json -rA /tests/{FINAL_TEST}
rc=$?
if [ "$rc" -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; fi
exit "$rc"
"""

PYTEST_INI = "[pytest]\naddopts =\n"


def inputs_json(protected: list[str], hidden: list[str]) -> str:
    return json.dumps({"protected": sorted(protected), "hidden": sorted(hidden)}, indent=1) + "\n"


@dataclass
class Converted:
    """One task, converted: everything `materialize` writes."""

    task_id: str
    instruction: str
    env: dict[str, str]
    files: dict[str, bytes]  # task-dir-relative path -> content
    domain: str | None = None
    skill_type: str | None = None


def convert(
    source: Source,
    task_id: str,
    protected: list[str] = (),
    hidden: list[str] = (),
    run: int = 0,
) -> Converted:
    """Convert one task. `protected` and `hidden` come from the manifest
    (measured in the box phase); `run` picks the successful run to use as
    the reference. Raises ValueError for a task the static stage excludes."""
    definition = parse_definition(source.text(task_id, "container.def"))
    split = split_post(definition.post)
    if split.problems:
        raise ValueError(f"{task_id}: {sorted(split.problems)}")
    files = resolve_files(task_id, definition.files, source.files(task_id))
    if files is None:
        raise ValueError(f"{task_id}: files_unresolved")
    env = parse_environment(definition.environment)
    if env is None:
        raise ValueError(f"{task_id}: environment_unparsed")
    task = json.loads(source.read(task_id, "task.json"))
    helper = (Path(__file__).parent / "tmax_box.py").read_bytes()
    out: dict[str, bytes] = {
        "instruction.md": task["description"].strip().encode() + b"\n",
        "setup/setup.sh": setup_script(task_id, files).encode(),
        "setup/post.sh": split.data.encode(),
        "setup/inputs.json": inputs_json(list(protected), list(hidden)).encode(),
        "setup/tmax_box.py": helper,
        "tests/test.sh": TEST_SH.encode(),
        "tests/pytest.ini": PYTEST_INI.encode(),
        "tests/tmax_box.py": helper,
        f"tests/{FINAL_TEST}": source.read(task_id, FINAL_TEST),
    }
    for rel, _ in files:
        for name in source.files(task_id):
            if name == rel or name.startswith(rel + "/"):
                out[f"setup/files/{name}"] = source.read(task_id, name)
    runs = successful_runs(source, task_id)
    if runs:
        out["solution/solve.sh"] = solve_script(runs[min(run, len(runs) - 1)]).encode()
    if source.has(task_id, INITIAL_TEST):
        out[f"checks/{INITIAL_TEST}"] = source.read(task_id, INITIAL_TEST)
    return Converted(
        task_id=task_id,
        instruction=task["description"].strip(),
        env=env,
        files=out,
        domain=task.get("domain"),
        skill_type=task.get("skill_type"),
    )


def materialize(converted: Converted, root: Path = CACHE) -> Path:
    """Write a converted task under `root/<task_id>`. Rewritten only when its content
    differs. Safe across processes: each writer stages in a directory of its own and
    renames it into place; a writer that finds the place taken by the same content
    keeps that one."""
    task_dir = root / converted.task_id
    digest = _content_digest(converted)
    if stamp_of(task_dir) == digest:
        return task_dir
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{converted.task_id}.partial-", dir=root))
    try:
        for rel, content in converted.files.items():
            path = staging / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            if rel.endswith(".sh"):
                path.chmod(0o755)
        (staging / ".content").write_text(digest)
        staging.chmod(0o755)
        for _ in range(8):
            try:
                os.rename(staging, task_dir)
                return task_dir
            except OSError:  # taken (a non-empty directory)
                if stamp_of(task_dir) == digest:
                    return task_dir
                stale = root / f".{converted.task_id}.stale-{uuid.uuid4().hex}"
                try:
                    os.rename(task_dir, stale)
                except FileNotFoundError:
                    continue
                shutil.rmtree(stale, ignore_errors=True)
        raise RuntimeError(f"could not materialize {task_dir}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def tree_digest(task_dir: Path) -> str:
    """A sha256 over a directory's files (relative path and bytes, sorted by path), the
    `.content` stamp left out: for a materialized task, the stamp's own value."""
    root = Path(task_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"no task directory at {root}")
    files = sorted((p.relative_to(root).as_posix(), p) for p in root.rglob("*")
                   if p.is_file() and p.relative_to(root).as_posix() != ".content")
    digest = hashlib.sha256()
    for rel, path in files:
        digest.update(rel.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


_VERIFIED: dict[str, str] = {}
"""Task directories whose files were checked against their stamp in this process."""


def stamp_of(task_dir: Path) -> str | None:
    """A materialized task's `.content` stamp, or None when there is none or the files
    no longer match it. Checked against the files once per process per stamp."""
    stamp = Path(task_dir) / ".content"
    try:
        value = stamp.read_text().strip()
    except OSError:
        return None
    key = str(Path(task_dir).resolve())
    if _VERIFIED.get(key) != value:
        try:
            if tree_digest(Path(task_dir)) != value:
                return None
        except OSError:
            return None
        _VERIFIED[key] = value
    return value


def _content_digest(converted: Converted) -> str:
    digest = hashlib.sha256()
    for rel in sorted(converted.files):
        digest.update(rel.encode() + b"\0" + converted.files[rel] + b"\0")
    return digest.hexdigest()


async def run_setup(runtime, bundle: Path, role: str) -> None:
    """Upload a task's setup bundle to SETUP_DIR, run it as `role`
    ("agent" or "grade"), and remove it whatever happens. Raises on failure:
    a box whose data was not generated must never be graded or handed to
    an agent."""
    if role not in ("agent", "grade"):
        raise ValueError(f"unknown setup role {role!r}")
    archive = f"{SETUP_DIR}.tgz"
    await runtime.write(archive, tar_bytes(bundle))
    script = (
        f"rm -rf {SETUP_DIR} && mkdir -p {SETUP_DIR} && "
        f"tar --no-same-owner -xzf {archive} -C {SETUP_DIR} && rm -f {archive} && "
        f"bash {SETUP_DIR}/setup.sh {role}; rc=$?; rm -rf {SETUP_DIR} {archive}; exit $rc"
    )
    result = await runtime.run(["sh", "-c", script], {})
    if result.exit_code:
        raise RuntimeError(
            f"tmax setup ({role}) failed (exit {result.exit_code}): "
            f"{(result.stderr or result.stdout).strip()[-800:]}"
        )


def tar_bytes(directory: Path) -> bytes:
    """A gzipped tar of `directory`'s contents, sorted, for upload."""
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in sorted(directory.iterdir()):
            tar.add(item, arcname=item.name)
    return buffer.getvalue()
