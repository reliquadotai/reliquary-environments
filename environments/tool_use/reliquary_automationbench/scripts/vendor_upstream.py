"""Vendor AutomationBench's world, API tools and rubric, made deterministic.

    python -I scripts/vendor_upstream.py <AutomationBench checkout at UPSTREAM_COMMIT>

Copies `automationbench/{schema,rubric,utils,tools}` into
`reliquary_automationbench/_upstream/`, then rewrites it, mechanically and
reproducibly:

- `automationbench.` imports point at the vendored package;
- every `uuid4()`, `random.*`, `datetime.now`, `datetime.utcnow`,
  `date.today` and `time.time()` reads `_determinism` instead (a seeded
  stream and a clock pinned to the scenario time);
- the OpenAI key reads become None, so the `chatgpt` service always answers
  upstream's offline stub and nothing reaches a network;
- a short list of exact patches (`PATCHES`), each of which must apply once:
  set-typed fields serialise sorted (set order depends on the process's hash
  seed), the API search reads its schemas through `importlib.resources` and
  caches its index instead of writing `index.txt` into the package.

Modules the API mode never reaches (most of `tools/zapier`) are then deleted,
and `_preload.py` lists the remainder so the entrypoint imports all of them
eagerly: upstream imports some modules inside functions, and a verified loader
only serves imports made while the entrypoint is being imported.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

UPSTREAM_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reliquary_automationbench"
TARGET = PACKAGE / "_upstream"
DOTTED = "reliquary_automationbench._upstream"
COPIED = ("schema", "rubric", "utils", "tools")

DET = "reliquary_automationbench._determinism"

# (relative path, old, new): each must occur exactly once.
PATCHES: list[tuple[str, str, str]] = [
    (
        "schema/mailchimp.py",
        "    tags: Set[str] = Field(default_factory=set)",
        "    tags: Annotated[Set[str], PlainSerializer(sorted, return_type=list)] = Field(default_factory=set)",
    ),
    ("schema/mailchimp.py", '"tags": list(self.tags),', '"tags": sorted(self.tags),'),
    (
        "schema/google_ads.py",
        "    members: Set[str] = Field(default_factory=set)",
        "    members: Annotated[Set[str], PlainSerializer(sorted, return_type=list)] = Field(default_factory=set)",
    ),
    (
        "schema/linkedin_ads.py",
        "    members: Set[str] = Field(default_factory=set)",
        "    members: Annotated[Set[str], PlainSerializer(sorted, return_type=list)] = Field(default_factory=set)",
    ),
    ("tools/api/search.py", 'SCHEMAS_DIR = Path(__file__).parent / "schemas"\n', ""),
    ("tools/api/search.py", 'INDEX_FILE = SCHEMAS_DIR / "index.txt"\n', ""),
    (
        "tools/api/search.py",
        '''    schemas: dict[str, dict] = {}
    for schema_file in SCHEMAS_DIR.glob("*.jsonc"):
        with open(schema_file) as f:
            text = "\\n".join(line for line in f if not line.lstrip().startswith("//"))
        data = json.loads(text)
        schemas[data["api"]] = data
    return schemas''',
        '''    # Vendored: read through importlib.resources (the package may be served
    # from verified bytes, not a directory), in name order, once per process.
    return {name: json.loads(json.dumps(data)) for name, data in _SCHEMAS().items()}


@functools.lru_cache(maxsize=1)
def _SCHEMAS() -> dict[str, dict]:
    import importlib.resources

    folder = importlib.resources.files(__package__).joinpath("schemas")
    schemas: dict[str, dict] = {}
    for entry in sorted(folder.iterdir(), key=lambda item: item.name):
        if not entry.name.endswith(".jsonc"):
            continue
        lines = entry.read_bytes().decode("utf-8").splitlines(keepends=True)
        text = "\\n".join(line for line in lines if not line.lstrip().startswith("//"))
        data = json.loads(text)
        schemas[data["api"]] = data
    return schemas''',
    ),
    (
        "tools/api/search.py",
        '''    lines = []
    for api_name, schema in sorted(schemas.items()):
        for endpoint in schema.get("endpoints", []):
            lines.append(_build_index_line(api_name, endpoint))
    try:
        INDEX_FILE.write_text("\\n".join(lines) + "\\n")
    except OSError:
        # Installed packages and sandboxed checkouts may be read-only. The index
        # is only a cache, so searching can continue from the in-memory lines.
        pass
    return lines''',
        '''    lines = []
    for api_name, schema in sorted(schemas.items()):
        for endpoint in schema.get("endpoints", []):
            lines.append(_build_index_line(api_name, endpoint))
    # Vendored: never written back into the package.
    return lines''',
    ),
    (
        "tools/api/search.py",
        '''    schema_files = list(SCHEMAS_DIR.glob("*.jsonc"))
    needs_regen = not INDEX_FILE.exists() or any(
        f.stat().st_mtime > INDEX_FILE.stat().st_mtime for f in schema_files
    )
    if needs_regen:
        return _regenerate_index(schemas)
    return INDEX_FILE.read_text().splitlines()''',
        '''    # Vendored: always the in-memory index (upstream's file cache keyed on mtimes).
    return _regenerate_index(schemas)''',
    ),
    ("tools/api/search.py", "import json\n", "import functools\nimport json\n"),
    # A set of thread ids, listed and cut at max_results in hash order: which
    # threads a listing shows depended on the process. First appearance instead.
    ("tools/zapier/gmail/thread.py", "    thread_ids = set()\n", "    thread_ids: dict = {}\n"),
    ("tools/zapier/gmail/thread.py", "        thread_ids.add(message.thread_id)\n", "        thread_ids[message.thread_id] = None\n"),
    # Unreachable once the key read is None; removed so no model client is left.
    (
        "tools/zapier/chatgpt/completions.py",
        "    from openai import OpenAI\n\n    client = OpenAI(api_key=api_key)\n",
        '    raise RuntimeError("vendored AutomationBench never calls a model")\n    client: Any = None\n',
    ),
    # Upstream's defaults, pinned: grading must not depend on a host's environment.
    (
        "rubric/registry.py",
        'os.environ.get("AUTOMATIONBENCH_STRICT_ASSERTIONS", "1")',
        '"1"',
    ),
    ("rubric/__init__.py", 'os.environ.get("AUTOMATIONBENCH_DEBUG_ASSERTIONS")', "None"),
    ("rubric/__init__.py", "atexit.register(print_assertion_error_summary)\n", ""),
]

# Applied after the generic rewrites, to the import blocks the patches need.
PATCH_IMPORTS = {
    "schema/mailchimp.py": "from typing import Annotated\nfrom pydantic import PlainSerializer\n",
    "schema/google_ads.py": "from typing import Annotated\nfrom pydantic import PlainSerializer\n",
    "schema/linkedin_ads.py": "from typing import Annotated\nfrom pydantic import PlainSerializer\n",
}

_CLOCK = [
    (re.compile(r"(?<![\w.])datetime\.now\b"), "_det_now"),
    (re.compile(r"(?<![\w.])datetime\.utcnow\b"), "_det_utcnow"),
    (re.compile(r"(?<![\w.])date\.today\b"), "_det_today"),
    (re.compile(r"(?<![\w.])time\.time\(\)"), "_det_time()"),
    # A naive datetime's .timestamp() reads the host's zone; upstream's results
    # then depend on TZ. Naive values are read as UTC instead.
    (re.compile(r"((?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\([^()]*\))?)\.timestamp\(\)"), r"_det_timestamp(\1)"),
]


def _insert_after_header(source: str, block: str) -> str:
    """Insert `block` after the docstring and any `from __future__` import."""
    tree = ast.parse(source)
    line = 0
    for node in tree.body:
        is_doc = isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str)
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if (is_doc and node is tree.body[0]) or is_future:
            line = node.end_lineno or line
            continue
        break
    lines = source.splitlines(keepends=True)
    return "".join(lines[:line]) + block + "".join(lines[line:])


def _rewrite(path: Path, relative: str) -> None:
    source = path.read_text()
    source = source.replace("from automationbench.", f"from {DOTTED}.")
    source = source.replace("import automationbench.", f"import {DOTTED}.")
    if "automationbench." in re.sub(rf"{re.escape(DOTTED)}\.", "", source).replace("api.", ""):
        # The name also occurs in prose and URLs; only imports must be gone.
        for match in re.finditer(r"^\s*(from|import)\s+automationbench\b.*$", source, re.M):
            raise SystemExit(f"{relative}: unrewritten import {match.group(0)!r}")
    source = re.sub(r"^(\s*)from uuid import uuid4$", rf"\1from {DET} import uuid4", source, flags=re.M)
    source = re.sub(r"^(\s*)import uuid$", rf"\1from {DET} import uuid", source, flags=re.M)
    source = re.sub(r"^(\s*)import random$", rf"\1from {DET} import random", source, flags=re.M)
    if re.search(r"^\s*from random import|^\s*from uuid import (?!uuid4$)", source, re.M):
        raise SystemExit(f"{relative}: unsupported random/uuid import")
    used = []
    for pattern, name in _CLOCK:
        source, count = pattern.subn(name, source)
        if count:
            used.append(re.match(r"\w+", name).group(0))
    if re.search(r"\.timestamp\(\)|fromtimestamp|localtime|mktime|\.astimezone\(\s*\)", source):
        raise SystemExit(f"{relative}: a host-zone dependent time call is left")
    source = source.replace('os.environ.get("OPENAI_API_KEY")', "None")
    source = source.replace('"PYTEST_CURRENT_TEST" in os.environ', "False")
    if used:
        source = _insert_after_header(source, f"from {DET} import {', '.join(sorted(set(used)))}\n")
    path.write_text(source)


def _patch() -> None:
    for relative, old, new in PATCHES:
        path = TARGET / relative
        source = path.read_text()
        if source.count(old) != 1:
            raise SystemExit(f"patch for {relative} matches {source.count(old)} times: {old[:60]!r}")
        path.write_text(source.replace(old, new))
    for relative, block in PATCH_IMPORTS.items():
        path = TARGET / relative
        path.write_text(_insert_after_header(path.read_text(), block))


def _module_file(name: str) -> Path | None:
    relative = Path(*name.split(".")[2:])
    for candidate in (TARGET / relative / "__init__.py", (TARGET / relative).with_suffix(".py")):
        if candidate.is_file():
            return candidate
    return None


def _closure() -> list[str]:
    """Modules the world, the three API tools and the rubric can import.

    Static, over every import statement including those inside functions
    (which a runtime probe would miss), plus each module's parent packages.
    """
    roots = [f"{DOTTED}.schema.world", f"{DOTTED}.tools.api", f"{DOTTED}.rubric", f"{DOTTED}.rubric.assertions"]
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen or _module_file(name) is None:
            continue
        seen.add(name)
        parts = name.split(".")
        for i in range(2, len(parts)):
            stack.append(".".join(parts[:i]))
        tree = ast.parse(_module_file(name).read_text())
        is_package = _module_file(name).name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                stack.extend(alias.name for alias in node.names if alias.name.startswith(DOTTED))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = parts if is_package else parts[:-1]
                    base = base[: len(base) - (node.level - 1)]
                    module = ".".join(base + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                if not module.startswith(DOTTED):
                    continue
                stack.append(module)
                stack.extend(f"{module}.{alias.name}" for alias in node.names)
    return sorted(seen)


def main(upstream: Path) -> None:
    head = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"upstream is at {head}, expected {UPSTREAM_COMMIT}")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir()
    for name in COPIED:
        shutil.copytree(upstream / "automationbench" / name, TARGET / name, ignore=shutil.ignore_patterns("__pycache__", "index.txt"))
    shutil.copy(upstream / "LICENSE", TARGET / "LICENSE")
    (TARGET / "__init__.py").write_text(
        f'"""AutomationBench (zapier/AutomationBench @ {UPSTREAM_COMMIT[:12]}, MIT), vendored and made deterministic.\n\n'
        'Generated by scripts/vendor_upstream.py; do not edit by hand.\n"""\n'
    )
    # Upstream's tools/__init__ imports every Zapier tool (ALL_TOOLS) for its
    # other toolsets; the API mode needs none of that.
    (TARGET / "tools" / "__init__.py").write_text('"""Vendored: the API toolset only (see tools.api)."""\n')
    for path in sorted(TARGET.rglob("*.py")):
        _rewrite(path, str(path.relative_to(TARGET)))
    _patch()

    modules = set(_closure())
    for path in sorted(TARGET.rglob("*.py")):
        relative = path.relative_to(TARGET).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        name = ".".join((DOTTED, *parts))
        if name not in modules:
            path.unlink()
    for folder in sorted((p for p in TARGET.rglob("*") if p.is_dir()), reverse=True):
        if not any(folder.iterdir()):
            folder.rmdir()
    modules = sorted(_closure())
    (PACKAGE / "_preload.py").write_text(
        '"""Every vendored module, imported eagerly (generated by scripts/vendor_upstream.py).\n\n'
        "Upstream imports some modules inside functions; a verified loader serves only the\n"
        'imports made while the entrypoint is imported, so all of them happen here.\n"""\n\n'
        "import importlib\n\nMODULES = (\n" + "".join(f'    "{m}",\n' for m in modules) + ")\n\n"
        "for _name in MODULES:\n    importlib.import_module(_name)\n"
    )
    print(f"vendored {len(modules)} modules from {head}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
