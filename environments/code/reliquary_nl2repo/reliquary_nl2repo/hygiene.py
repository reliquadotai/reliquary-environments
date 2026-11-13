"""What a task image keeps of the reference implementation, and the scripts removing it.

The images were built from the upstream project: its source was copied in, installed,
tested, and (mostly) deleted. What can survive, found by scanning every pinned image
(`scripts/scan_images.py`):

* a distribution installed from the project's directory (PEP 610 `direct_url.json` with
  `dir_info`): 21 images, 7 of them non-editable, so the reference itself sits in
  site-packages; legacy `setup.py develop` links (`*.egg-link`): 24 images;
* copies left outside `/workspace` (`REFERENCE_PATHS`), and build leftovers in `/tmp`
  (binaryalert's lambda packages) and pip's caches (built wheels).

`REMOVE_SCRIPT` deletes the per-task paths and pip's caches (both boxes);
`UNINSTALL_SCRIPT` uninstalls every `dir_info` distribution and removes every egg-link
and its `easy-install.pth` line, with the box's own Python (3.7 to 3.13), then checks
none is left. The agent's box also loses `/tmp`'s content and `/workspace`'s
(`task.CLEAR_WORKSPACE`). An image whose reference survives all this shows in the gold
check's empty-workspace run and leak probe, and is not served.
"""

from __future__ import annotations

REFERENCE_PATHS: dict[str, tuple[str, ...]] = {
    "asteval": ("/project",),
    "jinja": ("/jinja",),
    "mootdx": ("/tests", "/pyproject.toml", "/requirements.dev.txt"),
    "paillier": ("/app",),
}
"""Task id -> absolute paths outside `/workspace` holding reference copies (scan of the
pinned images; the agent's box and the grading box both lose them)."""

REMOVE_SCRIPT = r'''
set -u
for p in "$@"; do
  case "$p" in /*) ;; *) echo "not absolute: $p" >&2; exit 2 ;; esac
  rm -rf -- "$p"
  [ ! -e "$p" ] && [ ! -L "$p" ] || { echo "could not remove $p" >&2; exit 3; }
done
for c in /root/.cache/pip /home/*/.cache/pip; do
  [ -e "$c" ] || continue
  rm -rf -- "$c" 2>/dev/null
  [ ! -e "$c" ] || { echo "could not remove $c" >&2; exit 3; }
done
exit 0
'''
"""$@: the task's `REFERENCE_PATHS`."""

UNINSTALL_PY = r'''
import glob, json, os, shutil, site, subprocess, sys
def sites():
    found = set(p for p in sys.path if p)
    try:
        found.update(site.getsitepackages())
    except Exception:
        pass
    return sorted(p for p in found if os.path.isdir(p)
                  and ("site-packages" in p or "dist-packages" in p))
def local_dists():
    out = []
    for d in sites():
        for info in glob.glob(os.path.join(d, "*.dist-info")):
            try:
                with open(os.path.join(info, "direct_url.json")) as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and "dir_info" in doc:
                out.append(info)
    return out
def dist_name(info):
    try:
        with open(os.path.join(info, "METADATA"), errors="replace") as f:
            for line in f:
                if line.startswith("Name:"):
                    return line.split(":", 1)[1].strip()
                if not line.strip():
                    break
    except OSError:
        pass
    return None
def by_record(info):
    base = os.path.dirname(info)
    try:
        with open(os.path.join(info, "RECORD"), errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        lines = []
    for line in lines:
        rel = line.rsplit(",", 2)[0] if line.count(",") >= 2 else line
        path = os.path.normpath(os.path.join(base, rel))
        if path.startswith((base + os.sep, sys.prefix + os.sep)) and os.path.isfile(path):
            os.remove(path)
    shutil.rmtree(info, ignore_errors=True)
removed = []
for info in local_dists():
    name = dist_name(info)
    if name:
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", name],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.isdir(info):
        by_record(info)
    removed.append(name or os.path.basename(info))
for d in sites():
    links = glob.glob(os.path.join(d, "*.egg-link"))
    targets = set()
    for link in links:
        try:
            with open(link, errors="replace") as f:
                targets.add(os.path.normpath(f.readline().strip()))
        except OSError:
            pass
        os.remove(link)
        removed.append(os.path.basename(link))
    pth = os.path.join(d, "easy-install.pth")
    if targets and os.path.isfile(pth):
        with open(pth, errors="replace") as f:
            kept = [l for l in f.read().splitlines() if os.path.normpath(l.strip()) not in targets]
        with open(pth, "w") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))
left = local_dists() + [l for d in sites() for l in glob.glob(os.path.join(d, "*.egg-link"))]
print(json.dumps({"removed": removed, "left": left}))
sys.exit(1 if left else 0)
'''

UNINSTALL_SCRIPT = (
    'py=$(command -v python3 || command -v python) || { echo "no python" >&2; exit 2; }\n'
    'exec "$py" -I -c "$0"\n'
)
"""Run as `sh -c UNINSTALL_SCRIPT <UNINSTALL_PY>`: the Python source travels as $0."""

__all__ = ["REFERENCE_PATHS", "REMOVE_SCRIPT", "UNINSTALL_PY", "UNINSTALL_SCRIPT"]
