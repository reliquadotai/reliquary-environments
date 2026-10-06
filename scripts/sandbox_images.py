"""Task images for signed-episode sandbox hosts: pull, check, approve, build.

Image lists come from the env packages, never from here:

    python -m reliquary_swe.sandbox images --split r2e --num-tasks 300 > r2e.json
    python -m reliquary_terminal.sandbox images --split train > terminal.json

Then, on the sandbox host (the gateway never pulls; spec §7):

    python scripts/sandbox_images.py pull r2e.json terminal.json --docker-host unix:///run/docker.sock
    python scripts/sandbox_images.py check r2e.json --runtime runsc --require git
    python scripts/sandbox_images.py approve r2e.json terminal.json   # -> the gateway setting

SWE-smith, R2E, polyglot and MiMo images are upstream images pinned by digest: once the
harness runs on the miner, nothing installs at run time, so nothing needs baking. The one
image we build is tmax's shared base; it goes to a registry the operator names:

    python scripts/sandbox_images.py build-tmax --registry <REGISTRY> --push

without --push it only prints the command. It runs the terminal package's `scripts/tmax_validate.py build --repository
<REGISTRY>/reliquary-tmax-base` (docs/tmax.md) and prints the pushed digest to record
in `tmax_manifest.json`'s `base_image`. Run Docker commands only on a container host.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DIGEST = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
ROOT = Path(__file__).resolve().parent.parent
TERMINAL = ROOT / "environments" / "code" / "reliquary_terminal"
BASE_TOOLS = ("bash", "sleep")
Run = Callable[..., Any]


def load_images(paths: Sequence[Path]) -> list[str]:
    images: list[str] = []
    for path in paths:
        for image in json.loads(Path(path).read_text())["images"]:
            if not DIGEST.fullmatch(image):
                raise ValueError(f"{path}: {image!r} is not pinned by digest")
            images.append(image)
    return list(dict.fromkeys(images))


def _docker(host: str | None) -> list[str]:
    return ["docker", "-H", host] if host else ["docker"]


def pull(images: Sequence[str], *, docker: list[str], jobs: int, run: Run = subprocess.run) -> list[str]:
    """Pull what is missing; returns the images that could not be pulled."""

    def one(image: str) -> str | None:
        if run([*docker, "image", "inspect", image], capture_output=True, text=True).returncode == 0:
            return None
        done = run([*docker, "pull", image], capture_output=True, text=True)
        return image if done.returncode != 0 else None

    with ThreadPoolExecutor(max(1, jobs)) as pool:
        return [image for image in pool.map(one, images) if image is not None]


def check(images: Sequence[str], *, docker: list[str], runtime: str, require: Sequence[str],
          run: Run = subprocess.run) -> list[dict[str, Any]]:
    """Each image starts under `runtime` with no network and has what the episode
    tools and the env's hooks run: /bin/bash, sleep (the box's PID 1), and `require`."""
    probe = " ; ".join(
        ["m=", 'test -x /bin/bash || m="$m bash"']
        + [f'command -v {tool} >/dev/null 2>&1 || m="$m {tool}"' for tool in (*BASE_TOOLS[1:], *require)]
        + ['[ -z "$m" ] || { echo "missing:$m"; exit 3; }'])
    results = []
    for image in images:
        done = run([*docker, "run", "--rm", "--network", "none", "--runtime", runtime,
                    "--entrypoint", "/bin/sh", image, "-c", probe],
                   capture_output=True, text=True, timeout=600)
        results.append({"image": image, "ok": done.returncode == 0,
                        "detail": (done.stdout + done.stderr).strip()[-300:]})
    return results


def approve_line(images: Sequence[str]) -> str:
    return "RELIQUARY_SANDBOX_EPISODE_IMAGES=" + json.dumps(sorted(set(images)))


def build_tmax_command(registry: str, package: Path = TERMINAL) -> list[str]:
    return ["uv", "run", "--project", str(package), "python",
            str(Path(package) / "scripts" / "tmax_validate.py"), "build",
            "--repository", f"{registry.rstrip('/')}/reliquary-tmax-base"]


def main(argv: list[str] | None = None, run: Run = subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pull", "check", "approve"):
        sub = commands.add_parser(name)
        sub.add_argument("manifests", nargs="+", type=Path)
        if name != "approve":
            sub.add_argument("--docker-host")
    commands.choices["pull"].add_argument("--jobs", type=int, default=3)
    commands.choices["check"].add_argument("--runtime", default="runsc")
    commands.choices["check"].add_argument("--require", action="append", default=[])
    build = commands.add_parser("build-tmax")
    build.add_argument("--registry", required=True)
    build.add_argument("--terminal-package", type=Path, default=TERMINAL)
    build.add_argument("--push", action="store_true",
                       help="actually build and push; without it the command is only printed")
    args = parser.parse_args(argv)
    if args.command == "build-tmax":
        command = build_tmax_command(args.registry, args.terminal_package)
        if not args.push:
            print("dry run (add --push to build and push):", " ".join(command))
            return 0
        return run(command).returncode
    images = load_images(args.manifests)
    if args.command == "approve":
        print(approve_line(images))
        return 0
    docker = _docker(args.docker_host)
    if args.command == "pull":
        failed = pull(images, docker=docker, jobs=args.jobs, run=run)
        print(f"{len(images) - len(failed)}/{len(images)} present", file=sys.stderr)
        for image in failed:
            print(f"FAILED {image}")
        return 1 if failed else 0
    results = check(images, docker=docker, runtime=args.runtime, require=args.require, run=run)
    for result in results:
        print(json.dumps(result))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
