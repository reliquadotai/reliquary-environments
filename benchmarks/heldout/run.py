"""Run the held-out suite against one OpenAI-compatible endpoint.

    uv run python run.py --model teutonic --base-url http://127.0.0.1:8000/v1

Each benchmark is an unmodified upstream taskset driven by `configs/<name>.toml`.
What this adds is what every run must share and none should be able to forget:
the endpoint, the sampling, a local runtime, and no upload. Upstream defaults
are a hosted model, a hosted sandbox VM per episode, and `--push`, which sends
the finished run to the Prime platform.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
BENCHMARKS = sorted(path.stem for path in CONFIGS.glob("*.toml"))


# Docker with every destination denied: the episode reaches the model through
# the interception endpoint and nothing else. The tasks declare a network policy,
# which the subprocess runtime refuses outright, and the CLI cannot spell an empty
# list, so the runtime is written into the config the run actually reads.
RUNTIME = '\n[env.agent.runtime]\ntype = "docker"\nallow = []\n'


def write_config(args: argparse.Namespace, benchmark: str) -> Path:
    config = args.output_dir / "configs" / f"{benchmark}.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text((CONFIGS / f"{benchmark}.toml").read_text() + RUNTIME)
    return config


def eval_command(args: argparse.Namespace, benchmark: str, config: Path) -> list[str]:
    command = [
        str(Path(sys.executable).with_name("eval")),
        "@",
        str(config),
        "--model",
        args.model,
        "--client.base-url",
        args.base_url,
        "--client.api-key-var",
        args.api_key_var,
        "--sampling.temperature",
        str(args.temperature),
        "--sampling.top-p",
        str(args.top_p),
        "--max-concurrent",
        str(args.max_concurrent),
        "--output-dir",
        str(args.output_dir),
        "--run.dir",
        benchmark,
        "--no-push",
        "--no-rich",
    ]
    if args.num_tasks is not None:
        command += ["--num-tasks", str(args.num_tasks)]
    return command


def summarize(run_dir: Path) -> dict:
    traces = run_dir / "traces.jsonl"
    if not traces.exists():
        return {"rollouts": 0, "reward": None, "errors": None}
    rewards, errors = [], 0
    with traces.open() as handle:
        for line in handle:
            episode = json.loads(line)
            if not episode["ok"]:
                errors += 1
                continue
            # An episode's reward is its agents' weighted rewards summed, as
            # `Trace.reward` computes it; the benchmarks here have one agent.
            rewards.append(
                sum(
                    reward["score"] * reward["weight"]
                    for trace in episode["traces"]
                    for reward in trace["rewards"].values()
                    if reward is not None
                )
            )
    return {
        "rollouts": len(rewards) + errors,
        "reward": statistics.fmean(rewards) if rewards else None,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model name the endpoint serves.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. a local vLLM.")
    parser.add_argument(
        "--api-key-var",
        default="HELDOUT_API_KEY",
        help="Environment variable holding the endpoint's key; set to EMPTY when unset.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-concurrent", type=int, default=64)
    parser.add_argument("--num-tasks", type=int, default=None, help="Cap tasks per benchmark (smoke runs).")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--only", nargs="+", choices=BENCHMARKS, default=BENCHMARKS)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()

    os.environ.setdefault(args.api_key_var, "EMPTY")
    results = {}
    for benchmark in args.only:
        print(f"== {benchmark}", flush=True)
        completed = subprocess.run(eval_command(args, benchmark, write_config(args, benchmark)), cwd=HERE)
        results[benchmark] = summarize(args.output_dir / benchmark)
        results[benchmark]["exit_code"] = completed.returncode

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_tasks": args.num_tasks,
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{'benchmark':<15} {'reward':>8} {'rollouts':>9} {'errors':>7}")
    for benchmark, result in results.items():
        reward = "-" if result["reward"] is None else f"{result['reward']:.4f}"
        print(f"{benchmark:<15} {reward:>8} {result['rollouts']:>9} {str(result['errors']):>7}")
    return 0 if all(result["exit_code"] == 0 for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
