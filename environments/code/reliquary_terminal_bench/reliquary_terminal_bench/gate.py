"""Which tasks each served split holds: those whose reference grades right end to end.

Upstream grades in the agent's own box with network. A signed-episode sandbox stops the
agent's processes and grades a pristine box that receives the agent's files (`task`),
always offline. The agent's box is offline too, unless the gateway gives the env
controlled egress for the agent phase (reliquary-sandbox `docs/egress.md`: an allowlist
proxy, dated apt snapshots, pip/uv date pins). A task is served only after
`scripts/gold_check.py`, run on a gVisor host with the verifier images, has graded its
reference solution (`solution/solve.sh`, in the agent's box) 1 and a no-op episode 0,
through this package's own hooks, inside the grading budget:

* `VERIFIED`: the agent's box online through the egress profile (`EGRESS`), grading
  offline: the `eval` split, the closest to the leaderboard (upstream lets every task
  reach the internet).
* `VERIFIED_OFFLINE`: the agent's box offline too: the `eval_offline` split, for a
  gateway without egress.

`gold_check.py --write-gate` rewrites these (None until the gold check has run: the split
then refuses to load; `eval_all` serves all 89). `EXCLUDED` names tasks left out for a
reason known before the gold check, `NOT_VERIFIED` why the others failed it. A `.py` file
on purpose: the package's identity digests its `.py` files, so the served sets cannot
change under the same identity.
"""

from __future__ import annotations

EGRESS_PROFILE = "packages"
"""The gateway egress profile the `eval` split is gated with (agent phase only)."""
EGRESS_EXTRA_HOSTS = (
    # Beyond reliquary-sandbox's default `packages` allowlist: what the references fetch.
    "huggingface.co", "*.huggingface.co", "*.hf.co", "*.hf.space",
    "cloud.r-project.org", "cran.r-project.org", "cran.rstudio.com",
    "download.pytorch.org", "download-r2.pytorch.org",
    "*.githubusercontent.com", "www.povray.org", "19100.neocities.org", "benryves.com",
    "httpstan.readthedocs.io",
)
"""Hosts a gateway serving `eval` adds to its egress profile (the gold check's)."""

_LIVE_PROCESS = ("graded against a process the agent leaves running (server, daemon or VM); "
                 "the sandbox stops the agent's processes and grades a fresh box")

VERIFIED_OFFLINE: frozenset[str] | None = frozenset({
    'break-filter-js-from-html',
    'cancel-async-tasks',
    'circuit-fibsqrt',
    'cobol-modernization',
    'code-from-image',
    'constraints-scheduling',
    'custom-memory-heap-crash',
    'db-wal-recovery',
    'distribution-search',
    'dna-assembly',
    'dna-insert',
    'extract-elf',
    'extract-moves-from-video',
    'feal-differential-cryptanalysis',
    'feal-linear-cryptanalysis',
    'fix-code-vulnerability',
    'fix-git',
    'git-leak-recovery',
    'gpt2-codegolf',
    'headless-terminal',
    'large-scale-text-editing',
    'llm-inference-batching-scheduler',
    'log-summary-date-ranges',
    'make-mips-interpreter',
    'model-extraction-relu-logits',
    'modernize-scientific-stack',
    'multi-source-data-merger',
    'openssl-selfsigned-cert',
    'password-recovery',
    'path-tracing',
    'path-tracing-reverse',
    'polyglot-c-py',
    'polyglot-rust-c',
    'portfolio-optimization',
    'prove-plus-comm',
    'pytorch-model-recovery',
    'regex-chess',
    'regex-log',
    'sam-cell-seg',
    'sanitize-git-repo',
    'sparql-university',
    'sqlite-db-truncate',
    'torch-pipeline-parallelism',
    'torch-tensor-parallelism',
    'tune-mjcf',
    'video-processing',
    'vulnerable-secret',
    'winning-avg-corewars',
    'write-compressor',
})
"""`eval_offline`: reference 1 and no-op 0, both boxes offline, within the budget."""

GOLD_CHECK_OFFLINE: dict[str, str] = {
    'host': 'Targon cpu-medium-amd VM tb21-harbor-val-01 (8 vCPU)',
    'runtime': 'runsc release-20261109.0',
    'date': '2026-11-13',
    'budget_s': '700.0',
    'results_sha256': '05ee98032193cf7f8c62357f8b3801ab171ac2971d7e6b3ef3cc6a67f3b937b3',
    'verified': '49/89',
}
"""Provenance of `VERIFIED_OFFLINE` (host, runtime, date, results digest)."""

VERIFIED: frozenset[str] | None = frozenset({
    'adaptive-rejection-sampler',
    'bn-fit-modify',
    'break-filter-js-from-html',
    'build-pmars',
    'build-pov-ray',
    'cancel-async-tasks',
    'chess-best-move',
    'circuit-fibsqrt',
    'cobol-modernization',
    'code-from-image',
    'constraints-scheduling',
    'count-dataset-tokens',
    'crack-7z-hash',
    'custom-memory-heap-crash',
    'db-wal-recovery',
    'distribution-search',
    'dna-assembly',
    'dna-insert',
    'extract-elf',
    'extract-moves-from-video',
    'feal-differential-cryptanalysis',
    'feal-linear-cryptanalysis',
    'financial-document-processor',
    'fix-code-vulnerability',
    'fix-git',
    'git-leak-recovery',
    'gpt2-codegolf',
    'headless-terminal',
    'large-scale-text-editing',
    'largest-eigenval',
    'llm-inference-batching-scheduler',
    'log-summary-date-ranges',
    'make-doom-for-mips',
    'make-mips-interpreter',
    'merge-diff-arc-agi-task',
    'model-extraction-relu-logits',
    'modernize-scientific-stack',
    'mteb-retrieve',
    'multi-source-data-merger',
    'openssl-selfsigned-cert',
    'overfull-hbox',
    'password-recovery',
    'path-tracing',
    'path-tracing-reverse',
    'polyglot-c-py',
    'polyglot-rust-c',
    'portfolio-optimization',
    'prove-plus-comm',
    'pytorch-model-recovery',
    'raman-fitting',
    'regex-chess',
    'regex-log',
    'sam-cell-seg',
    'sanitize-git-repo',
    'sparql-university',
    'sqlite-db-truncate',
    'sqlite-with-gcov',
    'torch-pipeline-parallelism',
    'torch-tensor-parallelism',
    'train-fasttext',
    'tune-mjcf',
    'video-processing',
    'vulnerable-secret',
    'winning-avg-corewars',
    'write-compressor',
})
"""`eval`: reference 1 and no-op 0, the agent's box online through `EGRESS_PROFILE`,
grading offline, within the budget; None: not gated yet."""

GOLD_CHECK: dict[str, str] = {
    'host': 'Targon cpu-medium-amd VM tb21-harbor-egress-01 (8 vCPU)',
    'runtime': 'runsc',
    'date': '2026-11-13',
    'budget_s': '700.0',
    'results_sha256': '98e09b59c0a4d5caba618e3abdc18a44319245166ed557632f51ef83b76db762',
    'verified': '65/89',
    'egress': 'agent phase only; reliquary-sandbox 7743b57 egress-host-setup, profile packages + EGRESS_EXTRA_HOSTS, apt snapshot 20261111T000000Z, UV_EXCLUDE_NEWER and PIP_UPLOADED_PRIOR_TO 2026-11-11',
}
"""Provenance of `VERIFIED` (host, runtime, date, results digest)."""

EXCLUDED: dict[str, str] = {
    "configure-git-webserver": _LIVE_PROCESS + " (sshd + web server on 8080)",
    "git-multibranch": _LIVE_PROCESS + " (sshd + nginx on 8443)",
    "hf-model-inference": _LIVE_PROCESS + " (Flask on 5000); also downloads a model",
    "install-windows-3.11": _LIVE_PROCESS + " (QEMU + VNC + nginx)",
    "kv-store-grpc": _LIVE_PROCESS + " (gRPC server)",
    "mailman": _LIVE_PROCESS + " (postfix + mailman3)",
    "nginx-request-logging": _LIVE_PROCESS + " (nginx on 8080)",
    "pypi-server": _LIVE_PROCESS + " (package index on 8080)",
    "qemu-alpine-ssh": _LIVE_PROCESS + " (QEMU VM with sshd)",
    "qemu-startup": _LIVE_PROCESS + " (QEMU VM on telnet 6665)",
}
"""Task id -> why it is never served."""

NOT_VERIFIED: dict[str, str] = {
    # Over the state cap (MAX_STATE_BYTES, 512 MB) once the agent's installs are captured.
    **{t: "reference with egress: captured state over 512 MB" for t in (
        "caffe-cifar-10", "gcode-to-text", "mteb-leaderboard", "pytorch-model-cli")},
    # The reference fails with egress (hosts beyond the allowlist, or the task itself).
    "build-cython-ext": "reference with egress: 10/11 tests",
    "compile-compcert": "reference with egress: needs opam/coq hosts, then outlasts its agent timeout",
    "mcmc-sampling-stan": "reference with egress: 2/6 tests",
    "protein-assembly": "reference with egress: needs data.rcsb.org, fpbase.org, pubchem (0/1)",
    "rstan-to-pystan": "reference with egress: 1/6 tests",
    # Network at test time, inside the tests themselves (grading is offline).
    "filter-js-from-html": "a test downloads a GitHub archive",
    "reshard-c4-data": "a test loads a Hugging Face dataset",
    # Grading over the 810 s step on 1 CPU under gVisor.
    "fix-ocaml-gc": "test.sh rebuilds OCaml: grading over 810 s",
    "query-optimize": "test.sh's queries: grading over 810 s",
    "schemelike-metacircular-eval": "tests take 640 s and 5 time out under gVisor",
}
"""Task id -> why the gold check did not verify it for `eval` (informational, from its runs).
`eval_offline` also leaves out the 16 tasks whose reference needs network to solve."""
