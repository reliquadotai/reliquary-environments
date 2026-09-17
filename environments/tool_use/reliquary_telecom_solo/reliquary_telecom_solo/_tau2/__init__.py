"""τ²-bench's telecom domain, vendored.

Every module here carries a header saying which upstream file it came from and
what was changed. Vendored rather than imported so that the code a participant
runs is the code this package shipped: `tau2` is a research harness that moves,
and an environment whose tools changed underneath it is a different
environment, which a participant must not discover at grading time.

Only the solo telecom path is carried across. The LLM agent framework, the user
simulator, the orchestrator, the CLI and the other five domains are not here.

Upstream: https://github.com/sierra-research/tau2-bench, MIT, © Sierra
Research. The licence text is `LICENSE`, beside this file.
"""
