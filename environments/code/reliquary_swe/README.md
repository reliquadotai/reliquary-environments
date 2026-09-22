# reliquary-swe

SWE-bench Verified repair tasks: an agent is given a real GitHub issue and a
repository at its pre-fix commit, and is graded by running the tests the
human fix added or restored.

This package is under construction. This first module, `reliquary_swe.corpus`,
loads the pinned SWE-bench Verified rows and nothing else — it is importable
without Docker and without a Verifiers runtime, so questions about the
corpus itself (row count, repositories, task ids) can be answered on a
machine that cannot host containers. The container-backed grading and the
Verifiers `Taskset` land in later modules.
