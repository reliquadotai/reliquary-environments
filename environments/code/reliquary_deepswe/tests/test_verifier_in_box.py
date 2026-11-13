"""The staged verifier's `prepare` and `grade`, run as test.sh runs them, on a small
synthetic repository (its paths come from the graders' own environment overrides)."""

from __future__ import annotations

import json
import subprocess
import sys

from reliquary_deepswe import overlay, scoring, source

TEST_PATCH = """\
diff --git a/tests/test_new.py b/tests/test_new.py
new file mode 100644
--- /dev/null
+++ b/tests/test_new.py
@@ -0,0 +1,4 @@
+from mod import f
+
+def test_new():
+    assert f() == 2
"""
MODEL_PATCH = """\
diff --git a/conftest.py b/conftest.py
new file mode 100644
--- /dev/null
+++ b/conftest.py
@@ -0,0 +1 @@
+raise SystemExit("planted")
diff --git a/mod.py b/mod.py
--- a/mod.py
+++ b/mod.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 2
diff --git a/tests/test_old.py b/tests/test_old.py
--- a/tests/test_old.py
+++ b/tests/test_old.py
@@ -1,4 +1,4 @@
 from mod import f

 def test_old():
-    assert f() in (1, 2)
+    assert True
"""


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                   check=True, capture_output=True)


def box(tmp_path):
    app, tests, logs, artifacts = (tmp_path / n for n in ("app", "tests", "verifier", "artifacts"))
    for d in (app / "tests", tests, artifacts):
        d.mkdir(parents=True)
    (app / "mod.py").write_text("def f():\n    return 1\n")
    (app / "tests" / "test_old.py").write_text(
        "from mod import f\n\ndef test_old():\n    assert f() in (1, 2)\n")
    git(app, "init", "-q")
    git(app, "add", "-A")
    git(app, "commit", "-qm", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=app, capture_output=True,
                          text=True).stdout.strip()
    config = {"base_commit": base, "f2p_node_ids": ["test_new"], "p2p_node_ids": ["test_old"],
              "grade": {"format": "ctrf", "node_id": "name", "tool_label": "pytest",
                        "reports": [str(logs / "ctrf-report.json")]}}
    staged = overlay.staged_tests(source.task_ids()[0])
    (tests / "grader.py").write_bytes(staged["grader.py"])
    (tests / "upstream_grader.py").write_bytes(staged["upstream_grader.py"])
    (tests / "metadata.json").write_text(json.dumps({"language": "python", "task_id": "synthetic"}))
    (tests / "config.json").write_text(json.dumps(config))
    (tests / "test.patch").write_text(TEST_PATCH)
    (artifacts / "model.patch").write_text(MODEL_PATCH)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "APP_DIR": str(app),
           "TESTS_DIR": str(tests), "VERIFIER_DIR": str(logs), "ARTIFACTS_DIR": str(artifacts),
           "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig")}
    return app, tests, logs, config, env


def grader(tests, env, command):
    return subprocess.run([sys.executable, str(tests / "grader.py"), command], env=env,
                          capture_output=True, text=True)


def test_prepare_applies_the_patches_and_restores_test_files(tmp_path):
    app, tests, logs, _, env = box(tmp_path)
    result = grader(tests, env, "prepare")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "return 2" in (app / "mod.py").read_text()  # the candidate's source change
    assert not (app / "conftest.py").exists()  # a planted conftest is removed
    assert "f() in (1, 2)" in (app / "tests" / "test_old.py").read_text()  # restored
    assert (app / "tests" / "test_new.py").exists()  # the official test patch
    provenance = json.loads((tests / "provenance.json").read_text())
    assert set(provenance["restored_paths"]) == {"conftest.py", "tests/test_old.py"}


def test_grade_in_the_box_agrees_with_the_scoring_process(tmp_path):
    _, tests, logs, config, env = box(tmp_path)
    assert grader(tests, env, "prepare").returncode == 0
    for statuses in ({"test_new": "passed", "test_old": "passed"},
                     {"test_new": "failed", "test_old": "passed"},
                     {"test_new": "passed"}):
        report = json.dumps({"results": {"tests": [{"name": k, "status": v}
                                                   for k, v in statuses.items()]}})
        (logs / "ctrf-report.json").write_text(report)
        result = grader(tests, env, "grade")
        assert result.returncode == 0, result.stderr
        in_box = json.loads((logs / "reward.json").read_text())
        outside = scoring.assess_reports(config, "python",
                                         {config["grade"]["reports"][0]: report.encode()}, None)
        assert in_box["reward"] == outside["scores"]["reward"]
        assert in_box["reward"] == int(statuses == {"test_new": "passed", "test_old": "passed"})


def test_an_unappliable_patch_grades_0_without_running_tests(tmp_path):
    app, tests, logs, _, env = box(tmp_path)
    (tmp_path / "artifacts" / "model.patch").write_text(
        "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n-nope\n+x\n")
    assert grader(tests, env, "prepare").returncode == 0
    assert json.loads((logs / "reward.json").read_text())["apply_failed"] == 1
