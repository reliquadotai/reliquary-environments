"""grading.grade in two halves: `prepare_box` (the pristine grading box, before the agent's
patch exists there) and `grade_prepared` (the patch). A sandbox runs the first as the
task's `grading_setup`, so its failures are ours. Patches travel byte for byte, and a patch
carrying a binary file, a symlink or a submodule is refused (graded 0)."""

from types import SimpleNamespace

import pytest

from reliquary_swe import grading
from reliquary_swe.taskset import SweData


SHA = "b" * 40


def data(split="r2e", base_commit=SHA):
    return SweData(idx=0, prompt="p", image="r@sha256:" + "a" * 64, workdir="/testbed",
                   instance_id="i", repo="r", base_commit=base_commit, version="1",
                   fail_to_pass=("t.py::a",), pass_to_pass=(), gold_patch="", test_patch="",
                   split=split, test_command="bash t.sh")


class Box:
    """Records commands and files; answers commands by predicate (default: exit 0)."""

    def __init__(self):
        self.runs, self.files, self._answers = [], {}, []

    def on(self, predicate, exit_code=0, stdout="", stderr=""):
        self._answers.append((predicate, SimpleNamespace(exit_code=exit_code, stdout=stdout,
                                                         stderr=stderr)))

    def on_call(self, predicate, answer):
        """`answer(argv)` gives the result."""
        self._answers.append((predicate, answer))

    async def run(self, argv, env):
        self.runs.append(list(argv))
        for predicate, result in self._answers:
            if predicate(list(argv)):
                return result(list(argv)) if callable(result) else result
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def write(self, path, data_):
        self.files[path] = bytes(data_)

    async def read(self, path, max_bytes=None):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


TEXT = (b"diff --git a/new.py b/new.py\nnew file mode 100644\nindex 0000000..e69de29\n"
        b"--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+print('new file mode 120000')\n")


@pytest.mark.parametrize("raw, kind", [
    (b"diff --git a/l b/l\nnew file mode 120000\nindex 0000000..1111111\n", "symlink"),
    (b"diff --git a/l b/l\nold mode 100644\nnew mode 120000\n", "symlink"),
    (b"diff --git a/l b/l\nindex 1111111..2222222 120000\n--- a/l\n+++ b/l\n", "symlink"),
    (b"diff --git a/l b/l\ndeleted file mode 120000\nindex 1111111..0000000\n", "symlink"),
    (b"diff --git a/m b/m\nnew file mode 160000\nindex 0000000..1111111\n", "submodule"),
    (b"diff --git a/x.bin b/x.bin\nnew file mode 100644\nGIT binary patch\nliteral 3\n", "binary"),
    (b"diff --git a/x b/x\nBinary files a/x and b/x differ\n", "binary"),
    (b"diff --git a/x b/x\nFiles a/x and b/x differ\n", "binary"),
])
def test_binary_symlink_and_submodule_patches_are_refused(raw, kind):
    assert kind in grading.patch_shape_violations(raw)


@pytest.mark.parametrize("header", [
    # git reads a mode with strtoul(..., 8): leading blanks and zeros, a sign, CR, any
    # permission bits -- the file type is what makes the link.
    b"new file mode 0120000\n",
    b"new file mode  120000\n",
    b"new file mode 120000\r\n",
    b"new file mode 120644\n",
    b"new file mode +120000\n",
    b"index 0000000..1111111  0120000\n",
    b"index 0000000..1111111 120000\r\n",
])
def test_every_spelling_git_reads_as_a_symlink_is_refused(header):
    raw = b"diff --git a/l b/l\n" + header
    assert "symlink" in grading.patch_shape_violations(raw)


@pytest.mark.parametrize("header", [
    # strtoul skips a newline too: the mode may sit on the next line.
    b"new file mode \n120000\n",
    # The first '.' of an index line opens '..'; the mode follows the next space.
    b"index 00 00..1111111 120000\n",
])
def test_a_mode_git_finds_away_from_where_it_is_expected_is_still_read(header):
    assert "symlink" in grading.patch_shape_violations(b"diff --git a/l b/l\n" + header)


@pytest.mark.parametrize("mode", [b"000644", b"040000", b"170000", b"0",
                                  b"1" + b"0" * 24 + b"100644"])
def test_a_mode_git_turns_into_a_gitlink_is_refused(mode):
    # canon_mode: any file type but a regular file or a link is a gitlink; strtoul
    # saturates an overflow to ULONG_MAX, whose type bits are all set.
    raw = b"diff --git a/m b/m\nnew file mode " + mode + b"\n"
    assert grading.patch_shape_violations(raw) == ["submodule"]


def test_a_text_patch_adding_a_file_is_accepted_and_content_lines_never_count():
    assert grading.patch_shape_violations(TEXT) == []
    executable = (b"diff --git a/s b/s\nold mode 100644\nnew mode 100755\n"
                  b"index 1111111..2222222 100755\n--- a/s\n+++ b/s\n@@ -1 +1 @@\n"
                  b"-GIT binary patch\n+Binary files a/x and b/x differ\n")
    assert grading.patch_shape_violations(executable) == []


async def test_a_refused_shape_grades_zero_without_applying_anything():
    box = Box()
    raw = b"diff --git a/x.bin b/x.bin\nnew file mode 100644\nGIT binary patch\nliteral 3\n"
    report = await grading.grade_prepared(box, data(), raw)
    assert report.reward == 0.0 and report.applied is False
    assert report.test_output_tail == (
        "patch refused (env norm: no binary, symlink or submodule): binary")
    assert not any(argv[:2] == ["git", "apply"] for argv in box.runs)


async def test_the_patch_reaches_the_box_byte_for_byte():
    raw = b"diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-caf\xe9\n+caf\xe8\n"
    box = Box()
    box.on(lambda argv: argv[:3] == ["git", "apply", "-v"], exit_code=1, stderr="no")
    report = await grading.grade_prepared(box, data(), raw)
    assert box.files["/tmp/agent.diff"] == raw
    assert report.applied is False and report.reward == 0.0


async def test_a_str_patch_is_still_accepted():
    box = Box()
    box.on(lambda argv: argv[:3] == ["git", "apply", "-v"], exit_code=1)
    await grading.grade_prepared(box, data(), TEXT.decode())
    assert box.files["/tmp/agent.diff"] == TEXT


async def test_grade_is_prepare_box_then_grade_prepared(monkeypatch):
    calls = []

    async def prepare_box(runtime, d):
        calls.append("prepare")
        return d.model_copy(update={"base_commit": "abc"})

    async def grade_prepared(runtime, d, patch):
        calls.append(("grade", d.base_commit, patch))
        return "report"

    monkeypatch.setattr(grading, "prepare_box", prepare_box)
    monkeypatch.setattr(grading, "grade_prepared", grade_prepared)
    assert await grading.grade(object(), data(), "p") == "report"
    assert calls == ["prepare", ("grade", "abc", "p")]


@pytest.mark.parametrize("split", ["train", "polyglot", "r2e"])
async def test_prepare_box_returns_the_base_it_resolved(split):
    box = Box()
    box.on(lambda argv: argv == ["git", "rev-parse", "HEAD"], stdout="f00d\n")
    assert (await grading.prepare_box(box, data(split))).base_commit == "f00d"
    assert "/tmp/agent.diff" not in box.files


async def test_prepare_box_raises_on_a_box_that_is_not_what_the_corpus_says():
    box = Box()
    box.on(lambda argv: argv[:2] == ["git", "checkout"], exit_code=1, stderr="no such ref")
    with pytest.raises(grading.PristineBoxError):
        await grading.prepare_box(box, data("eval", "c0ffee"))


@pytest.mark.parametrize("split", ["train", "polyglot", "r2e"])
async def test_prepared_data_reads_the_base_back_from_the_prepared_box(split):
    box = Box()
    box.on(lambda argv: argv == ["git", "rev-parse", "HEAD"], stdout="f00d\n")
    assert (await grading.prepared_data(box, data(split))).base_commit == "f00d"


async def test_prepared_data_keeps_a_verified_base_commit():
    box = Box()
    assert (await grading.prepared_data(box, data("eval", "c0ffee"))).base_commit == "c0ffee"
    assert box.runs == []


async def test_prepared_data_refuses_a_box_whose_head_does_not_resolve():
    box = Box()
    box.on(lambda argv: argv == ["git", "rev-parse", "HEAD"], exit_code=128, stderr="bad")
    with pytest.raises(grading.PristineBoxError):
        await grading.prepared_data(box, data("train"))


# -- The paths a patch reads or writes: no symlink, no gitlink, UTF-8 names ----------------
#
# A patch's header can say nothing about a mode (a modeless hunk on a tracked symlink, a
# copy or a rename whose source is one, sides with differing names): git then reads the
# preimage through the link and keeps its mode. `git apply --numstat` names only the
# written path, so the sources come from `git apply --check -v`.


def _shaped_box(numstat, checking, symlinks=(), index=(), special_exit=0):
    """A box whose `git apply --numstat` prints `numstat`, `--check -v` prints `checking`,
    where `symlinks` are links in the worktree and `index` maps tracked paths to modes."""
    box = Box()
    box.on(lambda argv: argv[:3] == ["git", "apply", "--numstat"], stdout=numstat)
    box.on(lambda argv: argv[0] == "sh" and "--check -v" in argv[2], stdout=checking)

    def special(argv):
        paths = argv[4:]
        out = "".join(f"symlink {p}\0" for p in paths if p in symlinks)
        out += "".join(f"{index[p]} {'c' * 40} 0\t{p}\0" for p in paths if p in index)
        return SimpleNamespace(exit_code=special_exit, stdout=out, stderr="")

    box.on_call(lambda argv: argv[0] == "sh" and "[ -L" in argv[2], special)
    return box


MODELESS = b"diff --git a/l b/l\n--- a/l\n+++ b/l\n@@ -1 +1 @@\n-a\n+b\n"
COPY = b"diff --git a/l b/m\nsimilarity index 100%\ncopy from l\ncopy to m\n"
RENAME = b"diff --git a/l b/m\nsimilarity index 100%\nrename from l\nrename to m\n"


def _never_applied(box):
    return not any(argv[:3] == ["git", "apply", "-v"] for argv in box.runs)


@pytest.mark.parametrize("raw, numstat, checking", [
    (MODELESS, "1\t1\tl\0", "Checking patch l...\n"),
    (COPY, "0\t0\tm\0", "Checking patch l => m...\n"),
    (RENAME, "0\t0\tm\0", "Checking patch l => m...\n"),
])
async def test_a_patch_reading_or_writing_a_tracked_symlink_grades_zero(raw, numstat, checking):
    assert grading.patch_shape_violations(raw) == []  # nothing in the header says link
    box = _shaped_box(numstat, checking, symlinks={"l"}, index={"l": "120000", "a": "100644"})
    report = await grading.grade_prepared(box, data(), raw)
    assert report.reward == 0.0 and report.applied is False
    assert "l (symlink)" in report.test_output_tail
    assert _never_applied(box)


async def test_a_symlink_only_in_the_index_is_refused_too():
    box = _shaped_box("1\t1\tl\0", "Checking patch l...\n", index={"l": "120000"})
    report = await grading.grade_prepared(box, data(), MODELESS)
    assert report.reward == 0.0 and "l (symlink)" in report.test_output_tail
    assert _never_applied(box)


async def test_a_patch_into_a_gitlink_grades_zero():
    box = _shaped_box("1\t1\tsub\0", "Checking patch sub...\n", index={"sub": "160000"})
    report = await grading.grade_prepared(box, data(), MODELESS.replace(b"/l", b"/sub"))
    assert report.reward == 0.0 and "sub (submodule)" in report.test_output_tail
    assert _never_applied(box)


async def test_a_box_that_cannot_say_what_a_path_is_refuses_the_patch():
    box = _shaped_box("1\t1\tl\0", "Checking patch l...\n", special_exit=128)
    report = await grading.grade_prepared(box, data(), MODELESS)
    assert report.reward == 0.0 and report.applied is False
    assert _never_applied(box)


async def test_a_path_that_is_not_utf8_is_refused():
    box = _shaped_box("1\t1\tcaf\ufffd.py\0", "Checking patch caf\ufffd.py...\n")
    report = await grading.grade_prepared(box, data(), MODELESS)
    assert report.reward == 0.0 and report.applied is False
    assert _never_applied(box)
    assert not any(argv[0] == "sh" and "[ -L" in argv[2] for argv in box.runs)


async def test_regular_files_pass_the_path_checks_and_are_applied():
    box = _shaped_box("1\t1\tm\0", "Checking patch a => m...\n", symlinks={"l"},
                      index={"a": "100644", "l": "120000"})
    box.on(lambda argv: argv[:3] == ["git", "apply", "-v"], exit_code=1)
    await grading.grade_prepared(box, data(), MODELESS)
    special = next(argv for argv in box.runs if argv[0] == "sh" and "[ -L" in argv[2])
    assert {"a", "m"} <= set(special[4:])  # the source too, not only what numstat names
    assert not _never_applied(box)


def test_every_name_git_checks_is_read_including_quoted_and_ambiguous_ones():
    output = ('Checking patch "caf\\303\\251\\tx" => b...\n'
              "Checking patch x => y => z...\nerror: Checking patch q...\n")
    names = set(grading._checked_patch_names(output))
    assert {"caf\u00e9\tx", "b", "x", "y => z", "x => y", "z"} <= names
    assert "q" not in names


# -- grade_prepared trusts only a base prepare_box resolved ---------------------------------


@pytest.mark.parametrize("split", ["train", "polyglot", "r2e"])
@pytest.mark.parametrize("base", ["HEAD", "origin/x~1", "b" * 39, "B" * 40, "b" * 41, ""])
async def test_grade_prepared_refuses_a_base_that_is_not_a_resolved_commit(split, base):
    box = Box()
    with pytest.raises(grading.PristineBoxError):
        await grading.grade_prepared(box, data(split, base), TEXT)
    assert box.runs == [] and box.files == {}


async def test_a_verified_base_is_not_held_to_the_resolved_form():
    box = Box()
    report = await grading.grade_prepared(box, data("eval", "c0ffee"),
                                          b"diff --git a/x b/x\nGIT binary patch\n")
    assert report.reward == 0.0


# -- A box restoration did not confirm runs nothing, on every split -------------------------


@pytest.mark.parametrize("split", ["train", "polyglot", "r2e", "eval"])
async def test_a_failed_restoration_scores_zero_and_runs_nothing(split, monkeypatch):
    async def not_restored(runtime, d):
        return False

    monkeypatch.setattr(grading, "_restore_strategy_for", lambda d: not_restored)
    box = Box()  # every command would succeed: a test run would pass
    report = await grading.grade_prepared(box, data(split, "c0ffee" if split == "eval" else SHA),
                                          "")
    assert report.reward == 0.0 and report.restored is False
    assert box.runs == []
