"""A scripted TaskRuntime for sandbox hook tests: no box, no Docker, nothing runs.

Workdir-aware, like a box: `run` raises when `config.workdir` does not exist (a
docker exec into a deleted workdir fails). Paths exist when they are `/`, a
file in `files` or a directory in `dirs`; reading a directory raises
IsADirectoryError, as the sandbox's helper does."""

from types import SimpleNamespace


class ScriptedRuntime:
    def __init__(self, workdir="/app"):
        self.config = SimpleNamespace(type="reliquary-sandbox", workdir=workdir)
        self.env = {}
        self.runs = []
        self.reads = []
        self.log = []
        """Every run, read and stop_processes, in order."""
        self.files = {}
        self.dirs = set()
        self._answers = []
        self.archived = None
        self.restored = None

    def on(self, predicate, exit_code=0, stdout="", stderr=""):
        self._answers.append((predicate, SimpleNamespace(exit_code=exit_code, stdout=stdout,
                                                         stderr=stderr)))

    def exists(self, path):
        return path == "/" or path in self.files or path in self.dirs

    async def run(self, argv, env):
        if not self.exists(self.config.workdir):
            raise RuntimeError(f"the workdir {self.config.workdir} does not exist")
        self.runs.append((list(argv), dict(env)))
        self.log.append(("run", list(argv)))
        for predicate, result in self._answers:
            if predicate(list(argv)):
                return result
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def read(self, path, max_bytes=None):
        self.reads.append((path, max_bytes))
        self.log.append(("read", path))
        if path in self.dirs:
            raise IsADirectoryError(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        data = self.files[path]
        if isinstance(data, BaseException):
            raise data
        if max_bytes is not None and len(data) > max_bytes:
            raise OSError(27, f"{path} exceeds {max_bytes} bytes")
        return data

    async def stop_processes(self):
        self.log.append(("stop_processes",))

    async def write(self, path, data):
        self.files[path] = bytes(data)

    async def archive(self, paths, max_bytes):
        self.archived = (list(paths), max_bytes)
        return b"ARCHIVE:" + ",".join(paths).encode()

    async def restore_archive(self, data, roots):
        self.restored = (bytes(data), list(roots))
