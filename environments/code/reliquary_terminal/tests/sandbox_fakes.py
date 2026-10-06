"""A scripted TaskRuntime for sandbox hook tests: no box, no Docker, nothing runs."""

from types import SimpleNamespace


class ScriptedRuntime:
    def __init__(self, workdir="/app"):
        self.config = SimpleNamespace(type="reliquary-sandbox", workdir=workdir)
        self.env = {}
        self.runs = []
        self.files = {}
        self._answers = []
        self.archived = None
        self.restored = None

    def on(self, predicate, exit_code=0, stdout="", stderr=""):
        self._answers.append((predicate, SimpleNamespace(exit_code=exit_code, stdout=stdout,
                                                         stderr=stderr)))

    async def run(self, argv, env):
        self.runs.append((list(argv), dict(env)))
        for predicate, result in self._answers:
            if predicate(list(argv)):
                return result
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def read(self, path, max_bytes=None):
        if path not in self.files:
            raise FileNotFoundError(path)
        data = self.files[path]
        if isinstance(data, BaseException):
            raise data
        if max_bytes is not None and len(data) > max_bytes:
            raise OSError(27, f"{path} exceeds {max_bytes} bytes")
        return data

    async def write(self, path, data):
        self.files[path] = bytes(data)

    async def archive(self, paths, max_bytes):
        self.archived = (list(paths), max_bytes)
        return b"ARCHIVE:" + ",".join(paths).encode()

    async def restore_archive(self, data, roots):
        self.restored = (bytes(data), list(roots))
