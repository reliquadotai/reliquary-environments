"""The test-only `episode_task_shim` must match the real `reliquary_sandbox.episode_task`:
same public names, same signatures, same dataclass fields and constants. Runs wherever the
real module is installed (a dev machine), so drift fails there before CI runs on the shim."""

import dataclasses
import importlib
import inspect
import re
from pathlib import Path

import episode_task_shim as shim
import pytest

SHIM = Path(shim.__file__).resolve()


def _real():
    module = importlib.import_module("reliquary_sandbox.episode_task")
    if Path(module.__file__).resolve() == SHIM:
        pytest.skip("reliquary-sandbox is not installed here: the tests run on the shim")
    return module


def _public(module):
    return {name: value for name, value in vars(module).items()
            if not name.startswith("_") and not inspect.ismodule(value)}


def _shape(value, module_name):
    """A comparable description of a public value, its defining module's name removed."""
    if inspect.isclass(value):
        members = {name: _shape(member, module_name) for name, member in vars(value).items()
                   if not name.startswith("_") or name in ("__init__", "__post_init__")}
        fields = ([(f.name, str(f.type), repr(f.default), f.default_factory is not
                    dataclasses.MISSING) for f in dataclasses.fields(value)]
                  if dataclasses.is_dataclass(value) else None)
        return ("class", [base.__name__ for base in value.__mro__], fields, members)
    if isinstance(value, (property, staticmethod, classmethod)):
        return (type(value).__name__, _shape(value.__func__ if hasattr(value, "__func__")
                                             else value.fget, module_name))
    if inspect.isfunction(value):
        return ("function", str(inspect.signature(value)))
    if isinstance(value, re.Pattern):
        return ("pattern", value.pattern, value.flags)
    return ("value", repr(value).replace(module_name + ".", ""))


def test_the_shim_matches_the_real_episode_task():
    real = _real()
    real_public, shim_public = _public(real), _public(shim)
    assert sorted(shim_public) == sorted(real_public)
    for name, value in real_public.items():
        assert (_shape(shim_public[name], shim.__name__) == _shape(value, real.__name__)), name


def test_the_shim_is_installed_only_without_the_real_module():
    module = importlib.import_module("reliquary_sandbox.episode_task")
    assert hasattr(module, "EnvInfraError") and hasattr(module, "SandboxTask")
