"""Import smoke: every module under ``src/memory_kernel`` imports in a fresh process, and every lazy export resolves.

One fresh interpreter per module, because in-process imports hide import cycles and ordering bugs: ``sys.modules``
is already warm at collection. The expected failure set is EMPTY. A
module that cannot import is a bug or a dead file; delete it or fix it, do not allowlist it.

The children are independent, so the first test to need a result runs all of them 8 at a time and every test then
reads its own module's outcome. That turns ~100 s of serial interpreter start-ups into ~20 s.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
PACKAGE = SRC / "memory_kernel"
TIMEOUT_S = 120
WORKERS = 8


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _source_files() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def _defines_module_getattr(init: Path) -> bool:
    return any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in ast.parse(init.read_text()).body)


MODULES = [_module_name(p) for p in _source_files()]
LAZY_PACKAGES = [_module_name(p) for p in _source_files() if p.name == "__init__.py" and _defines_module_getattr(p)]

#: Resolves every name a package promises lazily (``__all__`` and the ``_LAZY_EXPORTS`` table) in the child.
LAZY_CHILD = """
import importlib, json, sys
pkg = importlib.import_module(sys.argv[1])
names = set(getattr(pkg, "__all__", ()) or ())
table = getattr(pkg, "_LAZY_EXPORTS", None)
if isinstance(table, dict):
    names |= set(table)
bad = {}
for name in sorted(names):
    try:
        getattr(pkg, name)
    except BaseException as exc:
        bad[name] = f"{type(exc).__name__}: {exc}"
print(json.dumps({"checked": len(names), "bad": bad}))
sys.exit(1 if bad else 0)
"""


def _python(child_env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "<no output>"


def test_the_tree_under_test_is_this_repo(child_env):
    """A child must import THIS tree's ``memory_kernel``, not an editable install elsewhere on the machine."""
    done = _python(child_env, "-c", "import memory_kernel; print(memory_kernel.__file__)")
    assert done.returncode == 0, done.stderr
    assert Path(done.stdout.strip()).resolve() == (PACKAGE / "__init__.py").resolve()


def test_every_module_was_found():
    assert "memory_kernel" in MODULES
    assert "memory_kernel.cli" in MODULES
    assert len(MODULES) == len(set(MODULES))


@pytest.fixture(scope="session")
def import_outcomes(child_env) -> dict[str, str | None]:
    """``{module: None}`` when ``import <module>`` succeeded in its own fresh process, else the error's last line."""

    def attempt(module: str) -> str | None:
        try:
            done = _python(child_env, "-c", f"import {module}")
        except subprocess.TimeoutExpired:
            return f"timeout after {TIMEOUT_S}s"
        return None if done.returncode == 0 else _last_line(done.stderr)

    with ThreadPoolExecutor(WORKERS) as pool:
        return dict(zip(MODULES, pool.map(attempt, MODULES), strict=True))


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_in_a_fresh_process(module, import_outcomes):
    assert import_outcomes[module] is None, f"import {module}: {import_outcomes[module]}"


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_every_lazy_export_resolves(package, child_env):
    done = _python(child_env, "-c", LAZY_CHILD, package)
    assert done.returncode == 0, f"{package}: {_last_line(done.stderr)}\n{done.stdout}"
    assert json.loads(done.stdout)["bad"] == {}
