# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import subprocess
import sys

import pytest

VIEWER_ONLY_MODULES = (
    "pyarrow",
    "py_eddy_tracker",
    "oceanbench.publish",
    "oceanbench.pyramids",
    "oceanbench.core.eddy_census",
)

# Runs in a fresh interpreter where pyarrow and py-eddy-tracker cannot be imported, as in a plain install.
_DEFAULT_PATH_SCRIPT = """
import importlib.abc
import sys


class BlockViewerExtra(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("pyarrow", "py_eddy_tracker"):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, BlockViewerExtra())

import oceanbench
import oceanbench.cli
import oceanbench.core.evaluate
import oceanbench.metrics

oceanbench.cli._build_parser()
loaded = sorted(
    name for name in sys.modules if any(name == module or name.startswith(module + ".") for module in MODULES)
)
assert not loaded, loaded
print("ok")
"""


def test_the_default_path_never_imports_the_viewer_extra() -> None:
    script = f"MODULES = {VIEWER_ONLY_MODULES!r}\n{_DEFAULT_PATH_SCRIPT}"
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_the_viewer_option_without_the_extra_fails_early_with_a_clear_error(monkeypatch) -> None:
    from oceanbench.core import evaluate

    monkeypatch.setattr(evaluate.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r"oceanbench\[viewer\]"):
        evaluate._evaluate_challenger("challenger.py", None, None, None, None, "viewer")


def test_the_evaluate_command_reports_the_missing_viewer_extra(monkeypatch, capsys) -> None:
    from oceanbench import cli
    from oceanbench.core import evaluate

    monkeypatch.setattr(evaluate.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cli, "_evaluate_all", lambda *arguments: pytest.fail("must not evaluate"))
    args = cli._build_parser()[0].parse_args(["evaluate", "challenger.py", "--viewer-artifacts-dir", "viewer"])

    assert cli._run_evaluate(args) == 1
    assert 'pip install "oceanbench[viewer]"' in capsys.readouterr().err
