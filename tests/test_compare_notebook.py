# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import importlib.util
from pathlib import Path

from deepdiff import DeepDiff


def _load_compare_notebook_module():
    module_path = Path(__file__).with_name("compare_notebook.py")
    spec = importlib.util.spec_from_file_location("compare_notebook", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_notebook = _load_compare_notebook_module()


def test_tolerates_text_float_rounding_drift() -> None:
    assert compare_notebook._strings_differ_only_by_tolerated_float_rounding(
        "Mixed layer depth (m)   23.491993   23.841009",
        "Mixed layer depth (m)   23.491991   23.841005",
    )


def test_rejects_material_text_float_changes() -> None:
    assert not compare_notebook._strings_differ_only_by_tolerated_float_rounding(
        "Mixed layer depth (m)   23.491993",
        "Mixed layer depth (m)   23.491000",
    )


def test_rejects_non_numeric_text_changes() -> None:
    assert not compare_notebook._strings_differ_only_by_tolerated_float_rounding(
        "Mixed layer depth (m)   23.491993",
        "Temperature (degC)   23.491991",
    )


def test_keeps_non_tolerated_diff_entries() -> None:
    diff = DeepDiff(
        {"cell": "Mixed layer depth (m)   23.491993", "other": "Temperature   12.0"},
        {"cell": "Mixed layer depth (m)   23.491991", "other": "Temperature   13.0"},
    )

    filtered_diff = compare_notebook._remove_tolerated_text_float_changes(diff)

    assert "root['cell']" not in filtered_diff["values_changed"]
    assert "root['other']" in filtered_diff["values_changed"]
