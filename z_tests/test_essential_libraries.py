import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Python Basics - Essential Libraries"))
from essential_libraries import is_available, library_overview, run_demo


def test_overview_has_ten_entries():
    assert len(library_overview()) == 10


def test_overview_values_are_nonempty_strings():
    for desc in library_overview().values():
        assert isinstance(desc, str) and desc.strip()


def test_overview_includes_core_names():
    keys = library_overview()
    for name in ("numpy", "pandas", "json", "math"):
        assert name in keys


def test_is_available_true_for_stdlib():
    assert is_available("math") is True
    assert is_available("json") is True
    assert is_available("datetime") is True
    assert is_available("random") is True


def test_is_available_false_for_nonsense():
    assert is_available("no_such_pkg_xyz") is False


def test_run_demo_missing_is_graceful():
    msg = run_demo("no_such_pkg_xyz")
    assert "not installed" in msg


def test_run_demo_math_runs():
    assert "math" in run_demo("math")


def test_run_demo_json_round_trip():
    assert "json" in run_demo("json")
