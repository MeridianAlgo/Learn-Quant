import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Python Basics - Imports and Modules"))
from import_basics import (
    classify,
    is_installed,
    module_origin,
    safe_import,
    search_paths,
)


def test_safe_import_returns_module():
    mod = safe_import("math")
    assert mod is not None
    assert mod.sqrt(144) == 12


def test_safe_import_missing_returns_none():
    assert safe_import("no_such_module_xyz") is None


def test_is_installed_true_for_stdlib():
    assert is_installed("json") is True


def test_is_installed_false_for_nonsense():
    assert is_installed("no_such_module_xyz") is False


def test_module_origin_nonempty_for_math():
    origin = module_origin("math")
    assert isinstance(origin, str) and origin


def test_module_origin_not_found():
    assert module_origin("no_such_module_xyz") == "not found"


def test_classify_stdlib():
    assert classify("json") == "standard library"


def test_classify_not_found():
    assert classify("no_such_module_xyz") == "not found"


def test_classify_third_party_for_numpy():
    # numpy is installed on this repo, so it should classify as third party.
    assert classify("numpy") == "third party"


def test_search_paths_is_nonempty_list():
    paths = search_paths()
    assert isinstance(paths, list) and len(paths) > 0
