"""Validate .readthedocs.yaml against the current official RTD config schema (v2).

Reference: https://docs.readthedocs.com/platform/stable/config-file/v2.html
Checked 2026-10-22 — the official example uses version 2, build.os
ubuntu-26.04 and tools.python "3.14".
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RTD_YAML = ROOT / ".readthedocs.yaml"

SUPPORTED_OS = {"ubuntu-22.04", "ubuntu-24.04", "ubuntu-26.04"}
SUPPORTED_PY = {"3.10", "3.11", "3.12", "3.13", "3.14"}


@pytest.fixture(scope="module")
def rtd_cfg():
    assert RTD_YAML.is_file(), ".readthedocs.yaml must exist at the repository root"
    return yaml.safe_load(RTD_YAML.read_text(encoding="utf-8"))


def test_version_2(rtd_cfg):
    assert rtd_cfg.get("version") == 2, "RTD requires 'version: 2' (v1 is rejected)"


def test_build_os_and_python(rtd_cfg):
    build = rtd_cfg.get("build") or {}
    assert build.get("os") in SUPPORTED_OS, f"unsupported build.os: {build.get('os')}"
    tools = build.get("tools") or {}
    py = str(tools.get("python"))
    assert py in SUPPORTED_PY, f"unsupported python tool version: {py}"


def test_sphinx_configuration_path(rtd_cfg):
    sphinx = rtd_cfg.get("sphinx") or {}
    conf = sphinx.get("configuration")
    assert conf, "sphinx.configuration is required for a Sphinx project"
    assert (ROOT / conf).is_file(), f"sphinx configuration not found: {conf}"


def test_python_install_paths_exist(rtd_cfg):
    installs = (rtd_cfg.get("python") or {}).get("install") or []
    assert installs, "declare python.install requirements for reproducible builds"
    for item in installs:
        if "requirements" in item:
            p = ROOT / item["requirements"]
            assert p.is_file(), f"requirements file missing: {item['requirements']}"
        if "path" in item:
            assert (ROOT / item["path"]).exists(), f"install path missing: {item['path']}"


def test_docs_requirements_are_light():
    """RTD builds must NOT pull torch/transformers — docs never execute notebooks."""
    req = (ROOT / "docs" / "requirements.txt").read_text(encoding="utf-8").lower()
    for heavy in ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"):
        lines = [
            ln for ln in req.splitlines()
            if ln.strip() and not ln.strip().startswith("#") and heavy in ln
        ]
        assert not lines, f"docs/requirements.txt must not install {heavy!r}: {lines}"


def test_no_unknown_top_level_keys(rtd_cfg):
    allowed = {"version", "build", "sphinx", "python", "formats", "conda", "submodules", "search"}
    unknown = set(rtd_cfg) - allowed
    assert not unknown, f"unknown top-level keys in .readthedocs.yaml: {unknown} (RTD fails on these)"


def test_formats_if_present(rtd_cfg):
    fmt = rtd_cfg.get("formats")
    if fmt is not None:
        assert fmt == "all" or set(fmt) <= {"htmlzip", "pdf", "epub"}
