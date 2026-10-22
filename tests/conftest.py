"""Shared fixtures — make ``qwen_ft`` importable even without ``pip install -e .``."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def sample_dataset(repo_root: Path) -> Path:
    p = repo_root / "data" / "sample_khmer_assistant.jsonl"
    assert p.is_file(), "sample dataset missing"
    return p


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    p = repo_root / "configs"
    assert p.is_dir(), "configs/ missing"
    return p
