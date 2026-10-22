"""Secret-scanning test: no credentials may ever be committed.

Scans every tracked text file for common credential patterns (GitHub PATs,
HF tokens, AWS keys, OpenAI-style keys, generic ``token =`` assignments, and
credentials embedded in URLs).
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {".git", "__pycache__", "docs/_build", ".pytest_cache", "node_modules", "*.egg-info"}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".ipynb", ".sh", ".rst", ".css", ".html", ".gitignore",
}

PATTERNS = [
    (r"github_pat_[A-Za-z0-9_]{20,}", "GitHub fine-grained PAT"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub classic PAT"),
    (r"gho_[A-Za-z0-9]{36}", "GitHub OAuth token"),
    (r"hf_[A-Za-z]{34}", "Hugging Face token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI-style secret key"),
    # assembled by concatenation so this source file does not match itself
    ("x-access-token:" + r"[^@\s]+@", "credential embedded in git URL"),
    (r"//[^/\s:]+:[^@\s]{20,}@github\.com", "user:password embedded in GitHub URL"),
]
COMPILED = [(re.compile(p), label) for p, label in PATTERNS]


def _iter_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = set(path.relative_to(ROOT).parts[:-1])
        if rel_parts & {".git", "__pycache__", "_build", ".pytest_cache", "node_modules"}:
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", "LICENSE"}:
            yield path


def test_text_files_found():
    files = list(_iter_text_files())
    assert len(files) > 20, "suspiciously few files scanned — check SKIP rules"


@pytest.mark.parametrize("pattern,label", [(p, l) for p, l in PATTERNS], ids=[l for _, l in PATTERNS])
def test_no_secret_pattern(pattern, label):
    rx = re.compile(pattern)
    hits = []
    for path in _iter_text_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{i} [{label}]")
    assert not hits, "possible secret committed:\n" + "\n".join(hits)


def test_git_remote_has_no_credentials():
    cfg = ROOT / ".git" / "config"
    if cfg.is_file():
        text = cfg.read_text(encoding="utf-8", errors="ignore")
        assert "github_pat_" not in text, "git remote URL contains a PAT — reset it to the clean HTTPS URL"
        assert "x-access-token:" not in text
