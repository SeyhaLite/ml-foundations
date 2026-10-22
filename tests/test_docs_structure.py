"""Documentation structure tests: Sphinx config, toctree integrity, links,
placeholders, and the guarantee that a docs build never executes notebooks."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def all_docs_pages():
    return sorted(DOCS.rglob("*.md"))


def test_docs_layout_exists():
    assert (DOCS / "conf.py").is_file()
    assert (DOCS / "index.md").is_file()
    assert (DOCS / "requirements.txt").is_file()
    assert (DOCS / "chapters").is_dir()


def test_conf_py_settings():
    conf = (DOCS / "conf.py").read_text(encoding="utf-8")
    assert "myst_nb" in conf, "conf.py must load myst-nb (for .ipynb rendering)"
    # myst_nb already loads myst_parser; listing both double-runs myst's setup
    # and crashes Sphinx (registry.transforms.remove called twice).
    assert '"myst_parser"' not in conf, (
        "conf.py must NOT list myst_parser in extensions — myst_nb replaces it"
    )
    # CRITICAL: docs builds must never execute notebooks (no GPU / no downloads on RTD)
    assert re.search(r'nb_execution_mode\s*=\s*["\']off["\']', conf), \
        "nb_execution_mode must be 'off' — RTD builds must not run training code"
    assert 'exclude_patterns' in conf
    assert not re.search(r'^\s*source_suffix\s*=', conf, re.M), (
        "conf.py must not override source_suffix — myst_nb registers .md/.ipynb itself"
    )
    # Dark-mode-only UI: RTD theme + override stylesheet loaded after theme CSS
    assert '"sphinx_rtd_theme"' in conf, "html_theme must be sphinx_rtd_theme"
    assert "dark-theme.css" in conf, "dark-theme.css override must be listed in html_css_files"
    assert (DOCS / "_static" / "dark-theme.css").is_file(), "docs/_static/dark-theme.css missing"


def test_chapter_files_present():
    chapters = sorted((DOCS / "chapters").glob("ch*.md"))
    assert len(chapters) >= 8, "expected at least 8 lesson chapters"
    numbering = [int(re.match(r"ch(\d+)", p.name).group(1)) for p in chapters]
    assert numbering == sorted(numbering), "chapter files must be numbered consecutively"
    assert numbering == list(range(1, len(numbering) + 1)), "chapter numbering must start at 1 without gaps"


@pytest.mark.parametrize("page", all_docs_pages(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_placeholder_content(page: Path):
    text = page.read_text(encoding="utf-8").lower()
    for bad in ("todo:", "lorem ipsum", "fixme", "coming soon", "xxx", "placeholder"):
        assert bad not in text, f"{page}: placeholder text {bad!r} found"


@pytest.mark.parametrize("page", all_docs_pages(), ids=lambda p: str(p.relative_to(ROOT)))
def test_single_h1(page: Path):
    """Exactly one H1 per page — ignoring '# ' comments inside code fences."""
    lines = page.read_text(encoding="utf-8").splitlines()
    in_fence = False
    h1 = []
    for ln in lines:
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and re.match(r"^# [^#]", ln):
            h1.append(ln)
    assert len(h1) == 1, f"{page}: expected exactly one H1 title, got {len(h1)}: {h1}"


def test_index_toctree_targets_exist():
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    toc = re.search(r"```{toctree}(.*?)```", index, re.S)
    assert toc, "index.md must contain a toctree directive"
    entries = [
        ln.strip() for ln in toc.group(1).splitlines()
        if ln.strip() and not ln.strip().startswith(":")
    ]
    assert entries, "toctree is empty"
    for entry in entries:
        # toctree entries are paths without extension; myst resolves .md/.ipynb
        candidates = [
            DOCS / f"{entry}.md",
            DOCS / f"{entry}.ipynb",
            ROOT / f"{entry}.ipynb",
            ROOT / f"{entry}.md",
        ]
        assert any(c.exists() for c in candidates), f"toctree target missing: {entry}"


def test_all_notebooks_in_toctree():
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    for nb in sorted((DOCS / "notebooks").glob("*.ipynb")):
        ref = f"notebooks/{nb.stem}"
        assert ref in index, f"notebook {nb.name} not referenced from docs/index.md toctree"


def _iter_relative_md_links():
    link_re = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:|#|<)([^)#\s]+)")
    for page in all_docs_pages():
        for target in link_re.findall(page.read_text(encoding="utf-8")):
            yield page, target


def test_relative_links_resolve():
    broken = []
    for page, target in _iter_relative_md_links():
        t = target.rstrip("/")
        if not t:
            continue
        resolved = (page.parent / t).resolve()
        if not resolved.exists():
            broken.append(f"{page.relative_to(ROOT)} -> {target}")
    assert not broken, "broken relative links:\n" + "\n".join(broken)


def test_readme_links_resolve():
    readme = ROOT / "README.md"
    assert readme.is_file()
    link_re = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:|#|<)([^)#\s]+)")
    broken = []
    for target in link_re.findall(readme.read_text(encoding="utf-8")):
        if not (ROOT / target.rstrip("/")).exists():
            broken.append(target)
    assert not broken, f"README broken links: {broken}"


def test_docs_mention_all_chapters_from_index_or_toc():
    """Every chapter file must be reachable from the toctree (no orphan pages)."""
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    for ch in sorted((DOCS / "chapters").glob("ch*.md")):
        ref = f"chapters/{ch.stem}"
        assert ref in index, f"orphan chapter: {ch.name} not in docs/index.md toctree"


def test_no_gpu_or_download_in_docs_conf():
    conf = (DOCS / "conf.py").read_text(encoding="utf-8")
    assert "from_pretrained" not in conf
    assert "import torch" not in conf


def test_conf_math_rendering_enabled():
    """$...$ math silently renders as literal text unless dollarmath+mathjax are on."""
    conf = (DOCS / "conf.py").read_text(encoding="utf-8")
    assert '"dollarmath"' in conf, "myst_enable_extensions must include dollarmath"
    assert "sphinx.ext.mathjax" in conf, "extensions must include sphinx.ext.mathjax"


_FENCE_INFO_RE = re.compile(r"^\s{0,3}(```+|~~~+)\s*(\{[^}]*\}\s*.*)$")


def _iter_markdown_texts():
    """Yield (where, text) for every markdown source: docs pages + notebook md cells."""
    for page in all_docs_pages():
        yield str(page.relative_to(ROOT)), page.read_text(encoding="utf-8")
    import json

    for nb in sorted((DOCS / "notebooks").glob("*.ipynb")):
        data = json.loads(nb.read_text(encoding="utf-8"))
        for ci, cell in enumerate(data["cells"]):
            if cell["cell_type"] == "markdown":
                src = cell["source"]
                yield (
                    f"{nb.name}[cell {ci}]",
                    "".join(src) if isinstance(src, list) else src,
                )


@pytest.mark.parametrize("where,text", list(_iter_markdown_texts()))
def test_directive_fence_titles_have_no_backticks(where, text):
    """CommonMark: a backtick fence's info string may NOT contain backticks.

    A title like '```{note} see `code` here' silently degrades the fence into
    a paragraph and desynchronizes every following fence in the file.
    """
    for i, ln in enumerate(text.splitlines(), 1):
        m = _FENCE_INFO_RE.match(ln)
        if m:
            info_after_directive = m.group(2).split("}", 1)[-1]
            assert "`" not in info_after_directive, (
                f"{where}:{i}: backtick in directive fence title desyncs parsing: {ln.strip()!r}"
            )


def test_deployment_and_compatibility_pages_exist():
    assert (DOCS / "deployment.md").is_file(), "Khmer RTD deployment guide missing"
    assert (DOCS / "compatibility.md").is_file(), "model compatibility matrix missing"
    assert (DOCS / "references.md").is_file()
