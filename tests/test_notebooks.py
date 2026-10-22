"""Structural integrity tests for the Jupyter notebooks in ``notebooks/``.

These tests NEVER execute the notebooks (no GPU, no model downloads). They
check:

* every ``.ipynb`` is valid nbformat v4 JSON (``nbformat.validate``);
* kernelspec metadata is python3;
* each notebook has a title cell, a reasonable number of cells, and both
  markdown + code cells (an "educational" notebook);
* notebooks contain no stored outputs (clean, reviewable, small diffs);
* heavy operations (transformers/torch imports, from_pretrained, .train())
  only appear in markdown code fences or environment-guarded cells, never in
  cells that a docs build or ``jupyter nbconvert --execute`` would run blindly;
* relative file references inside markdown cells point at existing files;
* the Colab badge link matches the canonical GitHub URL of the notebook.
"""

import json
import re
from pathlib import Path

import pytest

try:
    import nbformat
except ImportError:  # pragma: no cover
    nbformat = None

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "docs" / "notebooks").glob("*.ipynb"))

GITHUB_RAW_BASE = "https://colab.research.google.com/github/SeyhaLite/ml-foundations/blob/main/docs/notebooks/"

HEAVY_TOKENS = re.compile(
    r"\b(import\s+torch|from\s+torch|import\s+transformers|from\s+transformers"
    r"|import\s+trl|from\s+trl|import\s+peft|from\s+peft"
    r"|from_pretrained\s*\(|\.train\s*\(\s*\))"
)
GUARD_TOKEN = "QWEN_FT_ALLOW_HEAVY"


def setup_module(module):
    assert NOTEBOOKS, f"no notebooks found in {ROOT / 'docs' / 'notebooks'}"


def _cells(nb_path: Path):
    with nb_path.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.skipif(nbformat is None, reason="nbformat not installed")
@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_nbformat_valid(nb_path: Path):
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_kernelspec_python3(nb_path: Path):
    nb = _cells(nb_path)
    ks = nb.get("metadata", {}).get("kernelspec", {})
    assert ks.get("language") == "python"
    assert ks.get("name") in {"python3", "python"}


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_has_title_and_enough_cells(nb_path: Path):
    nb = _cells(nb_path)
    cells = nb["cells"]
    assert len(cells) >= 5, f"{nb_path.name}: too few cells for a lesson"

    def source_of(cell):
        src = cell["source"]
        return "".join(src) if isinstance(src, list) else src

    first_md = next(c for c in cells if c["cell_type"] == "markdown")
    title_lines = [ln for ln in source_of(first_md).splitlines() if ln.startswith("# ")]
    assert title_lines, "first markdown cell must contain an H1 title line"
    kinds = {c["cell_type"] for c in cells}
    assert {"markdown", "code"} <= kinds, "notebook must mix explanation and code"


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_stored_outputs(nb_path: Path):
    nb = _cells(nb_path)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            assert c.get("outputs", []) == [], f"{nb_path.name} cell {i} has stored outputs"
            assert c.get("execution_count") is None, f"{nb_path.name} cell {i} has execution_count"


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_unGuarded_heavy_code_cells(nb_path: Path):
    """Docs builds and CI must never trigger model downloads or training."""
    nb = _cells(nb_path)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        # strip comments: heavy ops may be mentioned in commented demo lines
        stripped = "\n".join(
            ln for ln in src.splitlines() if not ln.strip().startswith("#")
        )
        if HEAVY_TOKENS.search(stripped):
            assert GUARD_TOKEN in src, (
                f"{nb_path.name} cell {i} runs heavy ML code without the "
                f"{GUARD_TOKEN} environment guard"
            )


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_colab_badge_matches_path(nb_path: Path):
    text = nb_path.read_text(encoding="utf-8")
    expected = GITHUB_RAW_BASE + nb_path.name
    assert expected in text, f"{nb_path.name}: missing or wrong Colab badge link"


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_relative_links_exist(nb_path: Path):
    """Markdown links like ../configs/x.yaml or ../docs/chapters/y.md must resolve."""
    nb = _cells(nb_path)
    link_re = re.compile(r"\]\((\.\.?/[^)#\s]+)")
    for c in nb["cells"]:
        if c["cell_type"] != "markdown":
            continue
        for rel in link_re.findall("".join(c["source"])):
            target = (nb_path.parent / rel).resolve()
            assert target.exists(), f"{nb_path.name}: broken relative link {rel}"


def test_expected_notebook_set():
    """Guard against accidental deletion/renaming of the lesson notebooks."""
    names = {p.name for p in NOTEBOOKS}
    expected = {
        "01_hello_qwen_tokenizer.ipynb",
        "02_chat_templates.ipynb",
        "03_dataset_prep.ipynb",
        "04_lora_config_lab.ipynb",
        "05_smoke_training.ipynb",
        "06_inference_and_export.ipynb",
    }
    assert expected <= names, f"missing notebooks: {expected - names}"
