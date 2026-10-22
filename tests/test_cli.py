"""End-to-end tests for the ``qwen-ft`` CLI (pure-Python subcommands only)."""

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

from qwen_ft.cli import main


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def test_env_reports_stack():
    code, out, _ = run(["env"])
    assert code == 0
    assert "qwen-ft" in out
    assert "transformers" in out


def test_models_lists_registry():
    code, out, _ = run(["models"])
    assert code == 0
    assert "Qwen/Qwen3.5-4B" in out
    assert "Qwen/Qwen3-0.6B" in out


def test_models_filter_family():
    code, out, _ = run(["models", "--family", "qwen3.8"])
    assert code == 0
    assert "Qwen3.8" in out
    assert "Qwen3.5-4B" not in out


def test_validate_shipped_config(repo_root):
    code, out, err = run(["validate", str(repo_root / "configs" / "qwen3.5-4b-lora.yaml")])
    assert code == 0, err
    assert "OK" in out
    assert "24 records OK" in out or "records OK" in out


def test_validate_all_shipped_configs(configs_dir):
    for path in sorted(configs_dir.glob("*.yaml")):
        code, out, err = run(["validate", str(path)])
        assert code == 0, f"{path.name}: {err}"


def test_validate_rejects_bad_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  model_id: nonsense\n", encoding="utf-8")
    code, _, err = run(["validate", str(bad)])
    assert code == 1
    assert "CONFIG ERROR" in err


def test_validate_data_command(sample_dataset):
    code, out, _ = run(["validate-data", str(sample_dataset)])
    assert code == 0
    assert '"records"' in out


def test_validate_data_bad_file(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"messages": []}\n', encoding="utf-8")
    code, _, err = run(["validate-data", str(bad)])
    assert code == 1
    assert "DATASET ERROR" in err


def test_version_flag():
    with pytest.raises(SystemExit) as exc:
        run(["--version"])
    assert exc.value.code == 0
