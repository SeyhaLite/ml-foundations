"""Tests for qwen_ft.config — validation, YAML round-trip, warnings."""

from pathlib import Path

import pytest

from qwen_ft.config import (
    ConfigError,
    FineTuneConfig,
    InferenceConfig,
    LoRAConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
    save_config,
)


def test_defaults_are_valid():
    cfg = FineTuneConfig()
    cfg.validate()


def test_shipped_configs_are_valid(configs_dir: Path):
    yamls = sorted(configs_dir.glob("*.yaml"))
    assert len(yamls) >= 3, "expected at least 3 shipped configs"
    for path in yamls:
        cfg = load_config(path)
        assert cfg.model.model_id.startswith("Qwen/")


def test_unknown_top_level_section_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("modle:\n  model_id: Qwen/Qwen3.5-4B\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown top-level section"):
        load_config(bad)


def test_typo_in_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "lora:\n  rr: 16\n",  # typo: rr instead of r
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(bad)


def test_invalid_learning_rate():
    t = TrainingConfig(learning_rate=5.0)
    with pytest.raises(ConfigError, match="learning_rate"):
        t.validate()


def test_invalid_model_id():
    m = ModelConfig(model_id="not-a-hf-id")
    with pytest.raises(ConfigError, match="valid 'org/name'"):
        m.validate()


def test_invalid_lora_rank():
    l = LoRAConfig(r=0)
    with pytest.raises(ConfigError, match="lora.r"):
        l.validate()
    l2 = LoRAConfig(dropout=1.5)
    with pytest.raises(ConfigError, match="dropout"):
        l2.validate()


def test_invalid_target_modules_string():
    l = LoRAConfig(target_modules="everything")
    with pytest.raises(ConfigError, match="all-linear"):
        l.validate()


def test_target_modules_list_ok():
    l = LoRAConfig(target_modules=["q_proj", "v_proj"])
    l.validate()


def test_max_length_range():
    m = ModelConfig(max_length=10)
    with pytest.raises(ConfigError, match="max_length"):
        m.validate()


def test_yaml_roundtrip(tmp_path):
    cfg = FineTuneConfig()
    cfg.model.model_id = "Qwen/Qwen3-0.6B"
    cfg.lora.r = 32
    cfg.training.learning_rate = 5e-5
    cfg.inference.temperature = 0.9
    out = tmp_path / "cfg.yaml"
    save_config(cfg, out)
    reloaded = load_config(out)
    assert reloaded.to_dict() == cfg.to_dict()


def test_warning_for_unregistered_model():
    cfg = FineTuneConfig()
    cfg.model.model_id = "SeyhaLite/my-custom-qwen"
    cfg.validate()
    warns = cfg.warnings()
    assert any("not in the verified registry" in w for w in warns)


def test_warning_alpha_not_2r():
    l = LoRAConfig(r=16, alpha=8)
    l.validate()
    assert any("2x r" in w for w in l.warnings())


def test_qlora_warning_for_qwen35_float16():
    cfg = FineTuneConfig()
    cfg.model.model_id = "Qwen/Qwen3.5-4B"
    cfg.model.dtype = "float16"
    cfg.validate()
    warns = cfg.warnings()
    assert any("QLoRA is NOT recommended" in w or "bf16" in w for w in warns)


def test_max_length_exceeds_context_warning():
    m = ModelConfig(model_id="Qwen/Qwen3-0.6B", max_length=131072)
    m.validate()
    assert any("exceeds the native context" in w for w in m.warnings())


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(p)


def test_inference_ranges():
    i = InferenceConfig(top_p=1.5)
    with pytest.raises(ConfigError, match="top_p"):
        i.validate()
    i2 = InferenceConfig(max_new_tokens=0)
    with pytest.raises(ConfigError, match="max_new_tokens"):
        i2.validate()


def test_effective_batch_size():
    t = TrainingConfig(batch_size=2, grad_accum_steps=8)
    assert t.effective_batch_size() == 16
