"""Tests for qwen_ft.models — registry integrity (the model compatibility audit, codified).

The registry was verified against the HF API on 2026-10-22. These tests keep
the metadata internally consistent and pin the facts that matter most:

* every model id is a valid ``org/name`` HF identifier;
* every model id exists in the OFFICIAL_VERIFIED_IDS list captured from the
  Hugging Face API (so a typo in the registry fails the suite);
* minimum transformers floors match the documented history
  (Qwen3 -> 4.51, Qwen3.5/3.6 -> 5.2, Qwen3.8 -> 5.8);
* QLoRA is never "recommended" on the qwen3_5 architecture;
* no accidental cross-generation mixing (e.g. Qwen3.5 id claiming model_type
  "qwen3").
"""

import re

import pytest

from qwen_ft.models import (
    MODEL_REGISTRY,
    get_model_info,
    is_supported_model_id,
    list_models,
    minimum_transformers_version,
    resolve_transformers_requirement,
)

# Ids captured from https://huggingface.co/api/models?author=Qwen on 2026-10-22.
OFFICIAL_VERIFIED_IDS = {
    # Qwen3 (2025)
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-30B-A3B",
    # Qwen3.5 (2026)
    "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
    # Qwen3.6 (2026)
    "Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-35B-A3B",
    # Qwen3.8 (2026)
    "Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-Flash-Next", "Qwen/Qwen3.8-2.4T-A95B",
}

VERSION_RE = re.compile(r"^\d+(\.\d+)*$")


def test_registry_not_empty():
    assert len(MODEL_REGISTRY) >= 15


def test_all_ids_well_formed():
    for mid in MODEL_REGISTRY:
        assert is_supported_model_id(mid), mid


def test_all_registered_ids_are_official():
    unknown = set(MODEL_REGISTRY) - OFFICIAL_VERIFIED_IDS
    assert not unknown, f"registry contains ids not verified on HF: {unknown}"


def test_min_transformers_parseable():
    for info in MODEL_REGISTRY.values():
        assert VERSION_RE.match(info.min_transformers), info.model_id


def test_qwen3_generation_floor():
    for info in list_models(family="qwen3"):
        assert info.min_transformers == "4.51"
        assert info.model_type == "qwen3" or info.model_type == "qwen3_moe"
        assert info.architecture.startswith("Qwen3ForCausalLM") or info.architecture.startswith("Qwen3MoeForCausalLM")


def test_qwen35_family_requires_transformers_v5():
    for fam in ("qwen3.5", "qwen3.6", "qwen3.8"):
        for info in list_models(family=fam):
            major = int(info.min_transformers.split(".")[0])
            assert major >= 5, f"{info.model_id} claims transformers {info.min_transformers}"
            assert info.model_type in {"qwen3_5", "qwen3_5_moe"}


def test_qwen35_is_multimodal_architecture():
    for info in list_models(family="qwen3.5"):
        assert info.multimodal is True
        assert "ConditionalGeneration" in info.architecture


def test_qwen38_needs_5_8():
    for info in list_models(family="qwen3.8"):
        assert tuple(int(x) for x in info.min_transformers.split(".")) >= (5, 8)


def test_qlora_never_recommended_on_qwen3_5_arch():
    for info in MODEL_REGISTRY.values():
        if info.model_type.startswith("qwen3_5"):
            assert info.qlora_recommended is False, info.model_id


def test_context_lengths():
    for info in MODEL_REGISTRY.values():
        assert info.context_tokens >= 32768
        if info.family in {"qwen3.5", "qwen3.6", "qwen3.8"}:
            assert info.context_tokens == 262144


def test_small_qwen35_thinking_default_off():
    for mid in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"):
        info = get_model_info(mid)
        assert info is not None and info.thinking_default == "off", mid


def test_aliases_resolve():
    assert get_model_info("qwen3.5-4b").model_id == "Qwen/Qwen3.5-4B"
    assert get_model_info("qwen3-small").model_id == "Qwen/Qwen3-0.6B"
    assert get_model_info("Qwen/Qwen2.5-7B") is None  # out of scope generation


def test_minimum_transformers_version():
    assert minimum_transformers_version("Qwen/Qwen3.5-4B") == "5.2"
    assert minimum_transformers_version("Qwen/Qwen3.8-27B") == "5.8"
    assert minimum_transformers_version("Qwen/Qwen3-0.6B") == "4.51"
    assert minimum_transformers_version("someone/unknown") is None


def test_resolve_requirement_strings():
    assert "5.2" in resolve_transformers_requirement("Qwen/Qwen3.5-9B")
    assert "5.8" in resolve_transformers_requirement("Qwen/Qwen3.8-27B")
    assert "4.51" in resolve_transformers_requirement("Qwen/Qwen3-8B")
    # unregistered ids still get family-level guidance
    assert "5.2" in resolve_transformers_requirement("Qwen/Qwen3.5-70B-Imaginary")
    assert "check the model card" in resolve_transformers_requirement("meta-llama/Llama-4")


def test_vram_hint_ordering():
    small = get_model_info("Qwen/Qwen3.5-0.8B").lora_bf16_vram_gb
    big = get_model_info("Qwen/Qwen3.5-9B").lora_bf16_vram_gb
    assert small < big


@pytest.mark.parametrize("bad_id", ["", "no-slash", "too/many/slashes", "a/" + "x" * 300])
def test_invalid_ids_rejected(bad_id):
    assert not is_supported_model_id(bad_id)
