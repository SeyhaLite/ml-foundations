"""Registry of Qwen checkpoints verified for this course.

Every entry below was cross-checked against the official Hugging Face
organization ``Qwen`` (HF API query on 2026-10-22), the official
transformers model documentation (``model_doc/qwen3_5``), and the Qwen
GitHub repository (QwenLM/Qwen3.8).

Generation timeline (verified):

* **Qwen3** — April 2025. ``model_type: qwen3``, architecture
  ``Qwen3ForCausalLM``. First supported in **transformers 4.51.0**.
  Hybrid thinking mode toggled with the ``enable_thinking`` chat-template
  kwarg. July 2025 refresh: ``-Instruct-2507`` / ``-Thinking-2507``.
* **Qwen3.5** — released 2026-02-16 (397B-A17B first, then the Small
  series). ``model_type: qwen3_5``, architecture
  ``Qwen3_5ForConditionalGeneration`` (natively multimodal: text config +
  vision config). First supported in **transformers 5.2.0** — v4.x and
  v5.0/v5.1 CANNOT load these checkpoints. 3:1 hybrid attention stack
  (3 Gated DeltaNet linear-attention layers per 1 Gated Attention layer).
  262,144-token native context, 201 languages. Thinking is OFF by default
  on the 0.8B/2B/4B/9B "Small" models (``enable_thinking`` kwarg).
* **Qwen3.6** — mid 2026. Shares ``model_type: qwen3_5`` and the same
  transformers classes as Qwen3.5 (official transformers docs state this).
* **Qwen3.8** — August 2026. Also ``model_type: qwen3_5`` in its
  ``config.json`` (saved with transformers 5.8.0.dev0), so use
  **transformers >= 5.8** (5.20.x recommended). Adds ``reasoning_effort``
  and ``preserve_thinking`` chat-template controls.

QLoRA (4-bit bitsandbytes) is **not recommended** for Qwen3.5/3.6/3.8 per
the official Unsloth guidance ("higher than normal quantization
differences"); use bf16 LoRA instead. QLoRA remains a valid option for the
Qwen3 (2025) generation.

This module has NO network access and NO heavy imports — it is pure
metadata that tests and docs rely on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "ModelInfo",
    "MODEL_REGISTRY",
    "get_model_info",
    "is_supported_model_id",
    "list_models",
    "minimum_transformers_version",
    "resolve_transformers_requirement",
]

_HF_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}/[A-Za-z0-9._-]{1,100}$")


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for one verified Qwen checkpoint."""

    model_id: str
    family: str  # "qwen3" | "qwen3.5" | "qwen3.6" | "qwen3.8"
    kind: str  # "dense" | "moe"
    parameters: str  # human readable, e.g. "4B" or "35B-A3B"
    context_tokens: int
    min_transformers: str  # minimum transformers version that can load it
    architecture: str
    model_type: str
    multimodal: bool = False
    thinking_default: Optional[str] = None  # "on" | "off" | None (n/a)
    lora_bf16_vram_gb: Optional[float] = None  # approx. LoRA bf16 training VRAM
    qlora_recommended: bool = False
    notes: str = ""
    verified_on: str = "2026-10-22"
    aliases: List[str] = field(default_factory=list)


def _m(**kw) -> ModelInfo:
    return ModelInfo(**kw)


# ---------------------------------------------------------------------------
# Verified registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # ----------------------------- Qwen3 (2025) ----------------------------
    info.model_id: info
    for info in [
        _m(
            model_id="Qwen/Qwen3-0.6B",
            family="qwen3",
            kind="dense",
            parameters="0.6B",
            context_tokens=40960,
            min_transformers="4.51",
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            thinking_default="on",
            lora_bf16_vram_gb=2.0,
            qlora_recommended=True,
            notes=(
                "Smallest Qwen3; ideal first model for learning LoRA on a CPU/"
                "free-Colab tier. 32k native context (40,960 with the default "
                "YaRN setting in config.json)."
            ),
            aliases=["qwen3-0.6b", "qwen3-small"],
        ),
        _m(
            model_id="Qwen/Qwen3-1.7B",
            family="qwen3",
            kind="dense",
            parameters="1.7B",
            context_tokens=40960,
            min_transformers="4.51",
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            thinking_default="on",
            lora_bf16_vram_gb=5.0,
            qlora_recommended=True,
        ),
        _m(
            model_id="Qwen/Qwen3-4B",
            family="qwen3",
            kind="dense",
            parameters="4B",
            context_tokens=40960,
            min_transformers="4.51",
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            thinking_default="on",
            lora_bf16_vram_gb=9.0,
            qlora_recommended=True,
        ),
        _m(
            model_id="Qwen/Qwen3-8B",
            family="qwen3",
            kind="dense",
            parameters="8B",
            context_tokens=40960,
            min_transformers="4.51",
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            thinking_default="on",
            lora_bf16_vram_gb=18.0,
            qlora_recommended=True,
        ),
        _m(
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            family="qwen3",
            kind="dense",
            parameters="4B",
            context_tokens=262144,
            min_transformers="4.51",
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            thinking_default="off",
            lora_bf16_vram_gb=9.0,
            qlora_recommended=True,
            notes=(
                "July-2025 refresh, non-thinking Instruct variant with 256k "
                "context. Use the -Thinking-2507 sibling for reasoning."
            ),
        ),
        _m(
            model_id="Qwen/Qwen3-30B-A3B",
            family="qwen3",
            kind="moe",
            parameters="30B-A3B",
            context_tokens=40960,
            min_transformers="4.51",
            architecture="Qwen3MoeForCausalLM",
            model_type="qwen3_moe",
            thinking_default="on",
            qlora_recommended=False,
            notes="MoE with 3B active parameters.",
        ),
        # ---------------------------- Qwen3.5 (2026) -----------------------
        _m(
            model_id="Qwen/Qwen3.5-0.8B",
            family="qwen3.5",
            kind="dense",
            parameters="0.8B",
            context_tokens=262144,
            min_transformers="5.2",
            architecture="Qwen3_5ForConditionalGeneration",
            model_type="qwen3_5",
            multimodal=True,
            thinking_default="off",
            lora_bf16_vram_gb=3.0,
            qlora_recommended=False,
            notes="Smallest Qwen3.5; ~3GB VRAM for bf16 LoRA.",
            aliases=["qwen3.5-0.8b", "qwen35-0.8b"],
        ),
        _m(
            model_id="Qwen/Qwen3.5-2B",
            family="qwen3.5",
            kind="dense",
            parameters="2B",
            context_tokens=262144,
            min_transformers="5.2",
            architecture="Qwen3_5ForConditionalGeneration",
            model_type="qwen3_5",
            multimodal=True,
            thinking_default="off",
            lora_bf16_vram_gb=5.0,
            qlora_recommended=False,
        ),
        _m(
            model_id="Qwen/Qwen3.5-4B",
            family="qwen3.5",
            kind="dense",
            parameters="4B",
            context_tokens=262144,
            min_transformers="5.2",
            architecture="Qwen3_5ForConditionalGeneration",
            model_type="qwen3_5",
            multimodal=True,
            thinking_default="off",
            lora_bf16_vram_gb=10.0,
            qlora_recommended=False,
            notes=(
                "Recommended teaching model for this course: fits in ~10GB "
                "VRAM with bf16 LoRA and runs on a Colab L4/T4-x2 tier."
            ),
            aliases=["qwen3.5-4b"],
        ),
        _m(
            model_id="Qwen/Qwen3.5-9B",
            family="qwen3.5",
            kind="dense",
            parameters="9B",
            context_tokens=262144,
            min_transformers="5.2",
            architecture="Qwen3_5ForConditionalGeneration",
            model_type="qwen3_5",
            multimodal=True,
            thinking_default="off",
            lora_bf16_vram_gb=22.0,
            qlora_recommended=False,
        ),
        _m(
            model_id="Qwen/Qwen3.5-27B",
            family="qwen3.5",
            kind="dense",
            parameters="27B",
            context_tokens=262144,
            min_transformers="5.2",
            architecture="Qwen3_5ForConditionalGeneration",
            model_type="qwen3_5",
            multimodal=True,
            thinking_default="on",
            lora_bf16_vram_gb=56.0,
            qlora_recommended=False,
        ),
        _m(
            model_id="Qwen/Qwen3.5-35B-A3B",
            family="qwen3.5",
            kind="moe",
            parameters="35B-A3B",
            context_tokens=262144,
            min_transformers="5.2",
            architecture="Qwen3_5MoeForConditionalGeneration",
            model_type="qwen3_5_moe",
            multimodal=True,
            thinking_default="on",
            lora_bf16_vram_gb=74.0,
            qlora_recommended=False,
            notes="MoE; router layers are usually frozen for stability.",
        ),
        _m(
            model_id="Qwen/Qwen3.5-122B-A10B",
            family="qwen3.5",
            kind="moe",
            parameters="122B-A10B",
            context_tokens=262144,
            min_transformers="5.2",
            architecture="Qwen3_5MoeForConditionalGeneration",
            model_type="qwen3_5_moe",
            multimodal=True,
            thinking_default="on",
            lora_bf16_vram_gb=256.0,
            qlora_recommended=False,
            notes="bf16 LoRA needs ~256GB VRAM (multi-GPU).",
        ),
        _m(
            model_id="Qwen/Qwen3.5-397B-A17B",
            family="qwen3.5",
            kind="moe",
            parameters="397B-A17B",
            context_tokens=262144,
            min_transformers="5.2",
            architecture="Qwen3_5MoeForConditionalGeneration",
            model_type="qwen3_5_moe",
            multimodal=True,
            thinking_default="on",
            qlora_recommended=False,
            notes="Flagship release (2026-02-16). Reference only — not trainable on consumer hardware.",
        ),
        # ---------------------------- Qwen3.6 (2026) -----------------------
        _m(
            model_id="Qwen/Qwen3.6-27B",
            family="qwen3.6",
            kind="dense",
            parameters="27B",
            context_tokens=262144,
            min_transformers="5.2",
            architecture="Qwen3_5ForConditionalGeneration",
            model_type="qwen3_5",
            multimodal=True,
            thinking_default="on",
            qlora_recommended=False,
            notes=(
                "Stability/coding refresh on the Qwen3.5 architecture; loaded "
                "with the same qwen3_5 classes (official transformers docs)."
            ),
        ),
        _m(
            model_id="Qwen/Qwen3.6-35B-A3B",
            family="qwen3.6",
            kind="moe",
            parameters="35B-A3B",
            context_tokens=262144,
            min_transformers="5.2",
            architecture="Qwen3_5MoeForConditionalGeneration",
            model_type="qwen3_5_moe",
            multimodal=True,
            thinking_default="on",
            qlora_recommended=False,
        ),
        # ---------------------------- Qwen3.8 (2026) -----------------------
        _m(
            model_id="Qwen/Qwen3.8-27B",
            family="qwen3.8",
            kind="dense",
            parameters="27B",
            context_tokens=262144,
            min_transformers="5.8",
            architecture="Qwen3_5ForConditionalGeneration",
            model_type="qwen3_5",
            multimodal=True,
            thinking_default="on",
            qlora_recommended=False,
            notes=(
                "Qwen-Max-class open release (Aug 2026). Adds reasoning_effort "
                "and preserve_thinking chat-template controls. config.json was "
                "saved with transformers 5.8.0.dev0 — do not use <5.8."
            ),
        ),
        _m(
            model_id="Qwen/Qwen3.8-Flash-Next",
            family="qwen3.8",
            kind="dense",
            parameters="unknown",
            context_tokens=262144,
            min_transformers="5.8",
            architecture="Qwen3_5ForConditionalGeneration",
            model_type="qwen3_5",
            multimodal=True,
            qlora_recommended=False,
            notes="Fast-variant; parameter count not published on the card at verification time.",
        ),
        _m(
            model_id="Qwen/Qwen3.8-2.4T-A95B",
            family="qwen3.8",
            kind="moe",
            parameters="2.4T-A95B",
            context_tokens=262144,
            min_transformers="5.8",
            architecture="Qwen3_5MoeForConditionalGeneration",
            model_type="qwen3_5_moe",
            multimodal=True,
            qlora_recommended=False,
            notes="Flagship MoE. Reference only.",
        ),
    ]
}

# Friendly aliases -> canonical id
_ALIASES: Dict[str, str] = {}
for _info in MODEL_REGISTRY.values():
    for _a in _info.aliases:
        _ALIASES[_a.lower()] = _info.model_id


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Return :class:`ModelInfo` for a registry id or alias, else ``None``."""
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]
    canon = _ALIASES.get(model_id.lower())
    if canon:
        return MODEL_REGISTRY[canon]
    return None


def is_supported_model_id(model_id: str) -> bool:
    """True if the id is a well-formed ``org/name`` HF identifier.

    Membership in the registry is *not* required — users fine-tune their own
    checkpoints too — but registry membership enables extra validation
    (transformers floor, QLoRA warnings, VRAM hints).
    """
    return bool(_HF_ID_RE.match(model_id))


def list_models(family: Optional[str] = None) -> List[ModelInfo]:
    """List registry entries, optionally filtered by family."""
    items = list(MODEL_REGISTRY.values())
    if family:
        items = [m for m in items if m.family == family.lower()]
    return sorted(items, key=lambda m: (m.family, m.model_id))


def minimum_transformers_version(model_id: str) -> Optional[str]:
    """Minimum transformers version for a registered model (None if unknown)."""
    info = get_model_info(model_id)
    return info.min_transformers if info else None


def resolve_transformers_requirement(model_id: str) -> str:
    """Return a human-readable requirement string for any model id."""
    info = get_model_info(model_id)
    if info is None:
        if "Qwen3.5" in model_id or "Qwen3.6" in model_id:
            return "transformers>=5.2 (Qwen3.5/3.6 architecture 'qwen3_5')"
        if "Qwen3.8" in model_id:
            return "transformers>=5.8 (Qwen3.8 checkpoints saved with 5.8.0.dev0)"
        if "Qwen3" in model_id:
            return "transformers>=4.51 (Qwen3 generation)"
        return "unknown — check the model card's 'Framework' / config.json"
    return f"transformers>={info.min_transformers} ({info.model_type})"
