"""Typed, validated configuration for LoRA fine-tuning runs.

Everything here is pure Python + PyYAML so it can be exercised in tests,
CI and the documentation build without torch/transformers installed.

Example YAML (see ``configs/*.yaml``)::

    model:
      model_id: Qwen/Qwen3.5-4B
      max_length: 2048
      dtype: bfloat16
    lora:
      r: 16
      alpha: 32
      dropout: 0.05
      target_modules: all-linear
    training:
      output_dir: outputs/qwen3.5-4b-khmer
      num_epochs: 1
      batch_size: 2
      grad_accum_steps: 8
      learning_rate: 1.0e-4
      warmup_ratio: 0.03
      dataset_path: data/sample_khmer_assistant.jsonl
      eval_split_ratio: 0.1
      seed: 42
    inference:
      temperature: 0.7
      top_p: 0.8
      top_k: 20
      max_new_tokens: 1024
      enable_thinking: false
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from qwen_ft.models import get_model_info, is_supported_model_id

__all__ = [
    "ConfigError",
    "ModelConfig",
    "LoRAConfig",
    "TrainingConfig",
    "InferenceConfig",
    "FineTuneConfig",
    "load_config",
    "save_config",
]

_VALID_DTYPES = {"bfloat16", "float16", "float32", "auto"}
_TARGET_MODULE_HINTS = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


class ConfigError(ValueError):
    """Raised when a configuration file or object is invalid."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _reject_unknown_keys(cls, data: Dict[str, Any]) -> None:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{cls.__name__}: unknown key(s) {unknown}. "
            f"Valid keys: {sorted(known)}"
        )


@dataclass
class ModelConfig:
    """Which checkpoint to fine-tune and how to load it."""

    model_id: str = "Qwen/Qwen3.5-4B"
    max_length: int = 2048
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None  # "sdpa" | "flash_attention_2" | None

    def validate(self) -> "ModelConfig":
        _check(isinstance(self.model_id, str) and self.model_id,
               "model.model_id must be a non-empty string")
        _check(is_supported_model_id(self.model_id),
               f"model.model_id {self.model_id!r} is not a valid 'org/name' HF id")
        _check(isinstance(self.max_length, int) and 128 <= self.max_length <= 262144,
               f"model.max_length must be an int in [128, 262144], got {self.max_length!r}")
        _check(self.dtype in _VALID_DTYPES,
               f"model.dtype must be one of {sorted(_VALID_DTYPES)}, got {self.dtype!r}")
        if self.attn_implementation is not None:
            _check(self.attn_implementation in {"sdpa", "flash_attention_2", "eager"},
                   "model.attn_implementation must be 'sdpa', 'flash_attention_2', 'eager' or null")
        return self

    def warnings(self) -> List[str]:
        out: List[str] = []
        info = get_model_info(self.model_id)
        if info:
            if self.max_length > info.context_tokens:
                out.append(
                    f"max_length {self.max_length} exceeds the native context "
                    f"({info.context_tokens}) of {self.model_id}"
                )
            if info.family in {"qwen3.5", "qwen3.6", "qwen3.8"} and self.dtype == "float16":
                out.append(
                    f"{self.model_id} is trained in bf16; prefer dtype: bfloat16 "
                    "on hardware that supports it (Ampere+/Ada/Hopper)"
                )
        else:
            out.append(
                f"{self.model_id} is not in the verified registry — "
                "check its model card for the required transformers version"
            )
        return out


@dataclass
class LoRAConfig:
    """PEFT LoRA hyper-parameters."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Union[str, List[str]] = "all-linear"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def validate(self) -> "LoRAConfig":
        _check(isinstance(self.r, int) and 1 <= self.r <= 512,
               f"lora.r must be an int in [1, 512], got {self.r!r}")
        _check(isinstance(self.alpha, int) and 1 <= self.alpha <= 1024,
               f"lora.alpha must be an int in [1, 1024], got {self.alpha!r}")
        _check(isinstance(self.dropout, (int, float)) and 0.0 <= float(self.dropout) < 1.0,
               f"lora.dropout must be in [0.0, 1.0), got {self.dropout!r}")
        if isinstance(self.target_modules, str):
            _check(self.target_modules == "all-linear",
                   "lora.target_modules as a string must be 'all-linear' (or an explicit list)")
        else:
            _check(isinstance(self.target_modules, list) and len(self.target_modules) > 0,
                   "lora.target_modules must be 'all-linear' or a non-empty list")
            _check(all(isinstance(t, str) and t for t in self.target_modules),
                   "lora.target_modules entries must be non-empty strings")
            unknown = sorted(set(self.target_modules) - _TARGET_MODULE_HINTS)
            if unknown:
                # Not fatal: hybrid Qwen3.5 layers use extra names (in_proj_*,
                # out_proj, conv1d, ...). Just a hint.
                pass
        _check(self.bias in {"none", "all", "lora_only"},
               f"lora.bias must be one of none/all/lora_only, got {self.bias!r}")
        _check(self.task_type in {"CAUSAL_LM", "SEQ_2_SEQ_LM"},
               f"lora.task_type must be CAUSAL_LM or SEQ_2_SEQ_LM, got {self.task_type!r}")
        return self

    def warnings(self) -> List[str]:
        out: List[str] = []
        if isinstance(self.alpha, int) and isinstance(self.r, int) and self.alpha != 2 * self.r:
            out.append(
                f"lora.alpha ({self.alpha}) is not 2x r ({self.r}); the common "
                "starting point is alpha = 2r — deviate only deliberately"
            )
        return out


@dataclass
class TrainingConfig:
    """SFT training loop hyper-parameters (mapped onto trl.SFTConfig)."""

    output_dir: str = "outputs/run"
    dataset_path: str = "data/sample_khmer_assistant.jsonl"
    num_epochs: float = 1.0
    batch_size: int = 2
    grad_accum_steps: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    eval_split_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42
    gradient_checkpointing: bool = True
    packing: bool = False
    completion_only_loss: bool = True
    bf16: str = "auto"  # "auto" | "true" | "false"
    report_to: str = "none"

    def validate(self) -> "TrainingConfig":
        _check(isinstance(self.output_dir, str) and self.output_dir,
               "training.output_dir must be a non-empty string")
        _check(isinstance(self.dataset_path, str) and self.dataset_path,
               "training.dataset_path must be a non-empty string")
        _check(isinstance(self.num_epochs, (int, float)) and 0 < float(self.num_epochs) <= 100,
               f"training.num_epochs must be in (0, 100], got {self.num_epochs!r}")
        _check(isinstance(self.batch_size, int) and 1 <= self.batch_size <= 4096,
               f"training.batch_size must be an int in [1, 4096], got {self.batch_size!r}")
        _check(isinstance(self.grad_accum_steps, int) and 1 <= self.grad_accum_steps <= 4096,
               f"training.grad_accum_steps must be an int in [1, 4096], got {self.grad_accum_steps!r}")
        _check(isinstance(self.learning_rate, float) and 1e-7 <= self.learning_rate <= 1e-2,
               f"training.learning_rate must be a float in [1e-7, 1e-2], got {self.learning_rate!r}")
        _check(isinstance(self.warmup_ratio, (int, float)) and 0.0 <= float(self.warmup_ratio) <= 0.5,
               f"training.warmup_ratio must be in [0.0, 0.5], got {self.warmup_ratio!r}")
        _check(isinstance(self.eval_split_ratio, (int, float)) and 0.0 <= float(self.eval_split_ratio) < 0.5,
               f"training.eval_split_ratio must be in [0.0, 0.5), got {self.eval_split_ratio!r}")
        _check(isinstance(self.logging_steps, int) and self.logging_steps >= 1,
               "training.logging_steps must be >= 1")
        _check(isinstance(self.save_steps, int) and self.save_steps >= 1,
               "training.save_steps must be >= 1")
        _check(isinstance(self.seed, int), "training.seed must be an int")
        _check(str(self.bf16).lower() in {"auto", "true", "false"},
               f"training.bf16 must be 'auto', 'true' or 'false', got {self.bf16!r}")
        return self

    def warnings(self) -> List[str]:
        out: List[str] = []
        if self.learning_rate > 3e-4:
            out.append(
                f"learning_rate {self.learning_rate} is high for LoRA SFT; "
                "1e-4..2e-4 is the usual range"
            )
        return out

    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps


@dataclass
class InferenceConfig:
    """Sampling defaults (aligned with the official Qwen3.5 recommendations)."""

    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens: int = 1024
    enable_thinking: bool = False
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0

    def validate(self) -> "InferenceConfig":
        _check(isinstance(self.temperature, (int, float)) and 0.0 <= float(self.temperature) <= 2.0,
               f"inference.temperature must be in [0, 2], got {self.temperature!r}")
        _check(isinstance(self.top_p, (int, float)) and 0.0 < float(self.top_p) <= 1.0,
               f"inference.top_p must be in (0, 1], got {self.top_p!r}")
        _check(isinstance(self.top_k, int) and self.top_k >= 0,
               f"inference.top_k must be an int >= 0, got {self.top_k!r}")
        _check(isinstance(self.max_new_tokens, int) and 1 <= self.max_new_tokens <= 65536,
               f"inference.max_new_tokens must be in [1, 65536], got {self.max_new_tokens!r}")
        _check(isinstance(self.repetition_penalty, (int, float)) and 0.5 <= float(self.repetition_penalty) <= 2.0,
               f"inference.repetition_penalty must be in [0.5, 2], got {self.repetition_penalty!r}")
        return self

    def warnings(self) -> List[str]:
        return []


@dataclass
class FineTuneConfig:
    """Top-level config aggregating all sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def validate(self, strict: bool = True) -> "FineTuneConfig":
        """Validate every section; raise :class:`ConfigError` on the first problem."""
        self.model.validate()
        self.lora.validate()
        self.training.validate()
        self.inference.validate()
        if strict:
            # Cross-section rule: QLoRA-style float16 + 4-bit is discouraged on
            # the qwen3_5 architecture; we cannot see BitsAndBytesConfig here,
            # but we can warn about dtype/model mismatch.
            pass
        return self

    def warnings(self) -> List[str]:
        out: List[str] = []
        out += [f"[model] {w}" for w in self.model.warnings()]
        out += [f"[lora] {w}" for w in self.lora.warnings()]
        out += [f"[training] {w}" for w in self.training.warnings()]
        out += [f"[inference] {w}" for w in self.inference.warnings()]
        info = get_model_info(self.model.model_id)
        if info and not info.qlora_recommended and self.model.dtype == "float16":
            out.append(
                f"[model] {self.model.model_id}: 4-bit QLoRA is NOT recommended on "
                "Qwen3.5/3.6/3.8 (quantization sensitivity); use bf16 LoRA"
            )
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTION_CLASSES = {
    "model": ModelConfig,
    "lora": LoRAConfig,
    "training": TrainingConfig,
    "inference": InferenceConfig,
}


def _build_section(cls, data: Optional[Dict[str, Any]]):
    if data is None:
        return cls()
    _check(isinstance(data, dict), f"{cls.__name__} section must be a mapping")
    _reject_unknown_keys(cls, data)
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    return cls(**kwargs)


def load_config(path: Union[str, Path]) -> FineTuneConfig:
    """Load + validate a YAML config file. Raises :class:`ConfigError`."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML — {exc}") from exc
    if raw is None:
        raise ConfigError(f"{path}: file is empty")
    _check(isinstance(raw, dict), f"{path}: top level must be a mapping")
    unknown = sorted(set(raw) - set(_SECTION_CLASSES))
    if unknown:
        raise ConfigError(
            f"{path}: unknown top-level section(s) {unknown}. "
            f"Valid sections: {sorted(_SECTION_CLASSES)}"
        )
    cfg = FineTuneConfig(**{
        name: _build_section(cls, raw.get(name))
        for name, cls in _SECTION_CLASSES.items()
    })
    cfg.validate()
    return cfg


def save_config(cfg: FineTuneConfig, path: Union[str, Path]) -> Path:
    """Save a config to YAML (round-trips through :func:`load_config`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)
    return path
