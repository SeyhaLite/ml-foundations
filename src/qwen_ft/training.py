"""LoRA SFT training pipeline for Qwen models (lazy heavy imports).

This module is the "real" training code referenced by chapter 6.  It imports
torch/transformers/peft/trl ONLY inside :func:`run_training` so that the rest
of the package (and the test-suite) works on machines without those libraries.

Verified API targets (2026-10-22):

* transformers 5.20.x — ``AutoModelForCausalLM`` / ``AutoProcessor`` resolve
  Qwen3 (``Qwen3ForCausalLM``) and Qwen3.5/3.6/3.8 (``qwen3_5`` classes).
* peft 0.21.x — ``LoraConfig(target_modules="all-linear", ...)``.
* trl 1.16.x — ``SFTTrainer(model=..., args=SFTConfig(...), train_dataset=...,
  peft_config=...)`` with conversational ``messages`` datasets.

Typical use::

    from qwen_ft.config import load_config
    from qwen_ft.training import run_training

    cfg = load_config("configs/qwen3.5-4b-lora.yaml")
    run_training(cfg)          # requires the 'train' extra + a GPU
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from qwen_ft.config import FineTuneConfig, load_config
from qwen_ft.data import read_jsonl, split_dataset, validate_jsonl_file
from qwen_ft.models import get_model_info

if TYPE_CHECKING:  # pragma: no cover
    import torch  # noqa: F401

__all__ = [
    "HeavyDependencyError",
    "check_environment",
    "build_trainer",
    "run_training",
    "train_from_yaml",
]


class HeavyDependencyError(RuntimeError):
    """Raised when torch/transformers/peft/trl are missing."""


def check_environment(min_transformers: str = "5.2") -> Dict[str, Any]:
    """Report versions of the heavy stack without importing qwen_ft.training deps at module scope.

    Returns a dict like ``{"torch": "2.14.1", "transformers": "5.20.2", ...}``
    with ``None`` for anything not installed.  Never raises.
    """
    versions: Dict[str, Any] = {}
    for pkg in ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"):
        try:
            import importlib.metadata as md

            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = None
    versions["cuda_available"] = None
    try:
        import torch

        versions["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return versions


def _require(pkg: str):
    try:
        import importlib

        return importlib.import_module(pkg)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise HeavyDependencyError(
            f"Package {pkg!r} is required for training. Install with:\n"
            f"    pip install -e '.[train]'\n"
            f"(original error: {exc})"
        ) from exc


def _torch_dtype(name: str):
    torch = _require("torch")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }[name]


def build_trainer(cfg: FineTuneConfig, resume_from: Optional[str] = None):
    """Construct the tokenizer/model/dataset/SFTTrainer objects (needs the train extra)."""
    transformers = _require("transformers")
    peft = _require("peft")
    trl = _require("trl")
    torch = _require("torch")

    cfg.validate()
    report = validate_jsonl_file(cfg.training.dataset_path)
    records = read_jsonl(cfg.training.dataset_path)
    train_recs, eval_recs = split_dataset(
        records, eval_ratio=cfg.training.eval_split_ratio, seed=cfg.training.seed
    )

    info = get_model_info(cfg.model.model_id)
    if info is not None:
        # Fail fast with a clear message instead of a cryptic KeyError deep in
        # transformers when the installed version is too old for the checkpoint.
        import packaging.version as _v

        have = _v.parse(transformers.__version__.split("+")[0])
        need = _v.parse(info.min_transformers)
        if have < need:
            raise HeavyDependencyError(
                f"{cfg.model.model_id} requires transformers>={info.min_transformers}; "
                f"installed version is {transformers.__version__}. "
                "Upgrade with: pip install -U 'transformers>=5.8,<6'"
            )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.model_id, trust_remote_code=cfg.model.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": _torch_dtype(cfg.model.dtype),
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if cfg.model.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.model.attn_implementation
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = transformers.AutoModelForCausalLM.from_pretrained(cfg.model.model_id, **model_kwargs)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora_config = peft.LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=float(cfg.lora.dropout),
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )

    datasets = _require("datasets")
    train_ds = datasets.Dataset.from_list(train_recs)
    eval_ds = datasets.Dataset.from_list(eval_recs) if eval_recs else None

    bf16_flag = cfg.training.bf16.lower()
    use_bf16 = True if bf16_flag == "true" else (
        False if bf16_flag == "false" else torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    sft_config = trl.SFTConfig(
        output_dir=cfg.training.output_dir,
        num_train_epochs=float(cfg.training.num_epochs),
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accum_steps,
        learning_rate=float(cfg.training.learning_rate),
        warmup_ratio=float(cfg.training.warmup_ratio),
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        seed=cfg.training.seed,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        packing=cfg.training.packing,
        completion_only_loss=cfg.training.completion_only_loss,
        max_length=cfg.model.max_length,
        bf16=use_bf16,
        fp16=False,  # bf16-first policy: fp16 training of Qwen checkpoints is discouraged
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.training.save_steps if eval_ds is not None else None,
        save_total_limit=3,
        report_to=cfg.training.report_to,
    )

    trainer = trl.SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer._qwen_ft_dataset_report = report  # surface in logs if desired
    return trainer


def run_training(cfg: FineTuneConfig, resume_from: Optional[str] = None) -> Path:
    """Train and save the LoRA adapter. Returns the adapter directory."""
    trainer = build_trainer(cfg, resume_from=resume_from)
    trainer.train(resume_from_checkpoint=resume_from)
    adapter_dir = Path(cfg.training.output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(str(adapter_dir))
    return adapter_dir


def train_from_yaml(path: str | Path, resume_from: Optional[str] = None) -> Path:
    """Convenience entry point: ``train_from_yaml("configs/qwen3.5-4b-lora.yaml")``."""
    cfg = load_config(path)
    for w in cfg.warnings():
        print(f"[qwen-ft][warn] {w}")
    return run_training(cfg, resume_from=resume_from)
