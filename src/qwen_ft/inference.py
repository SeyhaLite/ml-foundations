"""Inference helpers: chat with a base model or a trained LoRA adapter,
merge adapters, and export.  Lazy heavy imports — see :mod:`qwen_ft.training`.

Chat-template controls (verified 2026-10-22):

* Qwen3 (2025): ``enable_thinking=True/False`` kwarg of
  ``apply_chat_template``. The original 0.6B–32B checkpoints default to
  thinking ON; ``-Instruct-2507`` variants are non-thinking,
  ``-Thinking-2507`` variants are thinking-only.
* Qwen3.5 (2026): hybrid reasoning, same ``enable_thinking`` kwarg.
  The Small models (0.8B/2B/4B/9B) default to thinking OFF; 27B+ default ON.
* Qwen3.8 (2026): adds ``reasoning_effort`` ("low"/"medium"/"high") and
  ``preserve_thinking`` chat-template controls on top of the Qwen3.5 stack.

Official sampling recommendations for Qwen3.5 (Unsloth/Qwen docs):

=========================  ===========  ==============  ======  =========
Mode                       temperature  top_p           top_k   presence_penalty
=========================  ===========  ==============  ======  =========
Instruct — general         0.7          0.8             20      1.5
Instruct — reasoning       1.0          0.95            20      1.5
Thinking — general         1.0          0.95            20      1.5
Thinking — precise coding  0.6          0.95            20      0.0
=========================  ===========  ==============  ======  =========
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from qwen_ft.training import HeavyDependencyError, _require  # reuse guard

__all__ = [
    "HeavyDependencyError",
    "load_for_chat",
    "chat",
    "merge_adapter",
    "SAMPLING_PRESETS",
]

SAMPLING_PRESETS: Dict[str, Dict[str, Any]] = {
    "instruct_general": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
    "instruct_reasoning": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
    "thinking_general": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
    "thinking_coding": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
    "greedy": {"temperature": 0.0, "do_sample": False},
}


def load_for_chat(
    model_id_or_path: str,
    adapter_path: Optional[str] = None,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
):
    """Load base model (+ optional LoRA adapter) and tokenizer for chat.

    Returns ``(model, tokenizer)``.  Requires the ``train`` extra.
    """
    transformers = _require("transformers")
    torch = _require("torch")

    tdt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    kwargs: Dict[str, Any] = {"torch_dtype": tdt}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if torch.cuda.is_available():
        kwargs["device_map"] = device_map

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path)

    if adapter_path:
        peft = _require("peft")
        model = peft.PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def chat(
    model,
    tokenizer,
    messages: Sequence[Dict[str, str]],
    max_new_tokens: int = 1024,
    enable_thinking: Optional[bool] = None,
    preset: Optional[str] = None,
    **generation_kwargs: Any,
) -> str:
    """One chat turn. ``enable_thinking=None`` keeps the model default."""
    torch = _require("torch")

    template_kwargs: Dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    prompt = tokenizer.apply_chat_template(list(messages), **template_kwargs)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    sampling = dict(SAMPLING_PRESETS.get(preset or "", {}))
    sampling.update({k: v for k, v in generation_kwargs.items() if v is not None})
    sampling.setdefault("max_new_tokens", max_new_tokens)
    do_sample = sampling.pop("do_sample", None)
    if do_sample is False or sampling.get("temperature", 1.0) == 0.0:
        sampling.pop("temperature", None)
        sampling.pop("top_p", None)
        sampling.pop("top_k", None)
        sampling["do_sample"] = False
    else:
        sampling["do_sample"] = True

    with torch.no_grad():
        out = model.generate(**inputs, **sampling)
    reply_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(reply_ids, skip_special_tokens=True)


def merge_adapter(
    model_id_or_path: str,
    adapter_path: str,
    out_dir: Union[str, Path],
    dtype: str = "bfloat16",
) -> Path:
    """Merge a LoRA adapter into the base weights and save a standalone model."""
    transformers = _require("transformers")
    peft = _require("peft")
    torch = _require("torch")

    tdt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    base = transformers.AutoModelForCausalLM.from_pretrained(model_id_or_path, torch_dtype=tdt)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path)
    merged = peft.PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
