"""Heavy smoke test: a REAL end-to-end LoRA training run.

Skipped automatically unless the full training stack is installed. Even when
installed it needs NO GPU and NO downloads: it builds a tiny *randomly
initialized* Qwen3-architecture model from a config (``from_config``), so it
exercises tokenizer-free trainer wiring, LoRA wrapping, save and reload.

Run with:  pytest -m heavy -q
"""

import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="heavy test: torch not installed")
transformers = pytest.importorskip("transformers", reason="heavy test: transformers not installed")
peft = pytest.importorskip("peft", reason="heavy test: peft not installed")
trl = pytest.importorskip("trl", reason="heavy test: trl not installed")
datasets = pytest.importorskip("datasets", reason="heavy test: datasets not installed")

from qwen_ft.config import FineTuneConfig  # noqa: E402


@pytest.mark.heavy
def test_tiny_random_lora_train_save_reload(tmp_path):
    torch.manual_seed(0)

    # Tiny Qwen3-architecture model — random weights, no download.
    tiny_cfg = transformers.Qwen3Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    model = transformers.Qwen3ForCausalLM(tiny_cfg)

    # Build a trivial offline WordLevel tokenizer — no network access required.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[PAD]": 0, "[UNK]": 1}
    for i in range(2, 1000):
        vocab[f"tok{i}"] = i
    tk = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]"
    )
    # TRL's SFTTrainer applies the tokenizer's chat template to conversational
    # `messages` datasets — real Qwen tokenizers always ship one, but this
    # offline toy tokenizer needs a minimal template. (Verified: trl 1.16.0
    # raises "Cannot use chat template functions..." without it.)
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )

    train_rows = [
        {"messages": [{"role": "user", "content": "tok1 tok2"},
                      {"role": "assistant", "content": "tok3 tok4"}]}
        for _ in range(8)
    ]
    ds = datasets.Dataset.from_list(train_rows)

    lora = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

    args = trl.SFTConfig(
        output_dir=str(tmp_path / "out"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        max_length=64,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=False,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = trl.SFTTrainer(model=model, args=args, train_dataset=ds, peft_config=lora, processing_class=tokenizer)
    trainer.train()

    adapter_dir = tmp_path / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # --- reload the adapter onto a fresh base model (save/reload workflow) ---
    base = transformers.Qwen3ForCausalLM(tiny_cfg)
    reloaded = peft.PeftModel.from_pretrained(base, str(adapter_dir))
    assert (adapter_dir / "adapter_model.safetensors").is_file() or (adapter_dir / "adapter_model.bin").is_file()
    assert reloaded.peft_config["default"].r == 4

    # --- forward pass produces finite logits ---
    ids = torch.tensor([[5, 6, 7, 8]])
    with torch.no_grad():
        out = reloaded(input_ids=ids, labels=ids)
    assert torch.isfinite(out.loss)

    shutil.rmtree(tmp_path / "out", ignore_errors=True)
