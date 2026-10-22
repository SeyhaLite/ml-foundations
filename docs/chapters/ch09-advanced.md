# ៩ · ប្រធានបទកម្រិតខ្ពស់ — QLoRA, MoE, Multimodal, DPO និង Qwen3.6/3.8

```{contents} ក្នុងជំពូកនេះ
:local: true
:depth: 2
```

## គោលបំណងនៃជំពូក

ជំពូកនេះសម្រាប់អ្នកដែលឆ្លងកាត់ជំពូក ១–៨ រួចហើយ។ យើងនឹងគ្របដណ្តប់៖

- QLoRA 4-bit សម្រាប់ជំនាន់ Qwen3 (២០២៥)
- ការហ្វឹក MoE (35B-A3B, 122B-A10B)
- Multimodal fine-tuning (Qwen3.5 ជា VL model ពីកំណើត)
- វិធីសាស្ត្រ post-training ក្រោយ SFT៖ **DPO** និង **GRPO**
- ការរក្សាសមត្ថភាព reasoning
- Unsloth និងលក្ខណៈថ្មីរបស់ Qwen3.6/3.8

## QLoRA (4-bit) — សម្រាប់តែជំនាន់ Qwen3 (២០២៥)

**QLoRA** = បង្រួមទម្ងន់ base ទៅ 4-bit (NF4) + LoRA adapters លើគេ។
VRAM ថយបន្ថែម ~៤០% ធៀប LoRA bf16។

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", quantization_config=bnb_config, device_map="auto"
)
```

```{warning} កំហុសដែលអ្នកចាប់ផ្តើមធ្វើញឹកញាប់
QLoRA **មិនណែនាំ** សម្រាប់ Qwen3.5/3.6/3.8។ ការណែនាំផ្លូវការ (Unsloth) បញ្ជាក់ថា
Qwen3.5 មានភាពងាយរងគ្រោះពី quantization ខ្ពស់ជាងធម្មតា — ជាពិសេស MoE
(bitsandbytes មានដែនកំណត់ជាមួយ MoE)។ សម្រាប់ Qwen3.5 → ប្រើ **bf16 LoRA**។
```

## MoE fine-tuning (Qwen3.5-35B-A3B / 122B-A10B)

Mixture-of-Experts មាន router ជ្រើស expert មួយចំនួនក្នុងមួយ token។ ចំណុចសំខាន់៖

1. **VRAM** — 35B-A3B bf16 LoRA ≈ 74 GB (A100 80GB) · 122B-A10B ≈ 256 GB (multi-GPU,
   `device_map="balanced"` ឬ deepspeed/FSDP)
2. **Router freezing** — tooling (ដូចជា Unsloth) បង្កក router ជា default
   ដើម្បីស្ថិរភាព។ ការហ្វឹក router អាចបំផ្លាញ load-balancing របស់ experts។
3. **`target_modules="all-linear"`** នៅតែត្រឹមត្រូវ — PEFT ដាក់ LoRA លើ expert
   layers ផង (ចំនួន adapter ច្រើនជាង dense តែមួយ expert តូច)
4. MoE លឿនក្នុង inference (active params តិច) តែ **ហ្វឹក** នៅតែត្រូវការ VRAM
   សម្រាប់ weights ទាំងអស់ក្នុង memory

## Multimodal — Qwen3.5 ជា VL model

Checkpoint Qwen3.5 ទាំងអស់មាន vision encoder។ បើចង់ហ្វឹកជាមួយរូបភាព៖

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-4B", torch_dtype=torch.bfloat16, device_map="auto"
)

# Dataset ទម្រង់ conversational + រូបភាព
example = {
    "messages": [
        {"role": "user", "content": [
            {"type": "image", "url": "https://example.org/angkor.jpg"},
            {"type": "text", "text": "រូបភាពនេះនៅឯណា?"},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "នេះជាប្រាសាទអង្គរវត្ត នៅសៀមរាប។"},
        ]},
    ]
}
```

ចំណាំបច្ចេកទេស៖

- LoRA លើ **text backbone** គឺគ្រប់គ្រាន់សម្រាប់កិច្ចការភាគច្រើន —
  vision tower អាចបង្កក
- ពិនិត្យ `image_token_id` / `video_token_id` ក្នុង config — កុំ tokenize រូបភាពខុស
- VRAM កើនតាមចំនួន/ទំហំរូបភាព (vision tokens ច្រើន)

## ក្រោយ SFT — DPO និង GRPO

| វិធីសាស្ត្រ | ទិន្នន័យត្រូវការ | ប្រើពេលណា |
|---|---|---|
| **SFT** (ជំពូក ៦) | ឧទាហរណ៍ (សំណួរ, ចម្លើយល្អ) | បង្រៀនទម្រង់/ភាសា/ជំនាញ |
| **DPO** | គូ (ចម្លើយល្អ, ចម្លើយអាក្រក់) | តម្រង់ចំណូលចិត្ត ដោយគ្មាន reward model |
| **GRPO** | reward function/reward model | reasoning ខ្ពស់ (ដូច R1-style) · ថ្លៃ compute |

```python
# DPO ជាមួយ TRL 1.x
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=base_model,           # គួរជាម៉ូដែលដែល SFT រួច (adapter ក៏បាន)
    args=DPOConfig(output_dir="outputs/dpo", beta=0.1, learning_rate=5e-5),
    train_dataset=pref_dataset, # កូឡុន: prompt, chosen, rejected
    peft_config=lora_config,
    processing_class=tokenizer,
)
```

លំដាប់ណែនាំ៖ **SFT → (វាយតម្លៃ) → DPO** បើចាំបាច់។ GRPO សម្រាប់អ្នកមាន
reward ច្បាស់លាស់ (គណិតវិទ្យា/កូដ) និង compute ច្រើន។

## រក្សាសមត្ថភាព Reasoning ពេល fine-tune

ការណែនាំផ្លូវការសម្រាប់ Qwen3.5 (ពី Unsloth fine-tuning guide)៖

- បើចង់រក្សា reasoning → លាយឧទាហរណ៍ **reasoning-style** (មាន thinking
  block + ចម្លើយ) យ៉ាងតិច **៧៥%** នៃ dataset
- បើមិនត្រូវការ reasoning → អាចហ្វឹកតែចម្លើយផ្ទាល់ (direct answers)
- លាយទិន្នន័យទូទៅ ~១០–២៥% ដើម្បីកាត់បន្ថយ catastrophic forgetting
  (ការបាត់ជំនាញចាស់)

## Unsloth — ជម្រើសលឿន

[Unsloth](https://unsloth.ai/docs/models/qwen3.5/fine-tune) គាំទ្រ Qwen3.5 ពេញលេញ
(text + vision + RL)៖

- លឿន ~១.៥× និង VRAM −៥០% ធៀប FA2 setups
- bf16 LoRA VRAM៖ 0.8B ≈ 3GB · 2B ≈ 5GB · 4B ≈ 10GB · 9B ≈ 22GB
- Colab notebooks ឥតគិតថ្លៃសម្រាប់ 0.8B/2B/4B
- Export GGUF/vLLM ងាយ

```python
# គំរូ Unsloth (API ខុសពី transformers បន្តិច)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",  # ឬកំណែ Unsloth Dynamic លើ hf.co/unsloth
    max_seq_length=2048,
    load_in_4bit=False,  # bf16 LoRA — 4-bit មិនណែនាំសម្រាប់ Qwen3.5
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, target_modules="all-linear")
```

ចំណាំ៖ Unsloth ប្រើ transformers v5 ស្វ័យប្រវត្តិ។ បើការហ្វឹក «យឺតខុសធម្មតា»
នៅ step ដំបូង — នោះជា Triton kernel compilation សម្រាប់ Gated DeltaNet (ធម្មតា)។

## Qwen3.6 និង Qwen3.8 — អ្វីថ្មី?

| លក្ខណៈ | Qwen3.5 | Qwen3.6 | Qwen3.8 |
|---|---|---|---|
| ចេញផ្សាយ | កុម្ភៈ ២០២៦ | ២០២៦ | សីហា ២០២៦ |
| model_type | `qwen3_5` | `qwen3_5` (ដដែល) | `qwen3_5` (ដដែល) |
| transformers អប្បបរមា | 5.2 | 5.2 | **5.8** |
| ចំណុចខ្លាំង | VL native, 201 ភាសា | ស្ថិរភាព + agentic coding + thinking preservation | ថ្នាក់ Qwen-Max កូដចំហ · multi-step tasks |
| ការគ្រប់គ្រង reasoning | `enable_thinking` | + preserve thinking context | + `reasoning_effort` (low/medium/high), `preserve_thinking` |

ដំណឹងល្អសម្រាប់សៀវភៅនេះ៖ **អ្វីដែលអ្នកបានរៀនអនុវត្តបានទាំងអស់** —
Qwen3.6/3.8 ប្រើ architecture និង class ដូច Qwen3.5 (ផ្ទៀងផ្ទាត់ពី config.json ពិត
និងឯកសារ transformers)។ គ្រាន់តែដំឡើង transformers ឱ្យគ្រប់កំណែ។

```python
# Qwen3.8 — reasoning_effort kwarg
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True, reasoning_effort="medium",
)
```

(ពិនិត្យ model card នៃ checkpoint ជាក់លាក់សម្រាប់តម្លៃដែលគាំទ្រ —
feature នេះថ្មី ហើយឯកសារនៅតែវិវត្ត។)

## សង្ខេប

- QLoRA៖ តែ Qwen3 (២០២៥) · Qwen3.5+ → bf16 LoRA
- MoE៖ VRAM ធំ បង្កក router · all-linear នៅតែល្អ
- Qwen3.5 = VL from birth → AutoProcessor + AutoModelForImageTextToText សម្រាប់រូបភាព
- SFT → DPO → GRPO ជាជណ្តើរ post-training
- រក្សា reasoning៖ ≥៧៥% reasoning-style examples
- Qwen3.6/3.8 = architecture ដដែល · គ្រាន់តែត្រូវការ transformers ថ្មីជាង

## លំហាត់

1. សរសេរ dataset ៥ ឧទាហរណ៍ជាទម្រង់ DPO (prompt + chosen + rejected)
   សម្រាប់ជំនួយការខ្មែរ។
2. គណនា៖ 35B-A3B មាន active 3B params — ហេតុអ្វី inference លឿន តែហ្វឹកនៅតែ
   ត្រូវការ 74GB VRAM?
3. សាកល្បង (Colab)៖ QLoRA លើ `Qwen/Qwen3-0.6B` ជាមួយ `load_in_4bit=True`
   ហើយប្រៀបធៀប VRAM ជាមួយ bf16 LoRA។
4. អាន [Qwen3.5 blog](https://qwen.ai/blog?id=qwen3.5) ហើយសរសេរសេចក្តីសង្ខេប
   ៥ ចំណុចជាភាសាខ្មែរ។

---

ជំពូកមុន៖ {doc}`៨ · រក្សាទុក · Merge · Deploy <ch08-save-merge-deploy>` ·
ជំពូកបន្ទាប់៖ {doc}`១០ · Troubleshooting <ch10-troubleshooting>`
