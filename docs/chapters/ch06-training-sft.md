# ៦ · ហ្វឹក SFT — ការហ្វឹកពិតប្រាកដជាមួយ TRL

```{contents} ក្នុងជំពូកនេះ
:local: true
:depth: 2
```

## គោលបំណងនៃជំពូក

- យល់ពីដំណើរការ **Supervised Fine-Tuning (SFT)** និង loss function
- ប្រើ `SFTTrainer` + `SFTConfig` (trl ≥ 1.0) ជាមួយ LoRA
- កំណត់ hyperparameters ដោយមានមូលដ្ឋាន
- តាមដានការហ្វឹក រក្សាទុក checkpoint និង resume

## SFT គឺជាអ្វី?

**Supervised Fine-Tuning** គឺការហ្វឹកម៉ូដែលឱ្យទស្សន៍ទាយ token នៃ **ចម្លើយ**
ដោយ minimize negative log-likelihood៖

$$
\mathcal{L}_{SFT}(\theta) = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t})
$$

ចំណុចសំខាន់ដែលអ្នកចាប់ផ្តើមតែងភ្លេច៖ យើង **មិនចង់** គណនា loss លើ prompt ទេ —
ត្រឹមតែលើចម្លើយរបស់ assistant។ នេះហើយជាតួនាទីរបស់ `completion_only_loss=True`
(TRL ដាក់ label token នៃ prompt ជា `-100` ដើម្បីមិនគណនា loss)។

## កូដហ្វឹកពេញលេញ — តាមរយៈឧបករណ៍ `qwen-ft`

វិធីងាយបំផុតក្នុងគម្រោងនេះ៖ កែ config YAML ហើយរត់ command តែមួយ។

```bash
# ១. ត្រួតពិនិត្យ config + dataset (គ្មាន GPU ក៏បាន)
qwen-ft validate configs/qwen3.5-4b-lora.yaml

# ២. ហ្វឹក (ត្រូវការ GPU + stack ពេញលេញ)
qwen-ft train configs/qwen3.5-4b-lora.yaml
```

ខាងក្នុង `qwen-ft train` ធ្វើអ្វីខ្លះ? សូមមើល {py:mod}`qwen_ft.training` —
វាផ្ទៀងផ្ទាត់ config, ត្រួតពិនិត្យ dataset, ត្រួតពិនិត្យ **កំណែ transformers**
(បើ < 5.2 សម្រាប់ Qwen3.5 → បញ្ឈរភ្លាមជាមួយសារច្បាស់លាស់ ជំនួស `KeyError` ចម្លែក),
បង្កើត LoRA + SFTTrainer ហើយរក្សាទុក adapter ទៅ `outputs/.../adapter/`។

## កូដហ្វឹកពេញលេញ — សរសេរដោយផ្ទាល់

បើអ្នកចង់យល់គ្រប់ជំហាន (នេះជាអ្វីដែល `qwen_ft.training` ធ្វើខាងក្នុង)៖

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3.5-4B"

# ១. Tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
model.config.use_cache = False  # មិនអាចប្រើជាមួយ gradient checkpointing

# ២. Dataset (ទម្រង់ messages — trainer ដាក់ chat template ឱ្យស្វ័យប្រវត្តិ)
dataset = load_dataset("json", data_files="data/sample_khmer_assistant.jsonl", split="train")

# ៣. LoRA
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear", task_type="CAUSAL_LM",
)

# ៤. Training arguments
args = SFTConfig(
    output_dir="outputs/qwen3.5-4b-khmer",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,      # effective batch = 16
    learning_rate=1e-4,                  # LoRA ទទួលយក lr ខ្ពស់ជាង full FT
    warmup_ratio=0.03,
    max_length=2048,                     # token — មិនមែនអក្សរ!
    completion_only_loss=True,           # រៀនតែពីចម្លើយ assistant
    packing=False,                       # True = ខ្ចប់ឧទាហរណ៍ខ្លីចូលគ្នា (លឿនជាង)
    gradient_checkpointing=True,         # ជំនួស VRAM ដោយការគណនាឡើងវិញ
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=200,
    seed=42,
    report_to="none",                    # ឬ "trackio"/"wandb"/"tensorboard"
)

# ៥. ហ្វឹក!
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model("outputs/qwen3.5-4b-khmer/adapter")
tokenizer.save_pretrained("outputs/qwen3.5-4b-khmer/adapter")
```

```{note} API ថ្មីរបស់ TRL 1.x
`SFTTrainer` អាចទទួល `model="Qwen/Qwen3-0.6B"` ជា **string** ផង —
វានឹងផ្ទុកម៉ូដែលឱ្យស្វ័យប្រវត្តិ។ យើងបង្ហាញការផ្ទុកដោយដៃ ដើម្បីឱ្យអ្នក
គ្រប់គ្រង dtype/device បានច្បាស់។ ឈ្មោះ argument ក៏ប្រែប្រួលពី TRL 0.x៖
`max_seq_length` → `max_length`, `dataset_text_field` មិនទៀតទេ (ប្រើ formatting func)។
```

```{warning} chat template គឺចាំបាច់សម្រាប់ dataset `messages`
TRL ប្រើ `tokenizer.apply_chat_template()` លើ dataset សន្ទនា។ បើ tokenizer
គ្មាន `chat_template` → `ValueError: Cannot use chat template functions...`។
Tokenizer របស់ Qwen គ្រប់ checkpoint មាន template ភ្ជាប់ស្រាប់ — បញ្ហានេះកើតមាន
តែពេលអ្នកប្រើ tokenizer ផ្ទាល់ខ្លួន ឬ tokenizer ដែលរក្សាទុកមិនពេញលេញ។
(បានផ្ទៀងផ្ទាត់ដោយ smoke test របស់ repo នេះលើ trl 1.16.0។)
```

## Hyperparameters — តារាងណែនាំ

| Argument | តម្លៃណែនាំ (LoRA SFT) | ហេតុផល |
|---|---|---|
| `learning_rate` | 1e-4 (0.6B–4B) · 5e-5 (≥9B) | LoRA ប្រើ lr ខ្ពស់ជាង full FT (2e-5) |
| `num_train_epochs` | 1–3 | dataset តូច → 3 · dataset ធំ → 1–2 (overfitting) |
| effective batch (`bs × grad_accum`) | 16–64 | ស្ថិរភាព gradient · បើ VRAM តិច → bs=1, accum=16 |
| `max_length` | 1024–2048 | គ្រប ៩៥% នៃឧទាហរណ៍ · token វែង = VRAM ច្រើន (quadratic សម្រាប់ full attention) |
| `warmup_ratio` | 0.03–0.05 | ការពារ loss spike ដំបូង |
| `lr_scheduler_type` | linear (default) | cosine ក៏ល្អដែរ |
| `gradient_checkpointing` | True | ថយ VRAM activation ~ច្រើន ដូរនឹង ~២០–៣០% ល្បឿន |
| `bf16` | True (GPU Ampere+) | fp16 អាច overflow ជាមួយម៉ូដែលទំនើប |
| `packing` | False សម្រាប់ចាប់ផ្តើម | True អាចលាយឧទាហរណ៍ច្រើនក្នុង sequence មួយ — លឿន តែត្រូវប្រាកដថា attention mask ញែកដាច់ |

## តាមដានការហ្វឹក

```text
{'loss': 2.31, 'grad_norm': 1.84, 'learning_rate': 9.7e-5, 'epoch': 0.05}
{'loss': 1.87, 'grad_norm': 1.52, 'learning_rate': 9.4e-5, 'epoch': 0.10}
{'loss': 1.42, 'grad_norm': 1.31, 'learning_rate': 9.1e-5, 'epoch': 0.15}
```

របៀបអាន៖

- **loss** គួរធ្លាក់ចុះជាលំដាប់ក្នុង ~១០០ step ដំបូង ហើយរាបស្មើបន្តិចម្តងៗ។
  loss ដែល **មិនធ្លាក់សោះ** → lr តូចពេក ឬ data/template ខុស។
  loss ធ្លាក់លឿនពេកដល់ ~០ → អាច overfit ឬមាន data leak។
- **grad_norm** គួរនៅស្ថិរ (~០.៥–៥)។ បើវាឡើងខ្ពស់រាប់រយ → អស្ថិរភាព បន្ថយ lr។
- **eval_loss** ធ្លាក់ហើយ **ឡើងវិញ** = ចំណុច overfitting → បញ្ឈប់ត្រង់ checkpoint ល្អបំផុត។

ចំណាំសម្រាប់ Qwen3.5៖ step ដំបូងៗអាច **យឺតខុសធម្មតា** ព្រោះ Triton kernels
សម្រាប់ Gated DeltaNet ត្រូវ compile (ជាពិសេសលើ T4)។ នេះធម្មតា — មិនមែនម៉ាស៊ីនអ្នកខូចទេ។

## Checkpoint និង Resume

`save_steps=200` រក្សាទុក checkpoint ក្នុង `outputs/.../checkpoint-200/` ជាដើម។
បើការហ្វឹកដាច់ពាក់កណ្តាល (Colab disconnect!)៖

```bash
qwen-ft train configs/qwen3.5-4b-lora.yaml \
    --resume-from outputs/qwen3.5-4b-khmer/checkpoint-400
```

```python
# ឬដោយផ្ទាល់
trainer.train(resume_from_checkpoint="outputs/qwen3.5-4b-khmer/checkpoint-400")
```

```{tip} Colab + Google Drive
ភ្ជាប់ Drive ហើយដាក់ `output_dir` ទៅ `/content/drive/MyDrive/qwen-checkpoints/...`
ដើម្បីកុំបាត់ checkpoint ពេល session ដាច់។
```

## លទ្ធផលដែលត្រូវបានរក្សាទុក

```text
outputs/qwen3.5-4b-khmer/adapter/
├── adapter_config.json        # LoraConfig ដែលបានប្រើ (r, alpha, target_modules...)
├── adapter_model.safetensors  # ទម្ងន់ LoRA តែប៉ុណ្ណោះ (~រាប់សិប MB)
├── tokenizer.json + ឯកសារ tokenizer
└── chat_template ពាក់ព័ន្ធ (រក្សាទុកជាមួយ tokenizer)
```

ជំពូក ៧ និង ៨ បង្ហាញការផ្ទុកវាឡើងវិញ ការវាយតម្លៃ និងការ merge។

## សង្ខេប

- SFT = រៀនទស្សន៍ទាយចម្លើយ · `completion_only_loss=True` ជានិច្ច
- TRL 1.x៖ `SFTTrainer(model=..., args=SFTConfig(...), peft_config=LoraConfig(...))`
- lr 1e-4, epochs 1–3, effective batch ≥16, `max_length` តាម token ពិតនៃ dataset
- តាមដាន loss/grad_norm/eval_loss — ដឹងពីសញ្ញា overfit
- Qwen3.5 មាន kernel compile យឺតនៅ step ដំបូង — ធម្មតា

## លំហាត់

1. រត់ `qwen-ft validate` លើ config ទាំង ៣ ក្នុង `configs/` — សរសេរផ្ទាំងតារាង
   effective batch size និងលេខ optimizer step សរុបសម្រាប់ dataset ២៤ ឧទាហរណ៍។
2. បើអ្នកមាន GPU៖ ហ្វឹក `configs/qwen3-0.6b-lora.yaml` (ឬ 3.5-0.8B) ១ epoch
   ហើយថតក្រាហ្វ loss។ បើគ្មាន GPU៖ រត់ smoke test
   `pytest -m heavy` លើម៉ាស៊ីនដែលមាន stack ពេញ ឬអានលទ្ធផលក្នុង
   {doc}`notebook ០៥ </../notebooks/05_smoke_training>`។
3. ពិសោធន៍៖ ប្តូរ `learning_rate` ទៅ 1e-2 ក្នុង config សាកល្បង —
   validator ត្រូវបដិសេធ។ ហេតុអ្វីយើងកំណត់ដែន [1e-7, 1e-2]?
4. គិត៖ ហេតុអ្វី `use_cache` ត្រូវបិទពេលហ្វឹកជាមួយ gradient checkpointing?
   (គន្លឹះ៖ KV cache រក្សា activation សម្រាប់ inference តែប៉ុណ្ណោះ។)

---

ជំពូកមុន៖ {doc}`៥ · LoRA ពីគោលការណ៍ <ch05-lora-fundamentals>` ·
ជំពូកបន្ទាប់៖ {doc}`៧ · វាយតម្លៃ និង Inference <ch07-evaluation-inference>`
