# ១០ · Troubleshooting និង Hardware

```{contents} ក្នុងជំពូកនេះ
:local: true
:depth: 2
```

## គោលបំណងនៃជំពូក

ជំពូកយោង — កុំអានទាំងអស់ឥឡូវនេះ! មករកពេលអ្នកជួបបញ្ហា។

## កំហុសដំឡើង (Installation)

### `KeyError: 'qwen3_5'` ឬ «does not recognize this architecture»

**មូលហេតុ៖** transformers ចាស់ពេក (< 5.2) សម្រាប់ Qwen3.5/3.6។
Qwen3.8 ត្រូវការ ≥ 5.8។

```bash
pip install -U "transformers>=5.8,<6"
python -c "import transformers; print(transformers.__version__)"
```

### `CUDA error: no kernel image is available`

**មូលហេតុ៖** torch ដែលដំឡើងមិនត្រូវនឹង CUDA/GPU architecture។

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# ដំឡើងឡើងវិញតាម cu126 (ឬ cu128)
pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

### `ImportError: ... causal_conv1d` / DeltaNet យឺតខ្លាំង

Qwen3.5 ប្រើ Gated DeltaNet។ Kernel លឿនត្រូវការ package ស្រាល
`causal-conv1d` និង `fla` (flash-linear-attention)។ បើអត់ — ម៉ូដែល **នៅតែដំណើរការ**
តែ fallback ទៅ PyTorch ops យឺតជាង និងចំណាយ memory ច្រើនជាង។

```bash
pip install causal-conv1d   # បើ build បរាជ័យ វាមិនចាំបាច់ — គ្រាន់តែយឺត
```

### ការហ្វឹក «ជាប់» នៅ step ដំបូងរាប់នាទី

**ធម្មតាសម្រាប់ Qwen3.5!** Triton kernels សម្រាប់ Gated DeltaNet ត្រូវ compile
ជាលើកដំបូង (ជាពិសេសលើ T4)។ រង់ចាំ ៥–១៥ នាទី — step បន្ទាប់នឹងលឿនធម្មតា។

## កំហុស VRAM (Out of Memory)

លំដាប់នៃដំណោះស្រាយ (សាកម្តងមួយៗ)៖

1. **បន្ថយ `per_device_train_batch_size` → 1** ហើយដំឡើង `gradient_accumulation_steps` ដើម្បីរក្សា effective batch
2. **បន្ថយ `max_length`** (2048 → 1024 → 512) — activation memory ធ្លាក់ខ្លាំង
3. **បើក `gradient_checkpointing=True`** (default របស់ SFTConfig 1.x គឺ True រួចហើយ)
4. **បន្ថយ LoRA `r`** (64 → 16 → 8)
5. **ជ្រើសម៉ូដែលតូចជាង** (9B → 4B → 0.8B)
6. QLoRA 4-bit — **តែ Qwen3 (២០២៥) ប៉ុណ្ណោះ** មិនមែន Qwen3.5+
7. ជួល GPU ធំជាង (RunPod/Vast/Colab Pro A100)

### តារាង VRAM សង្ខេប (LoRA bf16 ហ្វឹក)

| ម៉ូដែល | VRAM | max_length ណែនាំ @ 16GB |
|---|---|---|
| Qwen3-0.6B | ~2 GB | 4096+ |
| Qwen3.5-0.8B | ~3 GB | 4096 |
| Qwen3.5-2B | ~5 GB | 2048–4096 |
| Qwen3.5-4B | ~10 GB | 1024–2048 |
| Qwen3.5-9B | ~22 GB | ត្រូវការ ≥24GB GPU |
| Qwen3.5-27B | ~56 GB | A100 80GB |
| Qwen3.5-35B-A3B | ~74 GB | A100 80GB (តឹង) |

## កំហុសការហ្វឹក (Training)

### Loss មិនធ្លាក់

- [ ] lr តូចពេក? សាក 1e-4
- [ ] dataset តូចពេក/ស្ទួនពេក?
- [ ] chat template ខុស? ប្រើ TRL ជាមួយ dataset `messages` (វាដាក់ template ឱ្យ) កុំ tokenize ដៃហើយបញ្ជូន `input_ids` ខុសទម្រង់
- [ ] `completion_only_loss=False` + prompt វែងៗ → signal ត្រូវលិច

### Loss = NaN

- ប្រើ **bf16** ជំនួស fp16 (fp16 overflow ងាយកើត)
- បន្ថយ lr · ពិនិត្យ grad_norm · ពិនិត្យទិន្នន័យខូច (ប្រវែង ០, encoding ខូច)

### ម៉ូដែល «បាត់បង់ភាសាអង់គ្លេស» ក្រោយហ្វឹកខ្មែរ

Catastrophic forgetting — លាយទិន្នន័យទូទៅ/ពហុភាសា ១០–២៥% ចូល dataset
ហើយកុំហ្វឹក epochs ច្រើនពេក (១–៣ គ្រប់គ្រាន់សម្រាប់ LoRA)។

## កំហុស Inference

### ចម្លើយដដែលៗមិនឈប់ (repetition loop)

- `presence_penalty=1.5` (ណែនាំផ្លូវការ Qwen3.5 Instruct)
- `repetition_penalty=1.05` ជាជម្រើសបន្ទាប់ (កុំលើស 1.2 — ធ្វើឱ្យភាសាខូច)
- កំណត់ `max_new_tokens`

### អក្សរចម្លែក/គ្មានន័យ (gibberish)

- **context តូចពេក** — Qwen3.5 ត្រូវការ context គ្រប់គ្រាន់; gibberish ជាញឹកញាប់
  មកពី num_ctx/max_length តូចពេកធៀប prompt
- dtype ខុស — ប្រើ bf16 ដូចការហ្វឹក
- adapter និង base **មិន match** — ផ្ទៀងផ្ទាត់ `adapter_config.json`
  (`base_model_name_or_path` ត្រូវតែជា base ដូចគ្នា)

### `ValueError: Cannot handle batch size > 1 if using padding_side='right'`

```python
tokenizer.padding_side = "left"
```

## កំហុស Google Colab

| បញ្ហា | ដំណោះស្រាយ |
|---|---|
| Session ដាច់ បាត់ checkpoint | output_dir → Google Drive |
| ដំឡើង package រួច ប៉ុន្តែ import បរាជ័យ | Runtime → Restart session → install ម្តងទៀតក្នុង notebook |
| T4 អស់ VRAM ជាមួយ 4B | ប្រើ Qwen3.5-0.8B ឬ max_length 1024 + bs 1 |
| «You are connected to a GPU that is not...» | Colab ឥតគិតថ្លៃផ្តល់ T4 — គ្រប់គ្រាន់សម្រាប់ម៉ូដែលតូច |

## វិធីសាស្ត្រ debug ជាប្រព័ន្ធ

1. **បង្រួមបញ្ហា** — dataset តូច (៥ ឧទាហរណ៍), ម៉ូដែលតូច, ១០ steps
   (ចុច {kbd}`Ctrl+C` ដើម្បីបញ្ឈប់ការហ្វឹកដែលកំពុងរត់)
2. **ផ្ទៀងផ្ទាត់ដំណាក់កាលនីមួយៗ** — `qwen-ft validate` → `qwen-ft env` →
   smoke test (`pytest -m heavy`) → ហ្វឹកពិត
3. **អាន error ពីក្រោមឡើងលើ** — traceback ជួរចុងក្រោយប្រាប់ចំណុចខូច
4. **ស្វែងរក error message ទាំងស្រុង** លើ GitHub issues នៃ transformers/trl/peft
5. ប្រើ `transformers` logging៖ `export TRANSFORMERS_VERBOSITY=debug`

## FAQ

**សួរ៖ គ្មាន GPU សោះ រៀនបានទេ?**
បាន! ជំពូក ១–៥ + notebooks ភាគច្រើនរត់លើ CPU។ ការហ្វឹកពិតប្រើ Colab (T4 ឥតគិតថ្លៃ)។

**សួរ៖ ត្រូវការទិន្នន័យប៉ុន្មានដើម្បីឱ្យម៉ូដែល «ខ្មែរពិតប្រាកដ»?**
ចាប់ផ្តើម ១,០០០–៥,០០០ ឧទាហរណ៍គុណភាពខ្ពស់។ Qwen3.5 គាំទ្រខ្មែរពីកំណើត
(២០១ ភាសា) — fine-tuning គ្រាន់តែកែរចនាបថ/ជំនាញ។

**សួរ៖ Fine-tune ម៉ូដែលដែលខ្លួនឯង pre-train បានទេ?**
ទ្រឹស្តីបាន តែ pre-training ត្រូវការ GPU រាប់រយ + ទិន្នន័យ trillion tokens។
ចាប់ផ្តើមពី checkpoint ផ្លូវការជានិច្ច។

**សួរ៖ License?**
Qwen3 និង Qwen3.5 ភាគច្រើនជា **Apache 2.0** — ពាណិជ្ជកម្មបាន។
តែងពិនិត្យ LICENSE ក្នុង repo នៃ checkpoint នីមួយៗ។

## សង្ខេប

- កំហុស version → `pip install -U "transformers>=5.8,<6"`
- OOM → បន្ថយ batch/length/r តាមលំដាប់
- យឺត step ដំបូង (Qwen3.5) = kernel compilation — ធម្មតា
- gibberish = context តូច ឬ adapter/base មិន match
- debug ជាប្រព័ន្ធ៖ បង្រួម → ផ្ទៀងផ្ទាត់ → អាន traceback

---

ជំពូកមុន៖ {doc}`៩ · ប្រធានបទកម្រិតខ្ពស់ <ch09-advanced>` ·
សូមមើលបន្ថែម៖ {doc}`តារាង Compatibility </compatibility>` និង
{doc}`ឯកសារយោង </references>`
