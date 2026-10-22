# ៥ · LoRA ពីគោលការណ៍ — ហ្វឹកម៉ូដែលធំដោយ VRAM តូច

```{contents} ក្នុងជំពូកនេះ
:local: true
:depth: 2
```

## គោលបំណងនៃជំពូក

- យល់ថាហេតុអ្វី full fine-tuning ថ្លៃពេក និង LoRA ដោះស្រាយយ៉ាងដូចម្តេច
- ជ្រើសរើសប៉ារ៉ាម៉ែត្រ `r`, `alpha`, `dropout`, `target_modules` ដោយយល់ដឹង
- ដឹងពី VRAM ដែលត្រូវការសម្រាប់ម៉ូដែល Qwen នីមួយៗ
- យល់ពីភាពខុសគ្នា LoRA vs QLoRA និងហេតុអ្វី QLoRA **មិនណែនាំ** សម្រាប់ Qwen3.5+

## ហេតុអ្វី full fine-tuning ថ្លៃពេក?

ពេលហ្វឹកបណ្តាញណឺរ៉ូន អ្នកត្រូវរក្សាទុកក្នុង VRAM ៤ យ៉ាង (ជាមួយ Adam optimizer, bf16)៖

| ធាតុ | ទំហំធៀបនឹងម៉ូដែល 4B |
|---|---|
| ទម្ងន់ (weights) | ~8 GB |
| Gradients | ~8 GB |
| Optimizer states (Adam m + v, fp32) | ~32 GB |
| Activations | ប្រែប្រួលតាម batch/length |
| **សរុប** | **~50–60 GB+** |

នេះហួសពី GPU ទំនើបភាគច្រើន (16–24 GB)។

## គំនិតរបស់ LoRA

**LoRA** (Low-Rank Adaptation, Hu et al. 2021) សង្កេតឃើញថា ការកែប្រែទម្ងន់
ពេល fine-tuning មាន « intrinsic rank » ទាប។ ជំនួសឱ្យការកែម៉ាទ្រីស $W$ ផ្ទាល់
(d×d) យើងបន្ថែមផលបូកម៉ាទ្រីសតូចពីរ៖

$$
W' = W + \Delta W = W + \frac{\alpha}{r} \cdot BA
$$

ដែល $B \in \mathbb{R}^{d \times r}$ និង $A \in \mathbb{R}^{r \times d}$ ជាមួយ $r \ll d$
(ឧ. r=16 ខណៈ d=2560)។

លទ្ធផល៖

- **បង្កក** $W$ ដើមទាំងស្រុង — គ្មាន gradient, គ្មាន optimizer state សម្រាប់វា
- ហ្វឹកតែ $A$ និង $B$ — ជាធម្មតា **០.១–១%** នៃប៉ារ៉ាម៉ែត្រសរុប
- VRAM សម្រាប់ Qwen3.5-4B ធ្លាក់មក **~10 GB** (bf16 LoRA — តួលេខផ្លូវការពី Unsloth)
- Adapter មានទំហំតូច (រាប់សិប MB) → ចែកចាយងាយ ហើយអាចមាន adapter ច្រើនលើ base តែមួយ

## ប៉ារ៉ាម៉ែត្រ LoraConfig — ការណែនាំជាក់ស្តែង

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # rank — ទំហំ «សមត្ថភាពរៀន»
    lora_alpha=32,           # មាត្រដ្ឋាន — ច្បាប់ទូទៅ alpha = 2r
    lora_dropout=0.05,       # regularization — 0.05 សម្រាប់ dataset តូច
    target_modules="all-linear",  # ដាក់ LoRA លើស្រទាប់ linear ទាំងអស់
    bias="none",
    task_type="CAUSAL_LM",
)
```

| ប៉ារ៉ាម៉ែត្រ | តម្លៃណែនាំ | ការពន្យល់ |
|---|---|---|
| `r` | 8–64 (ចាប់ផ្តើម 16) | r ធំ = សមត្ថភាពរៀនច្រើន + VRAM ច្រើន។ កិច្ចការងាយ (រចនាបថ) r=8 គ្រប់គ្រាន់ |
| `lora_alpha` | 2×r | កំណត់ទំហំឥទ្ធិពល ΔW។ អ្នកខ្លះប្រើ alpha=r ឬ 4r — ផ្លាស់ប្តូរតែម្តងមួយៗ |
| `lora_dropout` | 0.0–0.1 | dataset តូច → ប្រើ 0.05 ដើម្បីកាត់បន្ថយ overfitting |
| `target_modules` | `"all-linear"` | សាមញ្ញ និងល្អបំផុតសម្រាប់ Qwen។ ការកំណត់ដៃ (`q_proj`, `k_proj`, …) អាចរំលងស្រទាប់ថ្មី |

```{important} ហេតុអ្វី **all-linear** សំខាន់សម្រាប់ Qwen3.5?
Qwen3.5 ជា **hybrid architecture** — ក្រៅពី attention ធម្មតា (q/k/v/o_proj)
វាមានស្រទាប់ Gated DeltaNet ដែលមានឈ្មោះខុសគ្នា (`in_proj_*`, `out_proj`, `conv1d` ជាដើម)
និង MLP (`gate/up/down_proj`)។ បើអ្នកសរសេរ target_modules ដោយដៃតែ
`["q_proj","k_proj","v_proj","o_proj"]` អ្នកនឹងរំលងស្រទាប់ភាគច្រើននៃម៉ូដែល។
`"all-linear"` គ្របទាំងអស់ដោយស្វ័យប្រវត្តិ (ដោយរំលង `lm_head`)។
```

## តារាង VRAM — LoRA bf16 (តួលេខផ្លូវការពី Unsloth, Qwen3.5)

| ម៉ូដែល | VRAM ហ្វឹក LoRA bf16 | GPU ដែលអាច |
|---|---|---|
| Qwen3.5-0.8B | ~3 GB | Colab T4 (16GB) ស្រួល |
| Qwen3.5-2B | ~5 GB | Colab T4 ស្រួល |
| Qwen3.5-4B | ~10 GB | T4/L4/3080 |
| Qwen3.5-9B | ~22 GB | 4090 (24GB) |
| Qwen3.5-27B | ~56 GB | A100 80GB |
| Qwen3.5-35B-A3B (MoE) | ~74 GB | A100 80GB |

ចំពោះជំនាន់ Qwen3 (២០២៥) លេខក៏ប្រហាក់ប្រហែលគ្នា (0.6B ~2GB, 4B ~9GB, 8B ~18GB)។
លេខទាំងនេះ **មិនរាប់បញ្ចូល** activation នៃ context វែង — បើ `max_length` ធំ
VRAM កើនបន្ថែម។

## LoRA vs QLoRA

| | LoRA (bf16) | QLoRA (4-bit NF4) |
|---|---|---|
| ទម្ងន់ base រក្សាជា | bf16 | 4-bit quantized |
| VRAM | ខ្ពស់ជាងបន្តិច | ទាបបំផុត |
| គុណភាព | ស្តង់ដារ | ខ្ពស់ជិតស្មើ (លើ Qwen3 ២០២៥) |
| **Qwen3 (២០២៥)** | ✅ | ✅ អាចប្រើបាន |
| **Qwen3.5/3.6/3.8** | ✅ ណែនាំ | ❌ **មិនណែនាំ** |

```{warning} ហេតុអ្វី QLoRA មិនណែនាំសម្រាប់ Qwen3.5+?
ការណែនាំផ្លូវការ (Unsloth Qwen3.5 guide) ព្រមានថា Qwen3.5 មាន
«higher than normal quantization differences» — tensor មួយចំនួនងាយរងគ្រោះ
ពី quantization 4-bit ខ្លាំង។ លើសពីនេះ MoE + bitsandbytes មានដែនកំណត់បន្ថែម។
ចម្លើយសាមញ្ញ៖ ប្រើ **bf16 LoRA** សម្រាប់ Qwen3.5 ឡើងទៅ។
```

ចំណាំបន្ថែមសម្រាប់ **MoE** (35B-A3B, 122B-A10B)៖ ស្រទាប់ **router** ជាធម្មតា
ត្រូវបានបង្កក (មិនហ្វឹក) ដើម្បីស្ថិរភាព — router ដែលរៀនខុសអាចបំផ្លាញ
ការចែក expert ទាំងមូល។

## គណនាចំនួនប៉ារ៉ាម៉ែត្រ LoRA

ចំពោះស្រទាប់ linear មួយ d×k ជាមួយ rank r៖ ប៉ារ៉ាម៉ែត្រ LoRA = $r(d+k)$។
ឧទាហរណ៍ Qwen3.5-4B attention projection d=2560, r=16 →
$16 \times (2560+2560) = 81{,}920$ ប៉ារ៉ាម៉ែត្រក្នុងមួយស្រទាប់
(ប្រៀបធៀបនឹង $2560^2 = 6.5$ លានបើហ្វឹកពេញលេញ — តិចជាង ~៨០ ដង!)។

ពេលហ្វឹកចប់ `peft` បង្ហាញសង្ខេបដូចជា៖
`trainable params: 20,971,520 || all params: 4,000,000,000 || trainable%: 0.52`

## សង្ខេប

- LoRA = បង្កកទម្ងន់ដើម + ហ្វឹកម៉ាទ្រីស low-rank តូចៗ → VRAM ថយច្រើនដង
- ចាប់ផ្តើមជាមួយ `r=16, alpha=32, dropout=0.05, target_modules="all-linear"`
- Qwen3.5+ → **bf16 LoRA** ជានិច្ច (QLoRA 4-bit មិនណែនាំ)
- MoE → បង្កក router ស្រាប់ហើយក្នុង tooling ភាគច្រើន
- Adapter តូច ចែកចាយងាយ ហើយអាច merge ចូល base ពេល deploy (ជំពូក ៨)

## លំហាត់

1. គណនាដោយដៃ៖ ប៉ារ៉ាម៉ែត្រ LoRA សរុបសម្រាប់ MLP របស់ Qwen3.5-4B
   (intermediate 9216 ↔ hidden 2560, r=16, ស្រទាប់ gate/up/down ទាំង ៣២)។
2. សាកល្បង `LoraConfig` ផ្សេងៗក្នុង notebook
   {doc}`០៤ LoRA Config Lab </../notebooks/04_lora_config_lab>` —
   តើ r=8 និង r=64 ខុសគ្នាប៉ុន្មាន trainable %?
3. ពន្យល់ជាភាសាខ្មែរ (៣ ប្រយោគ) ហេតុអ្វី QLoRA មិនសាកសមនឹង Qwen3.5-4B។
4. GPU អ្នកមាន VRAM ប៉ុន្មាន? (`nvidia-smi`) — តើម៉ូដែលណាធំបំផុតដែលអ្នកអាចហ្វឹកបាន?

---

ជំពូកមុន៖ {doc}`៤ · ត្រៀមទិន្នន័យ <ch04-datasets>` ·
ជំពូកបន្ទាប់៖ {doc}`៦ · ហ្វឹក SFT <ch06-training-sft>`
