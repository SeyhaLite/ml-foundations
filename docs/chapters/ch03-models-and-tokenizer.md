# ៣ · ម៉ូដែល និង Tokenizer — ផ្ទុក Qwen ឱ្យត្រូវវិធី

```{contents} ក្នុងជំពូកនេះ
:local: true
:depth: 2
```

## គោលបំណងនៃជំពូក

- ផ្ទុក Qwen3 និង Qwen3.5 ដោយប្រើ API ត្រឹមត្រូវ
- យល់ពីភាពខុសគ្នា `AutoModelForCausalLM` vs `AutoModelForImageTextToText`/Processor
- ប្រើ **chat template** និង `enable_thinking` ឱ្យបានត្រូវ
- ស្គាល់ tokenizer របស់ Qwen និងលក្ខណៈពិសេសសម្រាប់ភាសាខ្មែរ

## ជ្រើសរើសម៉ូដែល — មិនមែនជ្រើសដោយចៃដន្យទេ!

ជំពូកទី ១ បានណែនាំជំនាន់នីមួយៗ។ សម្រាប់ការរៀន យើងណែនាំ៖

| ស្ថានភាពអ្នក | ម៉ូដែលណែនាំ | ហេតុអ្វី |
|---|---|---|
| Colab ឥតគិតថ្លៃ (T4 16GB) | `Qwen/Qwen3-0.6B` ឬ `Qwen/Qwen3.5-0.8B` | LoRA bf16 ត្រឹម ~2–3 GB VRAM |
| GPU 16–24GB (4090/L4) | `Qwen/Qwen3.5-4B` | ~10 GB VRAM គុណភាពល្អបំផុតក្នុងទំហំនេះ |
| GPU ≥ 32GB | `Qwen/Qwen3.5-9B` | ~22 GB VRAM |

```{warning} កុំច្រឡំ model ID
`Qwen/Qwen3-4B` (ជំនាន់ ២០២៥) **ខុសពី** `Qwen/Qwen3.5-4B` (ជំនាន់ ២០២៦)។
ឧទាហរណ៍កូដក្នុងសៀវភៅនេះប្រើ `Qwen/Qwen3.5-4B` — បើអ្នកប្រើ `Qwen3-4B` ជំនួស
ក៏ដំណើរការដែរ (transformers>=4.51) តែលទ្ធផល និង VRAM នឹងខុសពីតារាង។
```

## ផ្ទុកម៉ូដែល — Qwen3 (ជំនាន់ ២០២៥)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,   # GPU ទំនើប (Ampere+) — ប្រើ bf16
    device_map="auto",
)
```

- Architecture ដែលទទួលបាន៖ `Qwen3ForCausalLM` (`model_type: qwen3`)
- Vocab 151,936 token · Context 32,768 (config អនុញ្ញាតដល់ 40,960)

## ផ្ទុកម៉ូដែល — Qwen3.5 (ជំនាន់ ២០២៦)

Qwen3.5 checkpoint មាន **ពីរផ្នែកក្នុងឯកសារតែមួយ**៖ text backbone
(`qwen3_5_text`) និង vision encoder។ ដូច្នេះមានផ្លូវផ្ទុកពីរ៖

### ផ្លូវទី ១ — Text-only fine-tuning (អ្វីដែលសៀវភៅនេះប្រើ)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# transformers v5 map ទៅ Qwen3_5ForCausalLM (text backbone) ដោយស្វ័យប្រវត្តិ
```

### ផ្លូវទី ២ — Multimodal (រូបភាព + អត្ថបទ)

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-4B", torch_dtype=torch.bfloat16, device_map="auto"
)
# ទទួលបាន Qwen3_5ForConditionalGeneration — អាចបញ្ចូលរូបភាព/វីដេអូ
```

ព័ត៌មានលម្អិត architecture (ផ្ទៀងផ្ទាត់ពី config.json ពិត):

| លក្ខណៈ | Qwen3.5-4B |
|---|---|
| `model_type` | `qwen3_5` (top) / `qwen3_5_text` (backbone) |
| ស្រទាប់ | 32 (24 × Gated DeltaNet + 8 × Gated Attention តាមលំនាំ ៣:១) |
| hidden size / head dim | 2560 / 256 |
| attention heads (Q/KV) | 16 / 4 |
| vocab | **248,320** (ធំជាង Qwen3 ដល់ ៦៣% — គាំទ្រ ២០១ ភាសា) |
| context | 262,144 token |

## Tokenizer — ស្គាល់ «អក្សរ» របស់ម៉ូដែល

```python
tokens = tokenizer.tokenize("សួស្តី ខ្ញុំឈ្មោះ សុខា")
print(tokens)
print(len(tokenizer.encode("សួស្តី ខ្ញុំឈ្មោះ សុខា")))
```

ចំណុចគួរដឹងសម្រាប់ភាសាខ្មែរ៖

1. **ខ្មែរគ្មានដកឃ្លា** រវាងពាក្យ — BPE tokenizer របស់ Qwen បំបែកតាមអក្សរ/ក្រុមអក្សរ។
   អត្ថបទខ្មែរ ១ ពាក្យអាចចំណាយ token ច្រើនជាងអង់គ្លេស។
2. Qwen3.5 vocab ធំ (248,320) ធៀបនឹង Qwen3 (151,936) → ភាសាខ្មែរត្រូវបានគ្របដណ្តប់ល្អជាង។
3. កុំវាស់ប្រវែង dataset ជា «ពាក្យ» — វាស់ជា **token** ដើម្បីកំណត់ `max_length`។

### Padding side — ច្បាប់មាសសម្រាប់ generation

```python
tokenizer.padding_side = "left"   # សម្រាប់ batch generation
```

បើ pad ខាងស្តាំ ម៉ូដែលនឹងបន្តពី token pad ជំនួសពាក្យចុងក្រោយ → ចម្លើយខូច។
(ឯកសារផ្លូវការ transformers ណែនាំ left padding សម្រាប់ decoder-only LLMs។)

## Chat Template — ភាសាសន្ទនារបស់ Qwen

LLM មិនទទួល «សារឆៅ» ទេ — វាត្រូវការទម្រង់ពិសេស (chat template)។
**កុំសរសេរ template ដោយដៃ** — ប្រើ `apply_chat_template`៖

```python
messages = [
    {"role": "system", "content": "អ្នកជាជំនួយការភាសាខ្មែរ។"},
    {"role": "user", "content": "តើ LoRA គឺជាអ្វី?"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,   # Qwen3 និង Qwen3.5 ទាំងពីរគាំទ្រ kwarg នេះ
)
print(prompt)
```

### `enable_thinking` — បើក/បិទការ «គិត»

| ម៉ូដែល | Default | របៀបប្តូរ |
|---|---|---|
| Qwen3 (0.6B–32B, ២០២៥) | thinking **បើក** | `enable_thinking=False` |
| Qwen3-…-Instruct-2507 | មិនគិត (ស្ថិរ) | ប្រើ `-Thinking-2507` បើចង់គិត |
| Qwen3.5 Small (0.8B–9B) | thinking **បិទ** | `enable_thinking=True` |
| Qwen3.5 ≥ 27B, Qwen3.6/3.8 | thinking **បើក** | `enable_thinking=False` |
| Qwen3.8 បន្ថែម | — | `reasoning_effort="low|medium|high"`, `preserve_thinking=True` |

```{note}
ហេតុអ្វីសំខាន់សម្រាប់ fine-tuning? ពេលហ្វឹក អ្នកត្រូវជ្រើស **របៀបមួយ** ឱ្យច្បាស់
ហើយប្រើវាទាំងក្នុង dataset និងពេល inference។ បើ dataset មាន thinking block (tag `think` ដែលហ៊ុំដោយសញ្ញាមុំពីរខាង) តែ inference បិទ thinking — ម៉ូដែលនឹងឆ្លើយខុសរចនាបថ។
```

## Generation ដំបូងរបស់អ្នក

```python
import torch

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,     # តម្លៃណែនាំផ្លូវការ (Instruct general)
        top_p=0.8,
        top_k=20,
    )
reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(reply)
```

ចំណាំ៖ `output[0][inputs["input_ids"].shape[1]:]` កាត់យកតែផ្នែក **ឆ្លើយថ្មី** —
ព្រោះ `generate()` ត្រឡប់ prompt + ចម្លើយជាប់គ្នា។

តម្លៃ sampling ណែនាំផ្លូវការសម្រាប់ Qwen3.5 (ពេញលេញនៅជំពូក ៧)៖
Instruct ទូទៅ `temperature=0.7, top_p=0.8, top_k=20` · Thinking ទូទៅ
`temperature=1.0, top_p=0.95, top_k=20`។

## សង្ខេប

- Qwen3 → `AutoModelForCausalLM` លើ transformers ≥4.51 · Qwen3.5/3.6 → transformers ≥5.2 · Qwen3.8 → ≥5.8
- Qwen3.5 checkpoint ជា multimodal ដោយកំណើត — ប្រើ `AutoModelForImageTextToText` + Processor បើចង់បញ្ចូលរូបភាព
- ប្រើ `apply_chat_template` ជានិច្ច — កុំសរសេរ template ដៃ
- `enable_thinking` មាន default ខុសគ្នាតាមទំហំ/ជំនាន់ — ត្រួតពិនិត្យមុនហ្វឹក
- `padding_side="left"` សម្រាប់ batch generation

## លំហាត់

1. ផ្ទុក `Qwen/Qwen3-0.6B` (ឬ 0.8B បើ VRAM អំណោយផល) ហើយសួរវាជាភាសាខ្មែរ ៣ សំណួរ។
2. ប្រៀបធៀប `tokenize("ខ្មែរ")` រវាង Qwen3 និង Qwen3.5 — មួយណាបំបែកជា token តិចជាង?
3. សាកល្បង `enable_thinking=True` និង `False` លើសំណួរគណិតវិទ្យាមួយ —
   តើចម្លើយខុសគ្នាយ៉ាងដូចម្តេច?
4. រត់ notebook {doc}`០២ Chat Templates </../notebooks/02_chat_templates>` ឱ្យចប់។

---

ជំពូកមុន៖ {doc}`២ · ដំឡើងបរិស្ថាន <ch02-setup-environment>` ·
ជំពូកបន្ទាប់៖ {doc}`៤ · ត្រៀមទិន្នន័យ <ch04-datasets>`
