# ៤ · ត្រៀមទិន្នន័យ — Dataset ខ្មែរត្រឹមត្រូវ

```{contents} ក្នុងជំពូកនេះ
:local: true
:depth: 2
```

## គោលបំណងនៃជំពូក

- ស្គាល់ទម្រង់ទិន្នន័យដែល `trl.SFTTrainer` ទទួលយក
- សរសេរ dataset ជា **JSONL** ត្រឹមត្រូវសម្រាប់ភាសាខ្មែរ
- ផ្ទៀងផ្ទាត់ dataset ដោយស្វ័យប្រវត្តិជាមួយ `qwen_ft.data`
- យល់ពីបរិមាណ និងគុណភាពទិន្នន័យដែលត្រូវការ

## ទម្រង់ទិន្នន័យដែល TRL ទទួលយក

`SFTTrainer` (trl ≥ 1.0) ទទួលយកទម្រង់ស្តង់ដារ ៤ យ៉ាង។ សៀវភៅនេះប្រើ
**conversational language modeling** (ទម្រង់ `messages`)៖

```python
# ១. Conversational (យើងប្រើទម្រង់នេះ) — trainer ដាក់ chat template ឱ្យស្វ័យប្រវត្តិ
{"messages": [
    {"role": "user", "content": "តើអ្វីជា LoRA?"},
    {"role": "assistant", "content": "LoRA គឺជា..."}
]}

# ២. Standard language modeling
{"text": "អត្ថបទទាំងមូល..."}

# ៣. Conversational prompt-completion (ហ្វឹកតែចម្លើយ)
{"prompt": [{"role": "user", "content": "..."}],
 "completion": [{"role": "assistant", "content": "..."}]}

# ៤. Standard prompt-completion
{"prompt": "The sky is", "completion": " blue."}
```

ហេតុអ្វីជ្រើស `messages`?

1. Trainer អនុវត្ត **chat template** របស់ម៉ូដែលឱ្យស្វ័យប្រវត្តិ — គ្មានកំហុសសរសេរ template ដៃ
2. គាំទ្រសន្ទនាច្រើនជុំ (multi-turn) និង `system` prompt
3. ជាទម្រង់ដែល dataset សាធារណៈភាគច្រើនលើ HF Hub ប្រើ

## ឯកសារ JSONL — មួយជួរ មួយកំណត់ត្រា

**JSONL** = JSON Lines៖ វត្ថុ JSON មួយក្នុងមួយជួរ គ្មានក្បៀសចុងជួរ គ្មាន array ធំ។

```json
{"messages": [{"role": "user", "content": "សួស្តី!"}, {"role": "assistant", "content": "សួស្តី! តើខ្ញុំអាចជួយអ្វីបាន?"}]}
{"messages": [{"role": "system", "content": "អ្នកជាជំនួយការគណិតវិទ្យា។"}, {"role": "user", "content": "២+២?"}, {"role": "assistant", "content": "២+២=៤"}]}
```

```{warning} កំហុស Unicode ដែលជួបញឹកញាប់
រក្សាទុកជា **UTF-8** ជានិច្ច ហើយកុំប្រើ `ensure_ascii=True`
ព្រោះវាបំលែងអក្សរខ្មែរទៅ `\u17a2...` ធ្វើឱ្យអានមិនចេញ។
កូដ `write_jsonl` របស់យើងប្រើ `ensure_ascii=False` រួចហើយ។
```

## ច្បាប់ផ្ទៀងផ្ទាត់របស់យើង

ម៉ូឌុល {py:mod}`qwen_ft.data` ត្រួតពិនិត្យច្បាប់ទាំងនេះលើកំណត់ត្រានីមួយៗ៖

- `messages` ជា list មិនទទេ
- role នីមួយៗ ∈ {`system`, `user`, `assistant`, `tool`}
- `content` ជា string មិនទទេ (ច្រានចោលចន្លោះទទេសុទ្ធ)
- មាន `system` បានតែមួយ ហើយត្រូវនៅ **ដំបូងគេ**
- សន្ទនាត្រូវមានយ៉ាងតិចមួយ `assistant` turn និង **បញ្ចប់ដោយ `assistant`**
  (បើមិនដូច្នោះទេ គ្មានអ្វីឱ្យម៉ូដែលរៀន)
- ឆ្លាស់ user/assistant គ្នាជាប់ជានិច្ច (strict alternation)
- គ្មាន key ចម្លែកក្រៅពី `messages` និង `meta`

ការប្រើប្រាស់៖

```python
from qwen_ft.data import validate_jsonl_file, split_dataset, read_jsonl

report = validate_jsonl_file("data/sample_khmer_assistant.jsonl")
print(report)
# {'path': '...', 'records': 24, 'validated': 24, 'min_turns': 2, 'max_turns': 3,
#  'avg_chars': 180.8, 'max_chars': 392}

records = read_jsonl("data/sample_khmer_assistant.jsonl")
train, eval_ = split_dataset(records, eval_ratio=0.1, seed=42)
```

ឬតាម CLI៖

```bash
qwen-ft validate-data data/sample_khmer_assistant.jsonl
```

## ត្រូវការទិន្នន័យប៉ុន្មាន?

| គោលដៅ | ចំនួនឧទាហរណ៍ណែនាំ | ចំណាំ |
|---|---|---|
| សាកល្បងបច្ចេកទេស (ដូចក្នុងសៀវភៅ) | ២០–១០០ | គ្រប់គ្រាន់ដើម្បីឃើញការប្រែប្រួល |
| បង្រៀនរចនាបថ/ទម្រង់ឆ្លើយជាក់លាក់ | ៥០០–៥,០០០ | គុណភាពសំខាន់ជាងបរិមាណ |
| ជំនាញខ្ពស់ក្នុងវិស័យមួយ | ៥,០០០–៥០,០០០ | ត្រូវការការសម្អាតទិន្នន័យម៉ត់ចត់ |

ច្បាប់មាស៖ **ឧទាហរណ៍ល្អ ១,០០០ ប្រសើរជាងឧទាហរណ៍អន់ ១០,០០០**។
ម៉ូដែលរៀនចម្លង **អ្វីគ្រប់យ៉ាង** ក្នុង dataset — រួមទាំងកំហុសអក្ខរាវិរុទ្ធ។

### លក្ខណៈពិសេសសម្រាប់ទិន្នន័យខ្មែរ

1. **ប្រើសញ្ញាអក្សរខ្មែរ Unicode ឱ្យបានត្រឹមត្រូវ** — កុំលាយ Unicode និង legacy
   encodings (ដូចជា LIMON ឬ ABC-ZWSP)។ អត្ថបទ legacy នឹងបំផ្លាញ tokenizer។
2. **លេខខ្មែរ vs លេខអារ៉ាប់** — សម្រេចចិត្តឱ្យច្បាស់ថា dataset ប្រើបែបណា
   ហើយប្រើវាឱ្យសង្វាក់គ្នា។
3. **គិតអំពីប្រវែងជា token** — អក្សរខ្មែរមួយអាចចំណាយច្រើន token។
   សាកល្បង tokenize ឧទាហរណ៍វែងបំផុតរបស់អ្នក ដើម្បីកំណត់ `max_length`។
4. បើចង់រក្សាសមត្ថភាព **គិត (reasoning)** របស់ម៉ូដែល — ការណែនាំផ្លូវការសម្រាប់
   Qwen3.5 គឺលាយឧទាហរណ៍ reasoning-style យ៉ាងតិច **៧៥%** នៃ dataset
   (មើលជំពូក ៩)។

## ប្រភពទិន្នន័យខ្មែរសាធារណៈ

- [Hugging Face Hub](https://huggingface.co/datasets?language=km) — ស្វែងរកតាម tag `language:km`
  (ឧ. [`kimleang123/khmer-text-dataset`](https://huggingface.co/datasets/kimleang123/khmer-text-dataset))
- [awesome-khmer-language](https://github.com/seanghay/awesome-khmer-language) —
  បញ្ជីធនធានភាសាខ្មែរដ៏ធំដែលរក្សាទុកដោយសហគមន៍
- បង្កើតដោយខ្លួនឯង — សួរម៉ូដែលធំៗឱ្យជួយ draft រួច **កែសម្រួលដោយមនុស្ស** មុនហ្វឹក
  (synthetic data ដែលមិនបានត្រួតពិនិត្យអាចបង្រៀនកំហុស)

```{note}
ឯកសារ `data/sample_khmer_assistant.jsonl` ក្នុង repo នេះមាន ២៤ ឧទាហរណ៍គំរូ
សម្រាប់ការរៀន។ វាតូចណាស់ — សម្រាប់ម៉ូដែលពិត អ្នកត្រូវការទិន្នន័យច្រើនជាងនេះ។
```

## សង្ខេប

- ប្រើទម្រង់ `messages` JSONL + UTF-8 (`ensure_ascii=False`)
- ផ្ទៀងផ្ទាត់ជានិច្ចដោយ `qwen-ft validate-data` ឬ `validate_jsonl_file()`
- សន្ទនាត្រូវបញ្ចប់ដោយ assistant turn និងឆ្លាស់ user/assistant
- គុណភាព ស្ថិរភាពនិងភាពសង្វាក់គ្នា សំខាន់ជាងបរិមាណ
- `split_dataset()` បំបែក train/eval ដោយ deterministic (seed ដដែល → លទ្ធផលដដែល)

## លំហាត់

1. សរសេរឧទាហរណ៍សំណួរ-ចម្លើយខ្មែរ ១០ គូអំពីប្រធានបទដែលអ្នកចង់ហ្វឹក
   ទៅជាឯកសារ JSONL ហើយផ្ទៀងផ្ទាត់ដោយ `qwen-ft validate-data`។
2. បង្កើតឯកសារដែលមានកំហុសដោយចេតនា (role ខុស, សន្ទនាបញ្ចប់ដោយ user)
   ហើយពិនិត្យថា validator ចាប់បាន — សារកំហុសចេញមកមានប្រយោជន៍ទេ?
3. រត់ notebook {doc}`០៣ Dataset Prep </../notebooks/03_dataset_prep>`។
4. គណនា៖ បើ dataset មាន ១,០០០ ឧទាហរណ៍ និង `eval_split_ratio=0.1`
   តើ train និង eval មានប៉ុន្មាន? សាកល្បងផ្ទៀងផ្ទាត់ដោយ `split_dataset()`។

---

ជំពូកមុន៖ {doc}`៣ · ម៉ូដែល និង Tokenizer <ch03-models-and-tokenizer>` ·
ជំពូកបន្ទាប់៖ {doc}`៥ · LoRA ពីគោលការណ៍ <ch05-lora-fundamentals>`
