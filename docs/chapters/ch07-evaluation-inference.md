# ៧ · វាយតម្លៃ និង Inference

```{contents} ក្នុងជំពូកនេះ
:local: true
:depth: 2
```

## គោលបំណងនៃជំពូក

- ផ្ទុក adapter ដែលហ្វឹករួច ហើយសន្ទនាជាមួយវា
- ប្រើ sampling parameters ត្រឹមត្រូវតាមការណែនាំផ្លូវការរបស់ Qwen
- វាយតម្លៃម៉ូដែលឱ្យបាន **ជាប្រព័ន្ធ** — មិនមែនគ្រាន់តែ «សាកមើលមួយសំណួរ»
- ស្គាល់ failure modes និងវិធីកែ

## ផ្ទុក adapter ឡើងវិញ

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.5-4B"
ADAPTER = "outputs/qwen3.5-4b-khmer/adapter"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)  # tokenizer ដែលរក្សាទុកជាមួយ adapter
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()
```

ឬតាមឧបករណ៍របស់យើង៖

```python
from qwen_ft.inference import load_for_chat, chat

model, tokenizer = load_for_chat(BASE, adapter_path=ADAPTER)
reply = chat(model, tokenizer,
             [{"role": "user", "content": "ពន្យល់ LoRA ជាភាសាខ្មែរខ្លីៗ"}],
             max_new_tokens=256, enable_thinking=False)
print(reply)
```

## Sampling — តារាងផ្លូវការសម្រាប់ Qwen3.5

ការណែនាំផ្លូវការ (Qwen/Unsloth docs) ខុសគ្នារវាង **Thinking** និង **Instruct (non-thinking)**៖

| Mode | temperature | top_p | top_k | presence_penalty | repetition_penalty |
|---|---|---|---|---|---|
| Instruct — ទូទៅ | 0.7 | 0.8 | 20 | 1.5 | 1.0 (បិទ) |
| Instruct — reasoning | 1.0 | 0.95 | 20 | 1.5 | 1.0 |
| Thinking — ទូទៅ | 1.0 | 0.95 | 20 | 1.5 | 1.0 |
| Thinking — coding ជាក់លាក់ | 0.6 | 0.95 | 20 | 0.0 | 1.0 |

ច្បាប់ទូទៅ៖

- **temperature** ទាប (0.6–0.7) → ចម្លើយស្ថិរ/ដដែលៗ · ខ្ពស់ (1.0) → សម្រាប់ការគិត/ច្នៃប្រឌិត
- **top_p + top_k** → កាត់ចោល token ដែលមិនទំនង · កុំប្រើតែមួយគត់ខ្លាំងពេក (top_p=0.3 → និយាយម្តងទៀតៗ)
- **presence_penalty 1.5** ជួយកាត់បន្ថយការនិយាយដដែលៗ (repetition) ក្នុងម៉ូដែល Qwen
- បើចេញ **អក្សរចម្លែកៗ (gibberish)** → ជាធម្មតាបណ្តាលពី `max_length` ទាបពេក ឬ dtype ខុស
- ចង់បានចម្លើយ **ដូចគ្នាជានិច្ច** (deterministic) → `do_sample=False` (greedy)

## Thinking vs Non-thinking — ប្រើឱ្យត្រូវនឹងការហ្វឹក

ច្បាប់សំខាន់បំផុតនៃជំពូកនេះ៖

```{important}
**Inference ត្រូវតែដូចការហ្វឹក។** បើ dataset របស់អ្នកគ្មាន thinking block
ហើយអ្នកហ្វឹកដោយ `enable_thinking=False` → ពេលសន្ទនាក៏ត្រូវបិទ thinking ដែរ។
Qwen3.5 Small (0.8B–9B) បិទជា default ស្រាប់ — ប៉ុន្តែបើអ្នកហ្វឹកដោយឧទាហរណ៍
reasoning (មាន thinking) ត្រូវបើកវាវិញពេល inference។
```

```python
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,   # ឬ True — ត្រូវនឹងរបៀបដែលអ្នកបានហ្វឹក
)
```

## ការវាយតម្លៃជាប្រព័ន្ធ

### ១. វាយតម្លៃអត្ថន័យ (qualitative) — checklist

បង្កើតសំណួរសាកល្បង **២០–៥០** ដែលមិនមានក្នុង dataset (held-out) ហើយពិនិត្យ៖

- [ ] ភាសាខ្មែរត្រឹមត្រូវ? (អក្ខរាវិរុទ្ធ វេយ្យាករណ៍)
- [ ] ឆ្លើយត្រង់សំណួរ ឬវក់វិន?
- [ ] រក្សារចនាបថដែលបានហ្វឹក? (ឧ. ឆ្លើយខ្លី/វែង មានរបៀប)
- [ ] គ្មានការនិយាយដដែលៗ (repetition loop)?
- [ ] សមត្ថភាពចាស់នៅដដែល? (សួរជាអង់គ្លេស/គណិតវិទ្យា — ត្រួតពិនិត្យ catastrophic forgetting)

### ២. វាយតម្លៃលេខ (quantitative)

- **eval loss** លើ held-out split (មានស្រាប់ពីជំពូក ៦) — ប្រៀបធៀបរវាង run
- **Perplexity** = exp(loss) — ទាបជាង = ល្អជាង (លើ distribution ដូចគ្នា)
- **Exact match / F1** សម្រាប់កិច្ចការមានចម្លើយច្បាស់លាស់ (បកប្រែ ចម្លើយខ្លី)
- **LLM-as-judge** — សួរម៉ូដែលធំជាង (ឧ. Qwen3.5-27B) ឱ្យពិន្ទុ ១–៥ លើចម្លើយ
  ជាមួយ rubric ច្បាស់លាស់ (ភាពត្រឹមត្រូវ ភាសា ភាពពាក់ព័ន្ធ)

```{warning} កុំវាយតម្លៃលើទិន្នន័យហ្វឹក
បើសំណួរសាកល្បងមានក្នុង dataset → លទ្ធផលល្អក្លែងក្លាយ។ ដូចសិស្សដែល
ចាំសៀវភៅណែនាំដំណោះស្រាយ។ ប្រើតែ held-out split។
```

### ៣. ប្រៀបធៀបជាមួយ baseline

ជានិច្ច ប្រៀបធៀបចម្លើយ **មុនហ្វឹក** (base model) និង **ក្រោយហ្វឹក** (adapter)
លើសំណួរដូចគ្នា sampling ដូចគ្នា។ បើមិនល្អជាង → dataset/hyperparameters
មានបញ្ហា មិនមែន «ម៉ូដែលមិនអាចរៀន» ទេ។

## Failure modes និងវិធីកែ

| រោគសញ្ញា | មូលហេតុទំនង | ដំណោះស្រាយ |
|---|---|---|
| និយាយដដែលៗមិនឈប់ | repetition · sampling ខ្សោយ | presence_penalty 1.5 · max_new_tokens កំណត់ · ពិនិត្យ dataset មានឧទាហរណ៍ដដែលៗ? |
| ឆ្លើយជាអង់គ្លេសវិញ | dataset ខ្មែរតិចពេក ឬ epoch តិចពេក | បន្ថែមទិន្នន័យខ្មែរ · ហ្វឹក 2–3 epochs |
| បាត់សមត្ថភាពចាស់ | catastrophic forgetting | បន្ថយ epochs · lr · លាយទិន្នន័យទូទៅចូល ~១០–២៥% |
| ចម្លើយខ្លីពេក/កាត់ | max_new_tokens តូច · EOS token រៀនខុស | ដំឡើង max_new_tokens · ពិនិត្យ template |
| អក្សរចម្លែក (gibberish) | dtype/context សង្កត់ខ្លាំង · adapter ខូច | bf16 · max_length ≥ ពាក់កណ្តាលនៃឧទាហរណ៍វែង · ផ្ទុក adapter ឡើងវិញ |
| loss មិនធ្លាក់សោះ | lr តូច · template ខុស · completion_only_loss បិទ | lr 1e-4 · ពិនិត្យថា trainer ប្រើ chat template របស់ម៉ូដែល |

## សង្ខេប

- ផ្ទុក adapter លើ base ដូចគ្នា (`PeftModel.from_pretrained`) — tokenizer ពីថត adapter
- ប្រើ sampling តាមតារាងផ្លូវការ · presence_penalty 1.5 សម្រាប់ Instruct ទូទៅ
- thinking mode ពេល inference ត្រូវតែ **ដូច** ពេលហ្វឹក
- វាយតម្លៃ ៣ ស្រទាប់៖ checklist អត្ថន័យ + លេខ (eval loss/PPL/judge) + ប្រៀបធៀប baseline
- ជានិច្ចវាយតម្លៃលើ held-out data

## លំហាត់

1. សរសេរ rubric ៥ ពិន្ទុសម្រាប់វាយតម្លៃជំនួយការខ្មែរ (ភាពត្រឹមត្រូវ ភាសា
   ភាពពាក់ព័ន្ធ រចនាបថ សុវត្ថិភាព)។
2. បង្កើតសំណួរសាកល្បង ១០ (មិនយកពី dataset!) ហើយរៀបចំជាឯកសារ JSONL សម្រាប់ eval។
3. ពិសោធន៍ sampling៖ សំណួរដូចគ្នា ជាមួយ temperature 0.2 / 0.7 / 1.3 —
   តើចម្លើយប្រែប្រួលយ៉ាងដូចម្តេច? សរសេរការសង្កេត ៣ ចំណុច។
4. រត់ notebook {doc}`០៦ Inference & Export </../notebooks/06_inference_and_export>`។

---

ជំពូកមុន៖ {doc}`៦ · ហ្វឹក SFT <ch06-training-sft>` ·
ជំពូកបន្ទាប់៖ {doc}`៨ · រក្សាទុក · Merge · Deploy <ch08-save-merge-deploy>`
