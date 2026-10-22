# ៨ · រក្សាទុក · Merge · Deploy

```{contents} ក្នុងជំពូកនេះ
:local: true
:depth: 2
```

## គោលបំណងនៃជំពូក

- យល់ពីអ្វីដែលត្រូវបានរក្សាទុក (adapter vs ម៉ូដែលពេញលេញ)
- **Merge** adapter ចូលទម្ងន់ base → ម៉ូដែលឯករាជ្យមួយ
- បញ្ជូនទៅ **Hugging Face Hub**
- Export ទៅ **GGUF** សម្រាប់ Ollama/llama.cpp
- Deploy ជាមួយ **vLLM** សម្រាប់ production

## Adapter vs ម៉ូដែលពេញលេញ

| | Adapter (LoRA) | ម៉ូដែល merge រួច |
|---|---|---|
| ទំហំ | ~១០–១០០ MB | ទំហំពេញ (4B ≈ 8 GB bf16) |
| ត្រូវការ base model ពេលផ្ទុក? | ✅ បាទ/ចាស | ❌ ទេ |
| ល្បឿន inference | យឺតជាងបន្តិច (គណនា BA បន្ថែម) | លឿនពេញលេញ |
| ប្តូរជំនាញលឿន (multi-adapter)? | ✅ | ❌ |
| Deploy ទៅ vLLM/GGUF ងាយ? | ⚠️ ត្រូវ merge ជាមុនសិន | ✅ |

## ១. Merge adapter ចូល base

```python
from qwen_ft.inference import merge_adapter

merge_adapter(
    model_id_or_path="Qwen/Qwen3.5-4B",
    adapter_path="outputs/qwen3.5-4b-khmer/adapter",
    out_dir="outputs/qwen3.5-4b-khmer-merged",
    dtype="bfloat16",
)
```

ខាងក្នុងវាធ្វើ៖

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "outputs/qwen3.5-4b-khmer/adapter")
merged = model.merge_and_unload()          # W' = W + (alpha/r)·BA រួចដក LoRA layers ចេញ
merged.save_pretrained("outputs/qwen3.5-4b-khmer-merged", safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained("outputs/qwen3.5-4b-khmer/adapter")
tokenizer.save_pretrained("outputs/qwen3.5-4b-khmer-merged")
```

ផ្ទៀងផ្ទាត់ថា merge ត្រឹមត្រូវ — សំណួរដូចគ្នា ចម្លើយត្រូវតែ **ដូចគ្នាបេះបិទ**
រវាង (base+adapter) និង merged (greedy decoding)៖

```python
# ពិនិត្យភាពស្មើគ្នាដោយ greedy — sampling ធម្មតាមានភាពចៃដន្យ
reply_a = chat(model_with_adapter, tok, msgs, enable_thinking=False, do_sample=False)
reply_b = chat(merged_model, tok, msgs, enable_thinking=False, do_sample=False)
assert reply_a == reply_b
```

```{note} Merge ត្រូវការ RAM/VRAM ស្មើម៉ូដែលពេញលេញមួយ — 4B bf16 ≈ 8 GB
(អាចធ្វើលើ CPU បាន គ្រាន់តែយឺត)។
```

## ២. បញ្ជូនទៅ Hugging Face Hub

```bash
pip install -U "huggingface_hub[cli]"
hf auth login    # ដាក់ HF token របស់អ្នក (read+write)
```

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="outputs/qwen3.5-4b-khmer/adapter",
    repo_id="SeyhaLite/qwen3.5-4b-khmer-assistant",   # ឈ្មោះរបស់អ្នក
    repo_type="model",
)
```

ឬ push adapter តូចតាម Trainer ផ្ទាល់ (`SFTConfig(push_to_hub=True, hub_model_id=...)`)។

```{warning} សុវត្ថិភាព token
កុំ hard-code HF token ក្នុងកូដ/notebook ដែល commit! ប្រើ `hf auth login`
ឬ environment variable `HF_TOKEN`។ Test suite របស់ repo នេះ (`test_no_secrets.py`)
ស្កេនរក token លេចធ្លាយរាល់ការរត់។
```

## ៣. Export ទៅ GGUF (Ollama / llama.cpp)

**GGUF** ជាទម្រង់ quantized សម្រាប់រត់លើ CPU/GPU តូចៗ។ ដំណើរការស្តង់ដារ៖

1. Merge adapter (ជំហានខាងលើ)
2. Convert ទៅ GGUF ដោយ llama.cpp៖

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt
python convert_hf_to_gguf.py ../outputs/qwen3.5-4b-khmer-merged \
    --outfile qwen35-4b-khmer-f16.gguf --outtype f16
```

3. Quantize បន្ថែម (Q4_K_M ≈ 4-bit គុណភាពល្អ)៖

```bash
./build/bin/llama-quantize qwen35-4b-khmer-f16.gguf qwen35-4b-khmer-Q4_K_M.gguf Q4_K_M
```

```{note} សម្រាប់ Qwen3.5 — ចំណាំពិសេស
Qwen3.5 ជា hybrid architecture (Gated DeltaNet) ដូច្នេះការ convert/quantize
ត្រូវការ llama.cpp កំណែថ្មីដែលគាំទ្រ `qwen3_5`។ ត្រួតពិនិត្យ release notes របស់
llama.cpp មុន។ ការណែនាំពី Unsloth៖ tensor មួយចំនួនរបស់ Qwen3.5 នឹងរងគ្រោះ
ពី quantization ទាប (Q2/Q3) — ប្រើ Q4_K_M ឡើងទៅ។ ជម្រើសងាយ៖ ប្រើ
[Unsloth Dynamic GGUFs](https://huggingface.co/collections/unsloth/qwen35)
ជាគោល រួច merge adapter លើវា។
```

### រត់ជាមួយ Ollama

បង្កើត `Modelfile`៖

```text
FROM ./qwen35-4b-khmer-Q4_K_M.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER num_ctx 8192
```

```bash
ollama create qwen-khmer -f Modelfile
ollama run qwen-khmer "សួស្តី! ណែនាំខ្លួនឯងមក"
```

## ៤. Deploy ជាមួយ vLLM (production)

[vLLM](https://docs.vllm.ai/) គាំទ្រ Qwen3.5 (កំណែថ្មី) និងផ្តល់ throughput ខ្ពស់៖

```bash
pip install -U vllm

# ម៉ូដែល merge រួច — serve តាម OpenAI-compatible API
vllm serve outputs/qwen3.5-4b-khmer-merged \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="qwen3.5-4b-khmer-merged",
    messages=[{"role": "user", "content": "តើ LoRA គឺជាអ្វី?"}],
    temperature=0.7, top_p=0.8,
    chat_template_kwargs={"enable_thinking": False},
)
print(resp.choices[0].message.content)
```

vLLM ក៏អាចផ្ទុក **base + LoRA adapter** ដោយមិន merge (`--enable-lora
--lora-modules khmer=/path/to/adapter`) — សម្រាប់ multi-tenant serving។

## សង្ខេប

- Adapter តូច/បត់បែន · merged model លឿន/ឯករាជ្យ — merge មុន deploy ភាគច្រើន
- ផ្ទៀងផ្ទាត់ merge ដោយ greedy decoding ប្រៀបធៀប base+adapter vs merged
- GGUF សម្រាប់ local (Ollama) — Qwen3.5 ត្រូវការ llama.cpp ថ្មី + Q4_K_M ឡើង
- vLLM សម្រាប់ production API · គោរព sampling ណែនាំ + `enable_thinking`
- រក្សា token មានសុវត្ថិភាព — កុំ commit credentials

## លំហាត់

1. Merge adapter របស់អ្នក (ឬ adapter គំរូពី smoke test) ហើយផ្ទៀងផ្ទាត់
   ភាពដូចគ្នានៃចម្លើយ greedy។
2. គណនាទំហំឯកសារ៖ ម៉ូដែល 4B ជា bf16 = ប៉ុន្មាន GB? ជា Q4_K_M (~4.5 bit/weight)?
3. សរសេរ `Modelfile` សម្រាប់ Ollama ដោយ system prompt ខ្មែរផ្ទាល់ខ្លួន។
4. (កម្រិតខ្ពស់) ដំឡើង vLLM ហើយវាស់ tokens/second នៃ merged model របស់អ្នក
   ធៀបនឹង base មិនបានហ្វឹក។

---

ជំពូកមុន៖ {doc}`៧ · វាយតម្លៃ និង Inference <ch07-evaluation-inference>` ·
ជំពូកបន្ទាប់៖ {doc}`៩ · ប្រធានបទកម្រិតខ្ពស់ <ch09-advanced>`
