# តារាង Compatibility — ម៉ូដែល Qwen × Library

ទំព័រនេះជាការផ្ទៀងផ្ទាត់ផ្លូវការនៃ **បន្សំម៉ូដែល/library** ដែលសៀវភៅនេះណែនាំ។
ការផ្ទៀងផ្ទាត់ធ្វើឡើងនៅ **ថ្ងៃទី ២២ ខែតុលា ឆ្នាំ ២០២៦** តាមប្រភព៖

- [Hugging Face — Qwen organization](https://huggingface.co/Qwen) (បញ្ជី checkpoint ពិតប្រាកដ តាម HF API)
- [transformers documentation — Qwen3.5](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5) (class និង model_type)
- [transformers releases លើ GitHub](https://github.com/huggingface/transformers/releases) (កំណែដែលបញ្ចូល `qwen3_5`)
- [QwenLM/Qwen3.8 GitHub](https://github.com/QwenLM/Qwen3.8) (Qwen3.5/3.6/3.8 announcements)
- [Unsloth Qwen3.5 Fine-tuning Guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) (VRAM និងការណែនាំ QLoRA)

## កំណែ library ដែលបានផ្ទៀងផ្ទាត់ (PyPI, ២០២៦-១០-២២)

| Package | កំណែចុងក្រោយ | កំណែអប្បបរមាសម្រាប់គម្រោងនេះ | ចំណាំ |
|---|---|---|---|
| transformers | 5.20.2 | **≥ 5.2** (Qwen3.5/3.6) · **≥ 5.8** (Qwen3.8) · ≥ 4.51 (Qwen3 ឆ្នាំ២០២៥) | `qwen3_5` model code បង្ហាញខ្លួនដំបូងក្នុង v5.2.0 — v4.x **មិន**អាចផ្ទុក Qwen3.5 |
| peft | 0.21.2 | ≥ 0.17 | `LoraConfig(target_modules="all-linear")` |
| trl | 1.16.0 | ≥ 1.0 | API ថ្មី `SFTTrainer(model=..., args=SFTConfig(...))` |
| accelerate | 1.15.0 | ≥ 1.0 | |
| datasets | 5.1.0 | ≥ 3.0 | |
| torch | 2.14.1 | ≥ 2.6 | |
| bitsandbytes | 0.50.2 | ≥ 0.45 | **មិនណែនាំ** សម្រាប់ QLoRA លើ Qwen3.5/3.6/3.8 |

## ម៉ូដែលដែលបានផ្ទៀងផ្ទាត់

| Model ID | ជំនាន់ | ប្រភេទ | Context | transformers អប្បបរមា | LoRA bf16 VRAM (ប្រហាក់ប្រហែល) | QLoRA 4-bit |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | Qwen3 (២០២៥) | dense | 40,960 | 4.51 | ~2 GB | ✅ អាចធ្វើបាន |
| `Qwen/Qwen3-1.7B` | Qwen3 | dense | 40,960 | 4.51 | ~5 GB | ✅ |
| `Qwen/Qwen3-4B` | Qwen3 | dense | 40,960 | 4.51 | ~9 GB | ✅ |
| `Qwen/Qwen3-8B` | Qwen3 | dense | 40,960 | 4.51 | ~18 GB | ✅ |
| `Qwen/Qwen3-4B-Instruct-2507` | Qwen3 (កក្កដា ២០២៥) | dense | 262,144 | 4.51 | ~9 GB | ✅ |
| `Qwen/Qwen3-30B-A3B` | Qwen3 | MoE | 40,960 | 4.51 | ត្រូវការ multi-GPU | ⚠️ មានកម្រិត |
| `Qwen/Qwen3.5-0.8B` | Qwen3.5 (កុម្ភៈ ២០២៦) | dense + vision | 262,144 | **5.2** | ~3 GB | ❌ មិនណែនាំ |
| `Qwen/Qwen3.5-2B` | Qwen3.5 | dense + vision | 262,144 | **5.2** | ~5 GB | ❌ |
| `Qwen/Qwen3.5-4B` | Qwen3.5 | dense + vision | 262,144 | **5.2** | ~10 GB | ❌ |
| `Qwen/Qwen3.5-9B` | Qwen3.5 | dense + vision | 262,144 | **5.2** | ~22 GB | ❌ |
| `Qwen/Qwen3.5-27B` | Qwen3.5 | dense + vision | 262,144 | **5.2** | ~56 GB | ❌ |
| `Qwen/Qwen3.5-35B-A3B` | Qwen3.5 | MoE + vision | 262,144 | **5.2** | ~74 GB (A100) | ❌ |
| `Qwen/Qwen3.5-122B-A10B` | Qwen3.5 | MoE + vision | 262,144 | **5.2** | ~256 GB (multi-GPU) | ❌ |
| `Qwen/Qwen3.5-397B-A17B` | Qwen3.5 | MoE flagship | 262,144 | **5.2** | — (សម្រាប់យោងប៉ុណ្ណោះ) | ❌ |
| `Qwen/Qwen3.6-27B` | Qwen3.6 (២០២៦) | dense + vision | 262,144 | **5.2** | — | ❌ |
| `Qwen/Qwen3.6-35B-A3B` | Qwen3.6 | MoE + vision | 262,144 | **5.2** | — | ❌ |
| `Qwen/Qwen3.8-27B` | Qwen3.8 (សីហា ២០២៦) | dense + vision | 262,144 | **5.8** | — | ❌ |
| `Qwen/Qwen3.8-Flash-Next` | Qwen3.8 | dense + vision | 262,144 | **5.8** | — | ❌ |
| `Qwen/Qwen3.8-2.4T-A95B` | Qwen3.8 | MoE flagship | 262,144 | **5.8** | — (សម្រាប់យោងប៉ុណ្ណោះ) | ❌ |

ព័ត៌មានលម្អិតបច្ចេកទេសសំខាន់ៗ៖

- **Qwen3.5/3.6/3.8 ទាំងអស់** ប្រើ `model_type: qwen3_5` (ឬ `qwen3_5_moe`) និង architecture
  `Qwen3_5ForConditionalGeneration` — ពោលគឺជា **ម៉ូដែល multimodal** (មាន vision encoder
  នៅក្នុង checkpoint តែមួយ)។ ការផ្ទុកជា text-only ធ្វើតាម `AutoModelForCausalLM`
  (transformers v5 map ទៅ backbone `Qwen3_5ForCausalLM`)។
- **Qwen3.5 ប្រើ hybrid attention ៣:១** — ស្រទាប់ Gated DeltaNet (linear attention) ៣
  សម្រាប់ Gated Attention (full attention) ១។ Kernel លឿនត្រូវការ package ស្រាល
  `causal_conv1d` និង `fla`; បើអត់ វាដំណើរការបានដែរ តែយឺតជាង (fallback PyTorch)។
- **Thinking mode**៖ Qwen3.5 Small (0.8B–9B) បិទ thinking ជា default —
  ចង់បើកត្រូវបញ្ជូន `enable_thinking=True` ទៅ `apply_chat_template`។ ម៉ូដែលធំ (27B+) បើកជា default។
- **Qwen3.8** បន្ថែមការគ្រប់គ្រង `reasoning_effort` និង `preserve_thinking` តាម chat template។

## វិសាលភាព និងដែនកំណត់នៃការផ្ទៀងផ្ទាត់

ការផ្ទៀងផ្ទាត់ក្នុងការចេញផ្សាយ v1.0.0 នេះគ្របដណ្តប់៖

1. ✅ **Model IDs ពិត** — ផ្ទៀងផ្ទាត់តាម HF API (មិនមាន ID បំភាន់ទេ)
2. ✅ **API/class names** — ផ្ទៀងផ្ទាត់តាមឯកសារ transformers v5.20 និង trl v1.16
3. ✅ **កំណែ library អប្បបរមា** — ផ្ទៀងផ្ទាត់តាមការប្រៀបធៀប source tree របស់ transformers
   តាម tag (v5.1.0 គ្មាន `qwen3_5`; v5.2.0 មាន)
4. ⚠️ **ការហ្វឹកពិតលើ GPU** — មិនបានរួមបញ្ចូលក្នុង CI របស់គម្រោងនេះទេ
   (structural tests ទាំងអស់រត់ដោយគ្មាន GPU)។ ទោះជាយ៉ាងណា **smoke test ហ្វឹកពិត**
   (`pytest -m heavy`) បានឆ្លងលើ CPU ជាមួយ stack ពេញលេញ (torch 2.14.1 ·
   transformers 5.20.2 · peft 0.21.2 · trl 1.16.0) ដោយប្រើម៉ូដែល random តូច
   Qwen3-architecture — គ្មានការ download ម៉ូដែល ឬការហ្វឹកលើ GPU ពិតនៅឡើយទេ។

```{warning}
កុំច្រឡំជំនាន់៖ ឧទាហរណ៍កូដដែលប្រើ `Qwen/Qwen3.5-4B` ត្រូវតែដំណើរការលើ
transformers **v5**។ បើអ្នកឃើញកំហុស `KeyError: 'qwen3_5'` ឬ
`does not recognize this architecture` — មានន័យថា transformers របស់អ្នកនៅ v4 ឬ v5.0/5.1។
```
