# Changelog

គម្រោងនេះអនុវត្តតាម [Keep a Changelog](https://keepachangelog.com/)
និង [Semantic Versioning](https://semver.org/)។

## [1.0.0] — 2026-10-22

ការចេញផ្សាយដំបូង។

### បន្ថែម (Added)

- ឯកសារអប់រំភាសាខ្មែរ ១០ ជំពូក (Sphinx + MyST) គ្របដណ្តប់៖ សេចក្តីផ្តើម,
  ដំឡើងបរិស្ថាន, ម៉ូដែល/tokenizer, dataset, LoRA, SFT training,
  វាយតម្លៃ/inference, save/merge/deploy, ប្រធានបទកម្រិតខ្ពស់, troubleshooting
- Jupyter notebooks ៦ មេរៀន ជាមួយតំណ Open in Colab
- Python package `qwen_ft`៖
  - `models.py` — registry នៃ checkpoint Qwen ដែលបានផ្ទៀងផ្ទាត់តាម HF API
    (Qwen3, Qwen3.5, Qwen3.6, Qwen3.8) ជាមួយ transformers floor, VRAM,
    thinking defaults
  - `config.py` — YAML config validation (dataclasses, ជំហានផ្ទៀងផ្ទាត់ច្រើនស្រទាប់)
  - `data.py` — messages-JSONL dataset validation + deterministic splitting
  - `training.py` / `inference.py` — LoRA SFT pipeline (TRL 1.x SFTTrainer),
    chat/merge/export helpers (lazy imports — គ្មាន torch ក៏ដំណើរការ)
  - `cli.py` — `qwen-ft env|validate|validate-data|models|train`
- Configs ៣ សម្រាប់ Qwen3-0.6B, Qwen3.5-0.8B, Qwen3.5-4B
- Dataset ខ្មែរគំរូ ២៤ ឧទាហរណ៍ (`data/sample_khmer_assistant.jsonl`)
- Test suite (pytest)៖ config, data, models registry, notebook integrity,
  docs structure, `.readthedocs.yaml` schema, secret scanning, CLI,
  heavy smoke training (skip ដោយស្វ័យប្រវត្តិបើគ្មាន torch)
- `.readthedocs.yaml` (v2 schema, ubuntu-26.04, Python 3.13, fail_on_warning)
- មគ្គុទ្ទេសក៍ deployment ទៅ Read the Docs ជាភាសាខ្មែរ (១០ ជំហាន)
- តារាង compatibility ពេញលេញ (docs/compatibility.md)

### ការផ្ទៀងផ្ទាត់ (Verified 2026-10-22)

- Model IDs ទាំងអស់ត្រួតពិនិត្យតាម Hugging Face API (author=Qwen)
- Qwen3.5/3.6/3.8 = `model_type: qwen3_5` ផ្ទៀងផ្ទាត់ពី config.json ផ្លូវការ
- transformers floor ផ្ទៀងផ្ទាត់តាម source tree នៃ tags v5.0–v5.20
  (`qwen3_5` បង្ហាញខ្លួនដំបូងក្នុង v5.2.0)
- TRL 1.16 SFTTrainer/SFTConfig API ផ្ទៀងផ្ទាត់តាមឯកសារផ្លូវការ
- **Smoke test ហ្វឹកពិតបានឆ្លង** (172/172 tests) លើ stack ពេញលេញ៖
  torch 2.14.1+cpu · transformers 5.20.2 · peft 0.21.2 · trl 1.16.0 ·
  datasets 5.1.0 (ម៉ូដែល random តូច Qwen3-architecture — គ្មាន download)
- Sphinx build (HTML + EPUB) ឆ្លងដោយ **zero warnings** ជាមួយ `-W --keep-going`
  លើ sphinx 8.2.3 + docutils 0.21.2 + myst-parser 5.1.0 + myst-nb 1.4.0

[1.0.0]: https://github.com/SeyhaLite/ml-foundations/releases/tag/v1.0.0
