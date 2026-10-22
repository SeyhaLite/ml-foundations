# Qwen Fine-Tuning from Zero · ការ Fine-Tune ម៉ូដែល Qwen ពីសូន្យ

សៀវភៅអនឡាញ + ឧបករណ៍ **ភាសាខ្មែរ** សម្រាប់រៀន fine-tune ម៉ូដែល LLM ត្រកូល
**Qwen** (Qwen3 · Qwen3.5 · Qwen3.6 · Qwen3.8) ដោយប្រើ **LoRA** ជាមួយ
`transformers` · `peft` · `trl` — ពីការដំឡើងបរិស្ថាន រហូតដល់ deployment។

> 📖 **ឯកសារពេញលេញ៖** បើកថត `docs/` ជាមួយ Sphinx (មើលខាងក្រោម) ឬអាន
> `docs/index.md` ផ្ទាល់លើ GitHub។ ការផ្សព្វផ្សាយលើ Read the Docs ត្រូវបាន
> **រៀបចំរួចរាល់** (`.readthedocs.yaml`) — សូមមើល
> [មគ្គុទ្ទេសក៍ deployment ជាភាសាខ្មែរ](docs/deployment.md)។

## ហេតុអ្វីគម្រោងនេះ?

- ✅ មាតិកាអប់រំ **ភាសាខ្មែរ** ពេញលេញ ១០ ជំពូក + លំហាត់
- ✅ **Model IDs និង API បានផ្ទៀងផ្ទាត់** តាមប្រភពផ្លូវការ (២០២៦-១០-២២) —
  មិនមានឧទាហរណ៍ដែលចង្អុលទៅម៉ូដែលខុសជំនាន់ទេ
- ✅ **Test suite រត់ដោយគ្មាន GPU** — validation, dataset checks, notebook
  integrity, docs structure, secret scanning
- ✅ **Docs build ថោក** — notebooks render ពី source (`nb_execution_mode = "off"`)
  មិនដែលទាញយក model weights
- ✅ Notebooks ទាំង ៦ មានប៊ូតុង **Open in Colab**

## រចនាសម្ព័ន្ធ

```text
docs/            សៀវភៅ Sphinx (Khmer) — ១០ ជំពូក + compatibility + deployment
                 + notebooks/ (Jupyter notebooks ៦ មេរៀន, Colab-ready)
src/qwen_ft/     Python package — config/dataset validation, training, inference, CLI
configs/         YAML configs ដែលបានផ្ទៀងផ្ទាត់ (Qwen3-0.6B, Qwen3.5-0.8B, Qwen3.5-4B)
data/            Dataset ខ្មែរគំរូ (messages JSONL, ២៤ ឧទាហរណ៍)
tests/           Structural tests (pytest) — គ្មាន GPU ចាំបាច់
scripts/         build_docs.sh / run_tests.sh
```

## ចាប់ផ្តើមរហ័ស

```bash
git clone https://github.com/SeyhaLite/ml-foundations.git
cd ml-foundations
python3 -m venv .venv && source .venv/bin/activate

# ១. ឧបករណ៍មូលដ្ឋាន (គ្មាន torch — សម្រាប់រៀន/ផ្ទៀងផ្ទាត់)
pip install -e ".[dev]"
python -m pytest -q                 # structural tests ទាំងអស់
qwen-ft validate configs/qwen3.5-4b-lora.yaml

# ២. Stack ហ្វឹកពេញលេញ (ត្រូវការ GPU)
pip install -e ".[train]"
qwen-ft env                         # ផ្ទៀងផ្ទាត់កំណែ library
qwen-ft train configs/qwen3-0.6b-lora.yaml

# ៣. ឯកសារ Sphinx ក្នុងម៉ាស៊ីន
pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
# បើក docs/_build/html/index.html
```

## តម្រូវការកំណែ (បានផ្ទៀងផ្ទាត់ ២០២៦-១០-២២)

| ម៉ូដែល | transformers អប្បបរមា |
|---|---|
| Qwen3 (២០២៥) | 4.51 (យើងណែនាំ v5) |
| Qwen3.5 / 3.6 | **5.2** |
| Qwen3.8 | **5.8** |

មើល [docs/compatibility.md](docs/compatibility.md) សម្រាប់តារាងពេញលេញ
(VRAM, QLoRA, thinking defaults)។

## ការសាកល្បង (Testing)

```bash
python -m pytest -q                    # structural — CPU only, គ្មាន download
python -m pytest tests/test_notebooks.py -q   # notebook integrity
pytest -m heavy -q                     # smoke training ពិត (ត្រូវការ torch stack)
```

## អាជ្ញាបណ្ណ

- កូដ៖ Apache-2.0 (ដូចម៉ូដែល Qwen)
- អត្ថបទឯកសារ៖ CC-BY-SA 4.0

ម៉ូដែល Qwen3/Qwen3.5 ភាគច្រើនចេញផ្សាយក្រោម Apache 2.0 —
ពិនិត្យ LICENSE ក្នុង repo checkpoint នីមួយៗជាមុនសិន។
