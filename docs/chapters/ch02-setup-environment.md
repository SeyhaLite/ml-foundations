# ២ · ដំឡើងបរិស្ថាន

```{contents} ក្នុងជំពូកនេះ
:local: true
:depth: 2
```

## គោលបំណងនៃជំពូក

- ដំឡើង Python + virtual environment បានត្រឹមត្រូវ
- ដំឡើង PyTorch, transformers **v5**, peft, trl និងឧបករណ៍ `qwen-ft` របស់យើង
- ជ្រើសរើសរវាង **ម៉ាស៊ីនផ្ទាល់ (GPU)** និង **Google Colab**
- ផ្ទៀងផ្ទាត់ថាអ្វីៗដំណើរការ មុនចូលជំពូកបន្ទាប់

```{warning} ចំណុចដែលមនុស្សភាគច្រើនបរាជ័យ
Qwen3.5 ត្រូវការ **transformers >= 5.2**។ បើឯកសារចាស់ (ឬ Colab នៅឆ្នាំមុន)
ណែនាំ `transformers==4.x` — វានឹងបរាជ័យជាមួយ `KeyError: 'qwen3_5'`។
ជំពូកនេះដំឡើងកំណែត្រឹមត្រូវឱ្យអ្នករួចហើយ។
```

## ជម្រើសទី ១ — ម៉ាស៊ីនផ្ទាល់របស់អ្នក (មាន GPU)

### ជំហាន ១៖ ត្រួតពិនិត្យ Python

ត្រូវការ **Python 3.10 ឡើងទៅ** (ផ្ទៀងផ្ទាត់ ២០២៦-១០-២២៖ torch 2.14, transformers 5.20
គាំទ្រ 3.10–3.13)។

```bash
python3 --version   # ត្រូវ >= 3.10
```

### ជំហាន ២៖ បង្កើត virtual environment

```bash
git clone https://github.com/SeyhaLite/ml-foundations.git
cd ml-foundations

python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
```

ហេតុអ្វី venv? វាញែក dependency របស់គម្រោងនេះពី Python របស់ប្រព័ន្ធ —
បើខូច គ្រាន់តែលុប `.venv` ហើយបង្កើតថ្មី។

### ជំហាន ៣៖ ដំឡើង PyTorch តាម CUDA របស់អ្នក

```bash
nvidia-smi   # មើល CUDA version ជួរខាងលើស្តាំ
```

```bash
# GPU NVIDIA (CUDA 12.x) — ណែនាំ
pip install torch --index-url https://download.pytorch.org/whl/cu126

# គ្មាន GPU (CPU only — អាចរៀនបាន តែហ្វឹកយឺត)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### ជំហាន ៤៖ ដំឡើង stack ហ្វឹកម៉ូដែល + ឧបករណ៍របស់យើង

```bash
pip install -e ".[train,dev]"
```

បញ្ជីនេះដំឡើង (កំណែអប្បបរមាដែលបានផ្ទៀងផ្ទាត់)៖

| Package | អប្បបរមា | ហេតុអ្វី |
|---|---|---|
| `transformers` | **5.8** | ផ្ទុក Qwen3, Qwen3.5/3.6 **និង Qwen3.8** |
| `peft` | 0.17 | LoRA adapters |
| `trl` | 1.0 | `SFTTrainer` API ថ្មី |
| `datasets` | 3.0 | ការគ្រប់គ្រង dataset |
| `accelerate` | 1.0 | device mapping / multi-GPU |

### ជំហាន ៥៖ ផ្ទៀងផ្ទាត់

```bash
qwen-ft env
```

លទ្ធផលគំរូ៖

```text
qwen-ft 1.0.0
  torch          2.14.1
  transformers   5.20.2
  peft           0.21.2
  trl            1.16.0
  datasets       5.1.0
  accelerate     1.15.0
  bitsandbytes   NOT INSTALLED
  cuda_available True
```

```{tip}
`bitsandbytes` មានតែពេលអ្នកចង់ធ្វើ QLoRA លើ **Qwen3 (២០២៥)** ប៉ុណ្ណោះ —
សម្រាប់ Qwen3.5+ យើងណែនាំ bf16 LoRA (មើលជំពូក ៥ និង ៩)។
```

## ជម្រើសទី ២ — Google Colab (គ្មាន GPU ផ្ទាល់ខ្លួន)

Colab ផ្តល់ GPU ឥតគិតថ្លៃ (T4 16GB) — គ្រប់គ្រាន់សម្រាប់ Qwen3-0.6B
និង Qwen3.5-0.8B (bf16 LoRA ~3GB VRAM)។

```ipython3
# Cell ទី ១ នៅក្នុង Colab
!nvidia-smi                      # ត្រូវឃើញ Tesla T4 (ឬ L4)
!pip install -U "transformers>=5.8,<6" "peft>=0.17" "trl>=1.0" datasets accelerate
!git clone https://github.com/SeyhaLite/ml-foundations.git
%cd ml-foundations
!pip install -e .
!qwen-ft env
```

```{important}
Colab **កំណត់ពេលវេលា** — session ឥតគិតថ្លៃអាចដាច់ក្រោយប៉ុន្មានម៉ោង។
រក្សាទុក checkpoint ទៅ Google Drive ជាប្រចាំ (ជំពូក ៦ បង្ហាញរបៀប)។
Notebooks ទាំង ៦ របស់សៀវភៅមានប៊ូតុង «Open in Colab» នៅក្បាល notebook នីមួយៗ។
```

## ជម្រើសទី ៣ — VS Code + Dev Containers (សម្រាប់អ្នកចូលចិត្ត Docker)

បើអ្នកចង់បានបរិស្ថានដែល replicate បានពិតប្រាកដ៖

```bash
docker run -it --gpus all -v "$PWD":/workspace python:3.12-slim bash
cd /workspace && pip install -e ".[train,dev]"
```

## ការផ្ទៀងផ្ទាត់ចុងក្រោយ — smoke test គ្មាន GPU

ទោះគ្មាន GPU ក៏អ្នកអាចរត់ **structural tests** ទាំងអស់បាន៖

```bash
python -m pytest -q          # tests ទាំងអស់ (heavy test នឹងត្រូវ skip ដោយស្វ័យប្រវត្តិ)
qwen-ft validate configs/qwen3.5-4b-lora.yaml
qwen-ft validate-data data/sample_khmer_assistant.jsonl
```

បើ command ទាំងនេះចេញ «OK» — បរិស្ថានរបស់អ្នករួចរាល់សម្រាប់រៀន។

## បញ្ហាដែលជួបញឹកញាប់ក្នុងការដំឡើង

| កំហុស | មូលហេតុ | ដំណោះស្រាយ |
|---|---|---|
| `pip: command not found` | មិនបានដំឡើង pip | `python3 -m ensurepip --upgrade` |
| `CUDA not available` បន្ទាប់ពី install | ដំឡើង torch កំណែ CPU | ដំឡើងឡើងវិញតាម `--index-url .../cu126` |
| `KeyError: 'qwen3_5'` | transformers < 5.2 | `pip install -U "transformers>=5.8,<6"` |
| `externally-managed-environment` | Python ប្រព័ន្ធហាម pip ផ្ទាល់ | ប្រើ venv (ជំហាន ២) |
| Colab `ImportError` ក្រោយ install | kernel ចាស់នៅ cache | `Runtime → Restart session` ហើយ install ម្តងទៀត |

## សង្ខេប

- ប្រើ **venv** ជានិច្ច — កុំដំឡើងចូល Python ប្រព័ន្ធ
- **transformers >= 5.8** គ្របជំនាន់ Qwen ទាំងអស់ក្នុងសៀវភៅនេះ
- `qwen-ft env` និង `pytest -q` ជាវិធីផ្ទៀងផ្ទាត់រហ័ស
- គ្មាន GPU? Colab T4 គ្រប់គ្រាន់សម្រាប់ម៉ូដែលតូច (0.6B–0.8B)

## លំហាត់

1. ដំឡើងបរិស្ថានតាមជម្រើសមួយ ហើយថតអេក្រង់លទ្ធផល `qwen-ft env`។
2. រត់ `python -m pytest -q` — តើមាន test ប៉ុន្មាន pass/skip? ហេតុអ្វី heavy test skip?
3. (Colab) ភ្ជាប់ Google Drive ហើយ clone repo ទៅ Drive ដើម្បីកុំបាត់ checkpoint
   ពេល session ដាច់។

---

ជំពូកមុន៖ {doc}`១ · សេចក្តីផ្តើម <ch01-introduction>` ·
ជំពូកបន្ទាប់៖ {doc}`៣ · ម៉ូដែល និង Tokenizer <ch03-models-and-tokenizer>`
