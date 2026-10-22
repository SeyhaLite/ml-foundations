# Qwen Fine-Tuning from Zero — ការ Fine-Tune ម៉ូដែល Qwen ពីសូន្យ

**សៀវភៅអនឡាញភាសាខ្មែរ** សម្រាប់រៀន fine-tune ម៉ូដែលភាសាធំ (LLM) ត្រកូល **Qwen**
(Qwen3 · Qwen3.5 · Qwen3.6 · Qwen3.8) ដោយប្រើបច្ចេកទេស **LoRA** ជាមួយ
`transformers` · `peft` · `trl` — ចាប់ពីការដំឡើងបរិស្ថាន រហូតដល់ការដាក់ពង្រាយ (deploy)។

```{note}
គោលដៅ៖ អ្នកអានដែលចេះ Python ជាមូលដ្ឋាន អាចហ្វឹក (fine-tune) ម៉ូដែល Qwen
ឱ្យឆ្លើយជាភាសាខ្មែរបានល្អ ដោយប្រើ GPU តូចមួយ (ឬ Google Colab) ក្នុងរយៈពេលប៉ុន្មានម៉ោង។
```

## តើសៀវភៅនេះសម្រាប់នរណា?

- សិស្ស/និស្សិត និងអ្នកអភិវឌ្ឍន៍ខ្មែរ ដែលចង់រៀន **fine-tuning** ពីដំណាក់កាលទីមួយ
- អ្នកដែលធ្លាប់ប្រើ ChatGPT/Qwen Chat ហើយចង់ **បង្កើតម៉ូដែលផ្ទាល់ខ្លួន**
- អ្នកដែលចង់យល់ពី **LoRA, SFT, dataset preparation, deployment** ជាភាសាខ្មែរ

## មាតិកា

````{grid} 1 2 2 3
:gutter: 2

```{grid-item-card} ១ · សេចក្តីផ្តើម
:link: chapters/ch01-introduction
:link-type: doc
តើ LLM និង Qwen ជាអ្វី? ហេតុអ្វី fine-tune? ផែនទីបញ្ជីជំពូក និងលក្ខខណ្ឌត្រៀម។
```

```{grid-item-card} ២ · ដំឡើងបរិស្ថាន
:link: chapters/ch02-setup-environment
:link-type: doc
Python, venv, PyTorch/CUDA, Google Colab និងការផ្ទៀងផ្ទាត់ការដំឡើង។
```

```{grid-item-card} ៣ · ម៉ូដែល និង Tokenizer
:link: chapters/ch03-models-and-tokenizer
:link-type: doc
របៀបផ្ទុក Qwen3/Qwen3.5, chat template, `enable_thinking` និងតម្រូវការ transformers។
```

```{grid-item-card} ៤ · ត្រៀមទិន្នន័យ
:link: chapters/ch04-datasets
:link-type: doc
ទម្រង់ `messages` JSONL, ការផ្ទៀងផ្ទាត់ និងបំបែក train/eval សម្រាប់ភាសាខ្មែរ។
```

```{grid-item-card} ៥ · LoRA ពីគោលការណ៍
:link: chapters/ch05-lora-fundamentals
:link-type: doc
LoRA ដំណើរការយ៉ាងដូចម្តេច? ប៉ារ៉ាម៉ែត្រ r, alpha, target_modules និង VRAM។
```

```{grid-item-card} ៦ · ហ្វឹក SFT
:link: chapters/ch06-training-sft
:link-type: doc
SFTTrainer/SFTConfig, hyperparameters, ការតាមដាន loss និង checkpoint។
```

```{grid-item-card} ៧ · វាយតម្លៃ និង Inference
:link: chapters/ch07-evaluation-inference
:link-type: doc
sampling ត្រឹមត្រូវតាមក្រុមហ៊ុន Qwen, thinking/non-thinking និងការវាយតម្លៃ។
```

```{grid-item-card} ៨ · រក្សាទុក · Merge · Deploy
:link: chapters/ch08-save-merge-deploy
:link-type: doc
save/reload adapter, merge ទម្ងន់, GGUF, vLLM, Ollama និង HF Hub។
```

```{grid-item-card} ៩ · ប្រធានបទកម្រិតខ្ពស់
:link: chapters/ch09-advanced
:link-type: doc
QLoRA, MoE, multimodal, DPO/GRPO, Unsloth និង Qwen3.6/3.8។
```

```{grid-item-card} ១០ · Troubleshooting
:link: chapters/ch10-troubleshooting
:link-type: doc
កំហុសដែលជួបញឹកញាប់ (OOM, version, kernel) និងតារាង VRAM ពេញលេញ។
```

```{grid-item-card} តារាង Compatibility
:link: compatibility
:link-type: doc
ម៉ូដែល Qwen ណាខ្លះដំណើរការជាមួយ library កំណែណា — ផ្ទៀងផ្ទាត់ ២០២៦-១០-២២។
```

```{grid-item-card} ការ Deploy ទៅ Read the Docs
:link: deployment
:link-type: doc
មគ្គុទ្ទេសក៍ខ្មែរ ១០ ជំហាន ដើម្បីផ្សព្វផ្សាយសៀវភៅនេះលើ Read the Docs។
```
````

## Notebooks (អនុវត្តផ្ទាល់)

````{grid} 1 2 2 2
:gutter: 2

```{grid-item-card} 📓 ០១ Tokenizer
:link: notebooks/01_hello_qwen_tokenizer
:link-type: doc
ស្គាល់ tokenizer របស់ Qwen និងការបំបែកពាក្យខ្មែរ។
```

```{grid-item-card} 📓 ០២ Chat Templates
:link: notebooks/02_chat_templates
:link-type: doc
chat template, `enable_thinking` និងទម្រង់សារ។
```

```{grid-item-card} 📓 ០៣ Dataset Prep
:link: notebooks/03_dataset_prep
:link-type: doc
បង្កើត + ផ្ទៀងផ្ទាត់ dataset JSONL ដោយខ្លួនឯង។
```

```{grid-item-card} 📓 ០៤ LoRA Config Lab
:link: notebooks/04_lora_config_lab
:link-type: doc
ពិសោធន៍ជាមួយ r/alpha/dropout និងការគណនាចំនួនប៉ារ៉ាម៉ែត្រ។
```

```{grid-item-card} 📓 ០៥ Smoke Training
:link: notebooks/05_smoke_training
:link-type: doc
ហ្វឹកសាកល្បងខ្លីៗ (ត្រូវការ GPU) និងការរក្សាទុក adapter។
```

```{grid-item-card} 📓 ០៦ Inference & Export
:link: notebooks/06_inference_and_export
:link-type: doc
ផ្ទុក adapter, សន្ទនា, merge និង export។
```
````

```{toctree}
:hidden:
:maxdepth: 2

chapters/ch01-introduction
chapters/ch02-setup-environment
chapters/ch03-models-and-tokenizer
chapters/ch04-datasets
chapters/ch05-lora-fundamentals
chapters/ch06-training-sft
chapters/ch07-evaluation-inference
chapters/ch08-save-merge-deploy
chapters/ch09-advanced
chapters/ch10-troubleshooting
notebooks/01_hello_qwen_tokenizer
notebooks/02_chat_templates
notebooks/03_dataset_prep
notebooks/04_lora_config_lab
notebooks/05_smoke_training
notebooks/06_inference_and_export
compatibility
deployment
references
```
