# មគ្គុទ្ទេសក៍ Deploy ទៅ Read the Docs (១០ ជំហាន)

ទំព័រនេះណែនាំជាភាសាខ្មែរ របៀបផ្សព្វផ្វាយសៀវភៅ «Qwen Fine-Tuning from Zero»
ទៅកាន់គេហទំព័រ [Read the Docs](https://readthedocs.org/) ដោយប្រើឯកសារ
`.readthedocs.yaml` ដែលមានស្រាប់ក្នុង repo នេះ។

```{note} ស្ថានភាពបច្ចុប្បន្ន (២០២៦-១០-២២)
គម្រោងនេះត្រូវបាន **រៀបចំរួចរាល់** សម្រាប់ Read the Docs (config + docs + tests)
ប៉ុន្តែការផ្សព្វផ្សាយត្រូវការគណនី GitHub និង Read the Docs **របស់អ្នកផ្ទាល់**។
អនុវត្តតាមជំហានខាងក្រោម — បន្ទាប់ពីជំហានទី ៦ ទើបមាន URL សាធារណៈពិត។
```

## ជំហានទី ១ — បង្កើត ឬជ្រើសរើស GitHub repository

1. ចូល [github.com](https://github.com) → ចុច **New repository** (សញ្ញា + ខាងលើស្តាំ)
2. ដាក់ឈ្មោះ ឧ. `ml-foundations`
3. សម្រាប់គេហទំព័រឯកសារសាធារណៈ → ជ្រើស **Public**

```{important} Public ឬ Private?
- **Public** — Read the Docs ឥតគិតថ្លៃ · តំណ «Open in Colab» ដំណើរការ
- **Private** — ត្រូវការ RTD plan សម្រាក់ organization ឬ «private project» feature
  ហើយតំណ Colab **នឹងមិនដំណើរការ** ទេ (Colab មិនអាចអាន repo private)
```

## ជំហានទី ២ — Commit និង push គម្រោងឯកសារ

```bash
# បើ clone រួចហើយ — គ្រាន់តែ pull
git clone https://github.com/<username>/ml-foundations.git
cd ml-foundations

# បើ repo ទទេ — បញ្ចូលឯកសារ រួច៖
git add .
git commit -m "docs: Qwen Fine-Tuning from Zero — initial release"
git branch -M main
git remote add origin https://github.com/<username>/ml-foundations.git
git push -u origin main
```

ផ្ទៀងផ្ទាត់ថាឯកសារសំខាន់ៗមានលើ GitHub៖ `.readthedocs.yaml` (នៅ **root**),
`docs/conf.py`, `docs/index.md`, `docs/notebooks/*.ipynb`។

## ជំហានទី ៣ — ចុះឈ្មោះ/ចូល Read the Docs

1. ទៅ [readthedocs.org](https://readthedocs.org/) → **Sign up**
2. ជ្រើស **«Sign up with GitHub»** — ងាយបំផុត (OAuth ផ្តល់សិទ្ធិ RTD អាន repo)
3. អនុញ្ញាតឱ្យ RTD ចូលដំណើរការគណនី GitHub របស់អ្នក

## ជំហានទី ៤ — Import repository ដែលត្រឹមត្រូវ

1. Dashboard → **Import a Project**
2. ស្វែងរក `ml-foundations` ក្នុងបញ្ជី (RTD រកឃើញដោយស្វ័យប្រវត្តិតាំ OAuth)
3. ចុច **+** ឬ **Import**
4. បើមិនឃើញ repo → ស្វែងរកតាម URL ផ្ទាល់ ឬត្រួតពិនិត្យសិទ្ធិ GitHub App
   (Settings → Applications → Read the Docs)

## ជំហានទី ៥ — ផ្ទៀងផ្ទាត់ configuration

RTD អាន `.readthedocs.yaml` ពី repo root ដោយស្វ័យប្រវត្តិ។ ត្រួតពិនិត្យ៖

```yaml
version: 2                      # schema បច្ចុប្បន្ន
build:
  os: ubuntu-26.04
  tools:
    python: "3.13"
sphinx:
  configuration: docs/conf.py   # path ធៀបនឹង repo root
  fail_on_warning: true         # កំហុស reference តូចមួយ = build បរាជ័យ
python:
  install:
    - requirements: docs/requirements.txt
```

ចំណុចសំខាន់៖

- គ្រប់ **path ក្នុង `.readthedocs.yaml` ធៀបនឹង repo root** ជានិច្ច
  (មិនមែនធៀបនឹងទីតាំងឯកសារ yaml ទេ)
- `docs/requirements.txt` មានតែ sphinx/myst — **គ្មាន torch/transformers**
  ព្រោះ notebooks ត្រូវបាន render ពី source (`nb_execution_mode = "off"`)
  → build មិនដែលហ្វឹកម៉ូដែល ឬទាញយក weights
- repo នេះមាន test `tests/test_readthedocs_yaml.py` ផ្ទៀងផ្ទាត់ schema ទាំងនេះ

## ជំហានទី ៦ — បង្កើត (build) លើកដំបូង

1. ក្នុង dashboard គម្រោង → ចុច **Build version** (version `latest`)
2. រង់ចាំ ២–៥ នាទី
3. បើស្ថានភាព **Passed** → គេហទំព័ររស់នៅ៖
   `https://<project-slug>.readthedocs.io/`
   (ឧ. បើ slug ជា `ml-foundations` → `https://ml-foundations.readthedocs.io`)

```{warning} កុំប្រកាស URL មុនពេល build ឆ្លងផុត!
URL មានតែពេលដែល build ដំបូង **ជោគជ័យ** ប៉ុណ្ណោះ។
```

## ជំហានទី ៧ — អាន build logs និង diagnose កំហុស

ពេល build បរាជ័យ៖

1. Dashboard → **Builds** → ចុច build ដែលបរាជ័យ → អាន log ពី **ក្រោមឡើងលើ**
2. កំហុសទូទៅ និងដំណោះស្រាយ៖

| សារកំហុស | មូលហេតុ | ដំណោះស្រាយ |
|---|---|---|
| `Your .readthedocs.yaml version is unsupported` | ប្រើ v1/ខ្វះ `version: 2` | ដាក់ `version: 2` |
| `Configuration file not found` | yaml មិននៅ repo root | ផ្លាស់ទី `.readthedocs.yaml` ទៅ root |
| `ModuleNotFoundError: myst_nb` | ខ្វះ dependency | បន្ថែមទៅ `docs/requirements.txt` |
| `WARNING: document isn't included in any toctree` + fail_on_warning | ទំព័រកំព្រា មិនបានបញ្ចូលក្នុង toctree | បន្ថែមឯកសារទៅ toctree ក្នុង `docs/index.md` |
| `Python version not supported` | tools.python ចាស់/ខុស | ប្រើ `"3.13"` |

3. កែ → `git push` → RTD build ស្វ័យប្រវត្តិម្តងទៀត

## ជំហានទី ៨ — កំណែឯកសារ (versioned docs)

1. បង្កើត tag ក្នុង git៖

```bash
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin v1.0.0
```

2. RTD → **Versions** → activate version `v1.0.0`
3. ឥឡូវមានពីរកំណែ៖
   - `latest` → កូដចុងក្រោយលើ branch ដែលកំណត់
   - `v1.0.0` → snapshot នៃ tag → `https://<slug>.readthedocs.io/en/v1.0.0/`
4. Settings → **Default version** កំណត់ថាតើអ្នកអានឃើញកំណែណាជា default

## ជំហានទី ៩ — ធ្វើបច្ចុប្បន្នភាពគេហទំព័រក្រោយ push

ដំណើរការស្វ័យប្រវត្តិទាំងស្រុង៖

```bash
git add .
git commit -m "docs: កែជំពូក ៥"
git push
```

→ RTD ទទួល webhook → build `latest` ថ្មី → គេហទំព័រធ្វើបច្ចុប្បន្នភាពក្នុង ២–៥ នាទី។

បើ webhook មិនដំណើរការ៖ RTD dashboard → Admin → Integrations → ផ្ទៀងផ្ទាត់
GitHub webhook ឬចុច **Build version** ដោយដៃ។

## ជំហានទី ១០ — សាកល្បងតំណ notebook និង Google Colab

Notebook នីមួយៗមានប៊ូតុង Colab ចង្អុលទៅ៖

```text
https://colab.research.google.com/github/<username>/ml-foundations/blob/main/docs/notebooks/<file>.ipynb
```

checklist សាកល្បង៖

- [ ] គេហទំព័រ RTD បើក notebook ទាំង ៦ បាន (render code + markdown ត្រឹមត្រូវ
      អក្សរខ្មែរមិនខូច)
- [ ] ចុចតំណ Colab មួយ → notebook បើកក្នុង Colab ដោយមិន error
- [ ] **repo ត្រូវតែ Public** សម្រាប់ Colab (repo private → Colab មើលមិនឃើញ)
- [ ] ក្នុង Colab រត់ cell ដំបូង (install) ដោយមិនទាញយកម៉ូដែលធំ
      (notebooks របស់យើង guard កូដធ្ងន់ដោយ `QWEN_FT_ALLOW_HEAVY`)

```{tip} សាកល្បងក្នុងម៉ាស៊ីនមុន push
python -m pytest tests/test_notebooks.py -q
python -m sphinx -W --keep-going -b html docs docs/_build/html
បើពីរ command នេះឆ្លង — build លើ RTD ក៏ទំនងជាឆ្លងដែរ។
```

## សង្ខេបលំហូរទាំងមូល

```text
GitHub repo (Public)  ──push──▶  Read the Docs webhook
        │                              │
        │                        build: ubuntu-26.04 + py3.13
        │                        pip install docs/requirements.txt
        │                        sphinx -W (fail_on_warning)
        ▼                              ▼
Open in Colab ◀── public notebooks   https://<slug>.readthedocs.io ✅
```
