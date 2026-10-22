#!/usr/bin/env bash
# Build the Sphinx documentation locally with the same rigor as Read the Docs:
#   * -W turns every warning into an error (RTD: fail_on_warning: true)
#   * doctrees are kept in a separate dir (RTD does the same; embedding them
#     inside the epub outdir produces spurious "unknown mimetype" warnings)
#   * builds HTML + EPUB, matching .readthedocs.yaml `formats: [epub]`
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -q -r docs/requirements.txt
python -m sphinx -W --keep-going -b html -d docs/_build/doctrees docs docs/_build/html
python -m sphinx -W --keep-going -b epub -d docs/_build/doctrees docs docs/_build/epub
echo ""
echo "✅ HTML → docs/_build/html/index.html"
echo "✅ EPUB → docs/_build/epub/*.epub"
