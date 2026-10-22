#!/usr/bin/env bash
# Run the project's structural test-suite (CPU only, no model downloads).
# Add --heavy to also run the real training smoke test (needs the train extra).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--heavy" ]; then
    python -m pytest -q -m "heavy or not heavy"
else
    python -m pytest -q -m "not heavy"
fi
