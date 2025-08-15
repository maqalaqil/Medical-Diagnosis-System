#!/usr/bin/env bash
set -euo pipefail
# macOS zsh-compatible: run with `bash scripts/run_all.sh`

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt || true

# If optional dependencies fail (e.g., lime, shap), continue; core will still work

python scripts/generate_synthetic_data.py --rows 1000 --out data/synthetic_disease.csv
python src/train.py --data data/synthetic_disease.csv --target disease --out-dir artifacts

# Start API (foreground). Comment out to skip.
# uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

echo "Done. Artifacts in ./artifacts."
