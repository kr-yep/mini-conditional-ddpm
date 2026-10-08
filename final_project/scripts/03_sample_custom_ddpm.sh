#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="$PYTHON"
elif [ -x ".venv-wsl/bin/python" ]; then
  PYTHON_BIN=".venv-wsl/bin/python"
else
  PYTHON_BIN="python3"
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"
fi
"$PYTHON_BIN" final_project/src/sample_ddpm.py --config final_project/configs/final_experiment.yaml "$@"
