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

"$PYTHON_BIN" - <<'PY'
import importlib
import platform

print("Python:", platform.python_version())
for name in ["torch", "torchvision", "numpy", "pandas", "matplotlib", "PIL", "yaml", "tqdm"]:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: ok", getattr(mod, "__version__", ""))
    except Exception as exc:
        print(f"{name}: missing ({exc})")

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print("GPU:", props.name)
        print("VRAM GB:", round(props.total_memory / (1024**3), 2))
except Exception as exc:
    print("Torch check failed:", exc)
PY
