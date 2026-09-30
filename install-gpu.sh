#!/usr/bin/env bash
# Install CUDA-enabled PyTorch for this project's existing Linux virtualenv.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Create the virtual environment and install requirements first." >&2
  exit 1
fi

# The wheel includes the CUDA runtime libraries. A compatible NVIDIA driver is
# still required; the full CUDA Toolkit is not.
# Force replacement: a CPU wheel can otherwise have a newer-looking version
# number and pip may leave it installed even when the cu128 index is selected.
"$python_bin" -m pip install --upgrade --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
"$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        'CUDA is unavailable. Install/update the NVIDIA driver, then run install-gpu.sh again.'
    )
arch = f'sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}'
if arch not in torch.cuda.get_arch_list():
    raise SystemExit(f'The installed PyTorch wheel does not support {arch}: {torch.cuda.get_arch_list()}')
print(f'CUDA ready: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}; {arch})')
PY

echo 'GPU setup complete. Start the app with: .venv/bin/python app.py'
