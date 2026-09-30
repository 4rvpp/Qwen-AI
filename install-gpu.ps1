$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first, then run .\install.ps1 followed by .\install-gpu.ps1.'
}
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    throw 'Run .\install.ps1 first to create .venv and install the application dependencies.'
}

# PyTorch packages CUDA runtime libraries in this wheel. The NVIDIA display
# driver must still be installed and current enough for CUDA 12.8.
uv pip install --python .venv\Scripts\python.exe --upgrade --reinstall --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
if ($LASTEXITCODE -ne 0) { throw 'The CUDA-enabled PyTorch installation failed.' }

& '.\.venv\Scripts\python.exe' -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable. Install/update the NVIDIA driver, then run this script again.'; arch=f'sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}'; assert arch in torch.cuda.get_arch_list(), f'The installed PyTorch wheel does not support {arch}: {torch.cuda.get_arch_list()}'; print(f'CUDA ready: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}; {arch})')"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch was installed, but Windows cannot access an NVIDIA CUDA GPU.' }

Write-Host 'GPU setup complete. Start the app with .\run.ps1.'
