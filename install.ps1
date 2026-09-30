$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ or use the manual Python instructions in README.md.'
}
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.cache\python'
uv python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
uv venv --python 3.12 .venv
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'Installed. Run .\run.ps1; the first transcription downloads the model.'
