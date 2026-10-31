# Forge LLM - CPU-only developer setup for Windows (PowerShell 5.1+).
# Creates .\.venv, installs CPU-only torch and the package in editable mode with dev extras.
#   powershell -ExecutionPolicy Bypass -File scripts\dev_setup_cpu.ps1
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$py = $env:PYTHON
if (-not $py) { $py = "python" }

& $py -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python >= 3.10 is required" }

if (-not (Test-Path ".venv")) {
    & $py -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
$vpy = Join-Path $Root ".venv\Scripts\python.exe"

& $vpy -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "torch (cpu) install failed" }
& $vpy -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "editable install failed" }

Write-Host ""
Write-Host "CPU dev environment ready. Run the tests with:"
Write-Host "  .\.venv\Scripts\python.exe -m pytest tests -q"
Write-Host "GPU code is NOT exercised on this machine (tests_gpu/ auto-skips without CUDA)."
