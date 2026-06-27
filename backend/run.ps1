# Launch the app using THIS project's venv Python, ignoring any active conda/base
# environment. Run from anywhere:  .\run.ps1   (add args, e.g. .\run.ps1 --port 8001)
$ErrorActionPreference = "Stop"
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Error "venv not found at $venvPy. Create it first: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}
Write-Host "Using $venvPy" -ForegroundColor Cyan
& $venvPy -m uvicorn app.main:app --reload @args
