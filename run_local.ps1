# RuseTect - run the API + Live Console on Windows.  Usage (PowerShell, in the repo folder):  .\run_local.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt polars

# Full account history (4.06M payments) makes the console match the benchmark.
# Point RUSETECT_HISTORY_GLOB at the IBM HI-Small transaction chunks, or leave it unset to use the bundled 100K sample.
$hist = Join-Path $env:USERPROFILE "OneDrive\Desktop\iit delhi\for_claude\aml_HI-Small_part0*.csv.gz"
if (-not $env:RUSETECT_HISTORY_GLOB -and (Test-Path $hist)) { $env:RUSETECT_HISTORY_GLOB = $hist }
if ($env:RUSETECT_HISTORY_GLOB) { Write-Host "Account history: $env:RUSETECT_HISTORY_GLOB" }
else { Write-Host "Full history not found - console uses the bundled 100K-payment sample (lower recall)." }

Start-Job -ScriptBlock { Start-Sleep 6; Start-Process "http://localhost:8001/" } | Out-Null
& .\.venv\Scripts\python.exe -m uvicorn index:app --host 127.0.0.1 --port 8001
