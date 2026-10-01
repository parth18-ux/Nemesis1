# RuseTect: run the API + Live Console on Windows.
# Usage (PowerShell, in the repo folder):
#   .\run_local.ps1                                   # bundled 100K-payment history sample
#   .\run_local.ps1 -HistoryGlob "D:\data\aml_HI-Small_part0*.csv.gz"   # full IBM HI-Small history (benchmark-level recall)
# You can also set $env:RUSETECT_HISTORY_GLOB instead of passing -HistoryGlob.
param(
    [string]$HistoryGlob = $env:RUSETECT_HISTORY_GLOB,
    [int]$Port = 8001
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt polars

# Legacy variable name from earlier versions is still honoured.
if (-not $HistoryGlob -and $env:RUSETECT_HISTORY_GLOB) { $HistoryGlob = $env:RUSETECT_HISTORY_GLOB }
if ($HistoryGlob) {
    $env:RUSETECT_HISTORY_GLOB = $HistoryGlob
    Write-Host "Account history: $HistoryGlob"
} else {
    Write-Host "No history glob set - the console uses the bundled 100K-payment sample (lower recall)."
}

Start-Job -ScriptBlock { param($p) Start-Sleep 6; Start-Process "http://localhost:$p/" } -ArgumentList $Port | Out-Null
& .\.venv\Scripts\python.exe -m uvicorn index:app --host 127.0.0.1 --port $Port
