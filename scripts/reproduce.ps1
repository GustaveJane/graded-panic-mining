<#
.SYNOPSIS
    Reproduce LLM-ABM Simulation Pipeline (PowerShell)
.DESCRIPTION
    One-click script to run the full simulation experiment.
    Requires DEEPSEEK_API_KEY environment variable.
#>

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  G-PACE / PEMF Simulation Reproduce   " -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# ---- Prerequisites ----
if (-not $env:DEEPSEEK_API_KEY) {
    Write-Host "[WARN] DEEPSEEK_API_KEY not set. LLM conditions (B, C) will be skipped." -ForegroundColor Yellow
}

# ---- Activate venv ----
$venv = Join-Path $ROOT ".venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Gray
    . $venv
}

# ---- Phase 1: LLM Validation ----
Write-Host "`n[Phase 1] LLM Panic Validation" -ForegroundColor Green
python -m src.simulation.pipeline --phase1 --data-path "data/processed/panic_stratified.csv"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Phase 1 skipped or failed (may be expected if no data)" -ForegroundColor Yellow
}

# ---- Phase 2+3: Full Experiment ----
Write-Host "`n[Phase 2+3] Three-Condition Counterfactual Experiment" -ForegroundColor Green
python -m src.simulation.pipeline --phase2
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Experiment failed!" -ForegroundColor Red
    exit 1
}

# ---- Results ----
$outDir = Join-Path $ROOT "data\simulation\outputs"
if (Test-Path $outDir) {
    Write-Host "`n[Results]" -ForegroundColor Green
    Get-ChildItem $outDir | ForEach-Object { Write-Host "  $($_.Name)" }
}
Write-Host "`nDone." -ForegroundColor Cyan
