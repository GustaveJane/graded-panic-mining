# ==========================================
# PowerShell reproduce script
# ==========================================
Write-Output "=== Phase 2+3: Run Simulation ==="
python -c "from src.simulation.experiments import Experiment; e = Experiment(); e.run_all(); e.report()"
Write-Output "Done. Check data/simulation/outputs/"
