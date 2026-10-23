#!/bin/bash
# ==========================================
# One-click reproduce: LLM Lazarus Simulation
# ==========================================
set -e

echo "=== 1. Setup ==="
pip install -q requests numpy pandas scipy matplotlib seaborn 2>/dev/null

echo "=== 2. Phase 1: LLM Validation (optional) ==="
echo "  Uncomment below after configuring DEEPSEEK_API_KEY"
echo "  python -c 'from src.simulation.pipeline import run_all; run_all(phase1=True)'"

echo "=== 3. Phase 2+3: Full Experiment ==="
echo "  python -c 'from src.simulation.pipeline import run_all; run_all()'"

echo "=== 4. Results ==="
echo "  ls -la data/simulation/outputs/"
