"""End-to-end pipeline orchestrator.

Usage:
    python -m src.pipeline.run_pipeline              # full pipeline
    python -m src.pipeline.run_pipeline --sim-only   # simulation only
    python -m src.pipeline.run_pipeline --check      # dry-run: import check
"""
import argparse
import logging
import sys
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

STEPS = [
    ("stage1_filter",      "src.data.filter_covid"),
    ("stage2_classify",    "src.classifier.train"),
    ("stage3_gmm",         "src.gmm.fit_gmm_full"),
    ("stage3_topic",       "src.topic.topic_model"),
    ("stage4_rdd",         "src.rdd.rdd_main"),
    ("stage4_denominator", "src.rdd.denominator"),
    ("stage5_placebo",     "src.rdd.placebo"),
]

def run_module(mod_path: str) -> bool:
    log.info("Running  %s", mod_path)
    result = subprocess.run(
        [sys.executable, "-m", mod_path],
        cwd=str(ROOT), capture_output=False,
    )
    if result.returncode == 0:
        log.info("Finished %s", mod_path)
        return True
    else:
        log.error("Failed  %s (rc=%d)", mod_path, result.returncode)
        return False

def run_all():
    log.info("=== G-PACE / PEMF Full Pipeline ===")
    ok = sum(run_module(mod) for _, mod in STEPS)
    total = len(STEPS)
    log.info("Pipeline: %d/%d stages succeeded", ok, total)
    return ok == total

def run_simulation():
    """Run the LLM-ABM simulation pipeline."""
    log.info("=== Simulation Pipeline ===")
    ok = run_module("src.simulation.pipeline")
    log.info("Simulation pipeline: %s", "OK" if ok else "FAILED")
    return ok

def check_imports():
    """Dry-run: verify all modules import cleanly."""
    modules = [mod for _, mod in STEPS] + [
        "src.simulation.pipeline",
        "src.simulation.analysis",
        "src.simulation.llm_client",
    ]
    all_ok = True
    for m in modules:
        try:
            __import__(m)
            log.info("  Import OK  %s", m)
        except Exception as e:
            log.error("  Import FAIL %s: %s", m, e)
            all_ok = False
    return all_ok

def main():
    parser = argparse.ArgumentParser(description="G-PACE Pipeline")
    parser.add_argument("--sim-only", action="store_true", help="Run simulation only")
    parser.add_argument("--check", action="store_true", help="Dry-run: verify imports")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check_imports() else 1)
    elif args.sim_only:
        sys.exit(0 if run_simulation() else 1)
    else:
        sys.exit(0 if run_all() else 1)

if __name__ == "__main__":
    main()
