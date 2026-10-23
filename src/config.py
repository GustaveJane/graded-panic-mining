"""
Central configuration loader.
Loads config/config.yaml and provides typed access to all paths and hyperparameters.
Usage:
    from src.config import cfg
    cfg["paths"]["raw_data_dir"]
    cfg["classifier"]["max_length"]
"""
import os
from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

def _load_config():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Singleton
cfg = _load_config()
PROJECT_ROOT = _PROJECT_ROOT

def resolve(path_key: str) -> Path:
    """Resolve a config path relative to project root."""
    return _PROJECT_ROOT / cfg["paths"][path_key]

def get(key: str, default=None):
    """Deep-get with dot notation, e.g. get("classifier.max_length")."""
    parts = key.split(".")
    val = cfg
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
            if val is None:
                return default
        else:
            return default
    return val
