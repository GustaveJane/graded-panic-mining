import pandas as pd
from pathlib import Path
from src.config import cfg, resolve

def load_daily_stats(version="correct"):
    key = "daily_stats" if version == "correct" else "daily_stats_old"
    return pd.read_csv(resolve("raw_data_dir") / cfg["data"]["raw_csv"][key], parse_dates=["date"])

def load_temp_files(base_dir=None):
    base = Path(base_dir or resolve("raw_data_dir"))
    dfs = [pd.read_csv(base / f"{n}.csv") for n in cfg["data"]["temp_sources"] if (base / f"{n}.csv").exists()]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
