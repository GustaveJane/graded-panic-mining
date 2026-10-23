import pandas as pd, numpy as np
from src.config import cfg, resolve

def compare_denominators(old_path=None, new_path=None):
    old = pd.read_csv(old_path or str(resolve("raw_data_dir") / cfg["data"]["raw_csv"]["daily_stats_old"]), parse_dates=["date"])
    new = pd.read_csv(new_path or str(resolve("raw_data_dir") / cfg["data"]["raw_csv"]["daily_stats"]), parse_dates=["date"])
    m = old.merge(new, on="date", suffixes=("_old", "_new"))
    m["old_rate"] = m["panic_posts_old"] / m["total_posts_old"] * 100
    m["correct_rate"] = m["panic_posts_new"] / m["covid_posts"]
    m["covid_share"] = m["covid_posts"] / m["total_posts_new"]
    return m

def corr_report(df):
    return {"old_corr": df["old_rate"].corr(df["covid_share"]), "correct_corr": df["correct_rate"].corr(df["covid_share"])}
