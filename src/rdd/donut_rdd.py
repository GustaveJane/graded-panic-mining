import pandas as pd
from src.rdd.rdd_main import fit_local_linear_rdd
from src.config import cfg

def run_donut_rdd(df, outcome="panic_coverage", date_col="date", hole_start=None, hole_end=None, bw=14):
    d = cfg["rdd"]["donut"]
    hs = pd.Timestamp(hole_start or d["hole_start"]); he = pd.Timestamp(hole_end or d["hole_end"])
    dates = pd.to_datetime(df[date_col])
    df2 = df[~((dates >= hs) & (dates <= he))].copy()
    return fit_local_linear_rdd(y=df2[outcome].values, x=(pd.to_datetime(df2[date_col]) - pd.Timestamp(cfg["rdd"]["cutoff_date"])).dt.days.values, bandwidth=bw)
