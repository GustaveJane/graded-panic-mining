# -*- coding: utf-8 -*-
"""
常态化活跃用户面板：窗口期活跃 且 疫情爆发前有发帖记录

筛选条件（默认）：窗口期 >=10 帖 且 Baseline >=1 帖
  - 窗口期 >=10 帖：保证有足够轨迹可追踪
  - Baseline >=1 帖：证明用户在疫情爆发前（2019-12-02~2020-01-19）就发过帖，
    剔除纯"疫情临时用户"

输入：user_counts_three_windows.csv（stageB 产物，含每用户 win/base/pre 计数）
输出：normal_user_panel.csv（这些用户的全部帖子，含 window 标记）
"""
import argparse
import os
import pandas as pd

MAIN_CSV = os.environ.get("MAIN_CSV", "/root/autodl-tmp/full_gmm_stratified.csv")
OUT_DIR = os.environ.get("OUT_DIR", "/root/autodl-tmp/active_panel")
os.makedirs(OUT_DIR, exist_ok=True)

CNT_CSV = os.path.join(OUT_DIR, "user_counts_three_windows.csv")
USECOLS = ["user_id", "created_at", "panic_prob", "risk_group_name"]
CHUNK = 2_000_000
ENCODING = "utf-8-sig"


def assign_window(ts):
    if pd.isna(ts):
        return None
    if ts < pd.Timestamp("2020-01-20"):
        return "Baseline"
    if ts < pd.Timestamp("2020-01-23"):
        return "Signal"
    if ts < pd.Timestamp("2020-01-23 10:00"):
        return "Announcement"
    if ts < pd.Timestamp("2020-01-27"):
        return "Implementation"
    if ts <= pd.Timestamp("2020-02-10 23:59:59"):
        return "Stable"
    return "Outside"


def main(min_win, min_base):
    print(f"reading user counts from {CNT_CSV}", flush=True)
    cnt = pd.read_csv(CNT_CSV, index_col="user_id")
    mask = (cnt["win"] >= min_win) & (cnt["base"] >= min_base)
    active = set(cnt[mask].index)
    print(f"active users (win>={min_win} & base>={min_base}): {len(active):,}", flush=True)

    parts = []
    for i, ch in enumerate(pd.read_csv(MAIN_CSV, usecols=USECOLS,
                                       encoding=ENCODING, low_memory=False,
                                       chunksize=CHUNK)):
        sub = ch[ch["user_id"].isin(active)]
        if len(sub):
            sub = sub.copy()
            sub["created_at"] = pd.to_datetime(sub["created_at"], format="mixed",
                                               errors="coerce")
            parts.append(sub)
        if (i + 1) % 5 == 0:
            print(f"  chunk {i}", flush=True)

    panel = pd.concat(parts, ignore_index=True)
    panel["window"] = panel["created_at"].apply(assign_window)
    panel["is_in_window"] = panel["window"].isin(
        ["Baseline", "Signal", "Announcement", "Implementation", "Stable"])

    out = os.path.join(OUT_DIR, f"normal_user_panel_win{min_win}_base{min_base}.csv")
    panel.to_csv(out, index=False)
    print(f"\nsaved: {out}")
    print(f"  rows: {len(panel):,}  users: {panel['user_id'].nunique():,}")
    print(f"  in-window rows: {int(panel['is_in_window'].sum()):,}")
    print(panel["window"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-win", type=int, default=10)
    ap.add_argument("--min-base", type=int, default=1)
    a = ap.parse_args()
    main(a.min_win, a.min_base)
