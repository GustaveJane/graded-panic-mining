# -*- coding: utf-8 -*-
"""
远端脚本（Seetacloud / autodl-tmp）：构建活跃用户全量面板

背景
----
step5 的 sample_corrected_labels.csv 是「文档级配额抽样」，抽样时按文档独立抽取，
未保留用户完整轨迹，导致窗内 76,662 用户 / 79,829 帖（平均 1.04 帖/用户），
无法支撑平衡面板与迁移矩阵。

正确做法：先在【全量主表】上按窗口期发帖量筛出活跃用户，再取这些用户的【全部帖子】。
即 "先选人，再取全量"，而非 "先抽样，再选人"。

本脚本两阶段：
  Stage 1  扫描全量主表，统计窗口期内每用户发帖数分布，输出各阈值下的活跃用户规模
  Stage 2  按指定阈值筛出活跃用户，回主表取这些用户的全部帖子（不限于窗口期），
           输出 active_user_panel.csv

用法
----
  python stepA_active_user_panel.py --stage 1
  python stepA_active_user_panel.py --stage 2 --min-posts 10

依赖字段（full_gmm_stratified.csv）：content, panic_prob, risk_group_name,
                                    user_id, created_at
"""
import argparse
import os
import sys
import pandas as pd

MAIN_CSV = os.environ.get("MAIN_CSV", "/root/autodl-tmp/full_gmm_stratified.csv")
OUT_DIR = os.environ.get("OUT_DIR", "/root/autodl-tmp/active_panel")
os.makedirs(OUT_DIR, exist_ok=True)

# 五段窗口（与 step5 完全一致）
WIN_START = pd.Timestamp("2019-12-02 00:00:00")
WIN_END = pd.Timestamp("2020-02-10 23:59:59")

USECOLS = ["user_id", "created_at", "panic_prob", "risk_group_name", "content"]
# Stage 1 只需计数，只读两列可大幅提速（38 GB 表）
USECOLS_COUNT = ["user_id", "created_at"]
CHUNK = 2_000_000
ENCODING = "utf-8-sig"  # 主表首列带 BOM


def stage1():
    """统计窗口期内每用户发帖数分布。"""
    print(f"[Stage 1] scanning {MAIN_CSV} (only user_id + created_at)", flush=True)
    counts = {}
    total_rows = 0
    in_win_rows = 0
    n_nat = 0
    for i, ch in enumerate(pd.read_csv(MAIN_CSV, usecols=USECOLS_COUNT,
                                       encoding=ENCODING,
                                       low_memory=False, chunksize=CHUNK)):
        total_rows += len(ch)
        # 主表混用两种日期格式（"2019/12/1 0:03" 斜杠 与 "2020-01-20 12:34" 横杠），
        # 默认推断在两种格式冲突时约 16% 行被误判为 NaT。必须用 format="mixed"。
        ch["created_at"] = pd.to_datetime(ch["created_at"], format="mixed", errors="coerce")
        n_nat += int(ch["created_at"].isna().sum())
        m = ch["created_at"].between(WIN_START, WIN_END)
        in_win_rows += int(m.sum())
        vc = ch.loc[m, "user_id"].value_counts()
        for u, c in vc.items():
            counts[u] = counts.get(u, 0) + int(c)
        print(f"  chunk {i}: rows={len(ch):,} in-window={int(m.sum()):,} "
              f"cum_users={len(counts):,}", flush=True)

    s = pd.Series(counts).sort_values(ascending=False)
    s.to_csv(os.path.join(OUT_DIR, "window_user_post_counts.csv"),
             header=["n_posts_in_window"], index_label="user_id")

    print(f"\n=== 全表总行数: {total_rows:,} ===")
    print(f"=== 时间戳解析失败(NaT): {n_nat:,} ({n_nat/total_rows*100:.2f}%) ===")
    print(f"=== 窗口期内行数: {in_win_rows:,} ===")
    print(f"=== 窗口期内唯一用户: {len(s):,} ===")
    print(f"\n每用户窗口期发帖数分布:")
    print(s.describe().to_string())
    print(f"\n各阈值下的活跃用户规模:")
    for thr in (3, 5, 10, 15, 20, 30, 50):
        n = int((s >= thr).sum())
        n_posts = int(s[s >= thr].sum())
        print(f"  >= {thr:2d} 帖: 用户 {n:>9,}  占用户 {n/len(s)*100:6.3f}%  "
              f"贡献帖数 {n_posts:>10,}  占窗内 {n_posts/in_win_rows*100:6.2f}%")


def stage2(min_posts: int, with_content: bool = False):
    """筛出窗口期发帖 >= min_posts 的用户，取其全部帖子。"""
    cnt_path = os.path.join(OUT_DIR, "window_user_post_counts.csv")
    if not os.path.exists(cnt_path):
        print("!! 请先运行 --stage 1 生成用户发帖统计")
        sys.exit(1)
    s = pd.read_csv(cnt_path, index_col="user_id")["n_posts_in_window"]
    active = set(s[s >= min_posts].index)
    print(f"[Stage 2] active users (>={min_posts} posts in window): {len(active):,}", flush=True)

    # content 是表内最大字段；层级迁移分析不需要它，跳过可大幅提速
    cols = USECOLS if with_content else [c for c in USECOLS if c != "content"]
    print(f"  reading columns: {cols}", flush=True)
    n_nat = 0
    n_kept = 0
    parts = []
    for i, ch in enumerate(pd.read_csv(MAIN_CSV, usecols=cols,
                                       encoding=ENCODING,
                                       low_memory=False, chunksize=CHUNK)):
        sub = ch[ch["user_id"].isin(active)]
        if len(sub):
            # 必须逐 chunk 转换：主表 created_at 为 "2019/12/1 0:03" 格式，
            # 且不同时段写法不一。若 concat 后统一转换，pandas 只按首个非空值
            # 推断格式，会把其余时段整批变成 NaT（曾导致 460 万行时间戳丢失）。
            sub = sub.copy()
            sub["created_at"] = pd.to_datetime(sub["created_at"], format="mixed", errors="coerce")
            n_nat += int(sub["created_at"].isna().sum())
            n_kept += len(sub)
            parts.append(sub)
        print(f"  chunk {i}: kept {len(sub):,}", flush=True)

    panel = pd.concat(parts, ignore_index=True)
    print(f"  timestamp parse: {n_kept - n_nat:,}/{n_kept:,} valid, {n_nat:,} NaT", flush=True)
    if n_kept and n_nat / n_kept > 0.01:
        print(f"  !! WARNING: {n_nat/n_kept*100:.1f}% timestamps failed to parse", flush=True)

    # 补上窗口标记，便于后续与 step5 口径对齐
    def assign_window(ts):
        if pd.isna(ts):
            return None
        if ts < pd.Timestamp("2020-01-20"):
            return "Baseline"
        if ts < pd.Timestamp("2020-01-23"):
            return "Signal"
        if ts < pd.Timestamp("2020-01-23 10:00:00"):
            return "Announcement"
        if ts < pd.Timestamp("2020-01-27"):
            return "Implementation"
        if ts <= WIN_END:
            return "Stable"
        return "Outside"

    panel["window"] = panel["created_at"].apply(assign_window)
    panel["is_in_window"] = panel["window"].isin(
        ["Baseline", "Signal", "Announcement", "Implementation", "Stable"])

    out = os.path.join(OUT_DIR, f"active_user_panel_min{min_posts}.csv")
    panel.to_csv(out, index=False)
    print(f"\nsaved: {out}")
    print(f"  rows: {len(panel):,}")
    print(f"  users: {panel['user_id'].nunique():,}")
    print(f"  in-window rows: {int(panel['is_in_window'].sum()):,}")
    print(f"\n各窗口帖子数:")
    print(panel["window"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--min-posts", type=int, default=10)
    ap.add_argument("--with-content", action="store_true",
                    help="also keep the text column (much slower; only needed for topic coding)")
    a = ap.parse_args()
    if a.stage == 1:
        stage1()
    else:
        stage2(a.min_posts, a.with_content)
