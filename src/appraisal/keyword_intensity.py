#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向A：扩展关键词信息强度分析 (FIXED v3)
=========================================
修复：合并对齐问题 + 相关系数NaN问题
VS Code右侧Plots可视化适配版
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import gc

# 全局绘图配置，解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = '/root/autodl-tmp/'
INPUT_DIR = '/root/autodl-tmp/group_event_study_v3d/'
OUTPUT_DIR = '/root/autodl-tmp/group_event_study_v3d/info_intensity_v3/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

KEYWORD_SETS = {
    'formal_policy': ['封城', '隔离', '管控', '封锁', '封闭', '限制'],
    'emotion': ['害怕', '恐惧', '担心', '焦虑', '崩溃', '完了', '怎么办', '睡不着'],
    'epidemic': ['疫情', '病毒', '传染', '感染', '确诊', '阳性', '发烧', '咳嗽'],
}

FILE_LIST = ['temp19.csv', 'temp1.csv']

# %% 1. 读取基础恐慌渗透率时序
# 读取渗透率
pen = pd.read_csv(f'{INPUT_DIR}daily_panic_penetration.csv')
pen['date'] = pd.to_datetime(pen['date_only']).dt.tz_localize(None)
pen = pen[['date', 'high_penetration_pct']].copy()

print("=" * 60)
print("方向A: 扩展关键词信息强度 (FIXED v3)")
print("=" * 60)
print(f"Penetration data: {len(pen)} days, {pen['date'].min().date()} ~ {pen['date'].max().date()}")

# %% 2. 循环三套关键词集计算、合并、相关、绘图
for set_name, keywords in KEYWORD_SETS.items():
    print(f"\n--- 关键词集: {set_name} ({', '.join(keywords[:3])}...) ---")

    intensity_records = []
    for fname in FILE_LIST:
        fp = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fp):
            continue
        print(f"  Processing {fname}...")
        for chunk in pd.read_csv(fp, chunksize=50000, on_bad_lines='skip', low_memory=False):
            chunk['date'] = pd.to_datetime(chunk['created_at'], errors='coerce')
            chunk = chunk[(chunk['date'] >= '2020-01-10') & (chunk['date'] <= '2020-01-30')]
            if len(chunk) == 0:
                continue
            chunk['covid_related'] = pd.to_numeric(chunk['covid_related'], errors='coerce').fillna(0)
            covid_chunk = chunk[chunk['covid_related'] == 1]
            if len(covid_chunk) == 0:
                continue

            # 提取日期（去掉时间部分）
            covid_chunk['date_only'] = pd.to_datetime(covid_chunk['date'].dt.date)

            # 关键词匹配
            pattern = '|'.join(keywords)
            covid_chunk['has_kw'] = covid_chunk['content'].astype(str).str.contains(pattern, na=False).astype(int)

            daily = covid_chunk.groupby('date_only').agg(
                total_posts=('content', 'size'),
                kw_count=('has_kw', 'sum')
            ).reset_index()
            intensity_records.append(daily)
            del covid_chunk, chunk
            gc.collect()

    if not intensity_records:
        print(f"  [WARN] No data for {set_name}")
        continue

    info_df = pd.concat(intensity_records, ignore_index=True)
    info_df = info_df.groupby('date_only').sum().reset_index()
    info_df['intensity_pct'] = info_df['kw_count'] / info_df['total_posts'] * 100
    info_df['date'] = pd.to_datetime(info_df['date_only']).dt.tz_localize(None)

    print(f"  Info intensity: {len(info_df)} days, {info_df['date'].min().date()} ~ {info_df['date'].max().date()}")

    # 合并（使用date列）
    merged = pen.merge(info_df[['date', 'intensity_pct']], on='date', how='inner')
    merged = merged.sort_values('date').reset_index(drop=True)

    print(f"  Merged: {len(merged)} days, {merged['date'].min().date()} ~ {merged['date'].max().date()}")
    print(
        f"  Penetration range: {merged['high_penetration_pct'].min():.4f} ~ {merged['high_penetration_pct'].max():.4f}")
    print(f"  Intensity range: {merged['intensity_pct'].min():.4f} ~ {merged['intensity_pct'].max():.4f}")

    if len(merged) < 3:
        print(f"  [WARN] Insufficient merged data")
        continue

    # 领先-滞后相关
    lags = range(-5, 6)
    best_corr = None
    best_lag = None

    for lag in lags:
        if lag > 0:
            # intensity 领先 penetration lag 天
            merged['intensity_shifted'] = merged['intensity_pct'].shift(lag)
            pair = merged[['high_penetration_pct', 'intensity_shifted']].dropna()
            if len(pair) >= 3:
                corr = pair['high_penetration_pct'].corr(pair['intensity_shifted'])
            else:
                corr = np.nan
        elif lag < 0:
            # penetration 领先 intensity |lag| 天
            merged['pen_shifted'] = merged['high_penetration_pct'].shift(-lag)
            pair = merged[['pen_shifted', 'intensity_pct']].dropna()
            if len(pair) >= 3:
                corr = pair['pen_shifted'].corr(pair['intensity_pct'])
            else:
                corr = np.nan
        else:
            # 同期
            pair = merged[['high_penetration_pct', 'intensity_pct']].dropna()
            if len(pair) >= 3:
                corr = pair['high_penetration_pct'].corr(pair['intensity_pct'])
            else:
                corr = np.nan

        if pd.notna(corr):
            print(f"    lag={lag:2d}: r={corr:.4f} (n={len(pair)})")
            if best_corr is None or abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        else:
            print(f"    lag={lag:2d}: NaN (insufficient data)")

    if best_corr is not None:
        direction = "intensity leads" if best_lag > 0 else "penetration leads" if best_lag < 0 else "same day"
        print(f"  ★ 最大相关: lag={best_lag} ({direction}), r={best_corr:.4f}")
    else:
        print(f"  [WARN] No valid correlation found")
        best_corr = 0
        best_lag = 0

    merged.to_csv(f'{OUTPUT_DIR}intensity_{set_name}.csv', index=False, encoding='utf-8-sig')

    # %% 绘图模块，每张图自动渲染至右侧Plots
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()
    event = merged[(merged['date'] >= '2020-01-15') & (merged['date'] <= '2020-01-30')]
    ax.plot(event['date'], event['high_penetration_pct'], 'o-', color='#C0392B', linewidth=2.5, markersize=8,
            label='Panic Penetration')
    ax2.plot(event['date'], event['intensity_pct'], 's--', color='#2980B9', linewidth=2.5, markersize=8,
             label=f'{set_name} Intensity')
    ax.axvline(pd.to_datetime('2020-01-20'), color='orange', linestyle='--', linewidth=2, label='Broadcast (Jan 20)')
    ax.axvline(pd.to_datetime('2020-01-23'), color='red', linestyle='--', linewidth=2, label='Lockdown (Jan 23)')
    ax.set_title(f'{set_name}: Lead-Lag max r={best_corr:.3f} @ lag={best_lag}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Panic Penetration (%)', fontsize=12, color='#C0392B')
    ax2.set_ylabel('Keyword Intensity (%)', fontsize=12, color='#2980B9')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig_intensity_{set_name}.png', dpi=300, bbox_inches='tight')
    plt.show() # 推送至VS Code右侧Plots面板
    plt.close()
    print(f"  [SAVE] fig_intensity_{set_name}.png")

print(f"\n✅ 方向A (FIXED v3) 完成! Output: {OUTPUT_DIR}")