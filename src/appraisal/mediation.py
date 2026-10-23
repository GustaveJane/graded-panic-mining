#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信息强度中介变量分析
=====================
计算"封城"+"武汉"关键词日频，与恐慌渗透率做领先-滞后相关+格兰杰因果
VS Code右侧Plots可视化适配版
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
import os
import re
import gc
import warnings

warnings.filterwarnings('ignore')

# 全局绘图配置，解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = '/root/autodl-tmp/'
INPUT_DIR = '/root/autodl-tmp/group_event_study_v3d/'
OUTPUT_DIR = '/root/autodl-tmp/group_event_study_v3d/info_intensity/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% 1. 读取渗透率数据
# ==================== 1. 读取v3d已输出的渗透率数据 ====================
print("=" * 60)
print("Phase 1: Loading panic penetration data")
print("=" * 60)

pen = pd.read_csv(f'{INPUT_DIR}daily_panic_penetration.csv')
pen['date'] = pd.to_datetime(pen['date_only'])
pen = pen[['date', 'high_penetration_pct', 'total_posts', 'high_posts']].copy()
print(f"Loaded: {len(pen)} days, {pen['date'].min().date()} ~ {pen['date'].max().date()}")

# %% 2. 计算信息强度关键词日频
# ==================== 2. 计算信息强度（关键词日频） ====================
print(f"\n{'=' * 60}")
print("Phase 2: Computing information intensity")
print(f"{'=' * 60}")
print("Keywords: 封城, 武汉")

KEYWORDS = ['封城', '武汉']
FILE_LIST = ['temp19.csv', 'temp1.csv']  # January files

info_intensity = []

for fname in FILE_LIST:
    fp = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fp):
        print(f"  [SKIP] {fname} not found")
        continue

    print(f"  Processing {fname}...")
    for chunk in pd.read_csv(fp, chunksize=50000, on_bad_lines='skip', low_memory=False):
        # 日期过滤
        chunk['date'] = pd.to_datetime(chunk['created_at'], errors='coerce')
        chunk = chunk[(chunk['date'] >= '2020-01-10') & (chunk['date'] <= '2020-01-30')]
        if len(chunk) == 0:
            continue

        # COVID筛选
        chunk['covid_related'] = pd.to_numeric(chunk['covid_related'], errors='coerce').fillna(0)
        covid_chunk = chunk[chunk['covid_related'] == 1].copy()
        if len(covid_chunk) == 0:
            continue

        # 提取日期
        covid_chunk['date_only'] = pd.to_datetime(covid_chunk['date'].dt.date)

        # 关键词匹配
        for kw in KEYWORDS:
            covid_chunk[f'has_{kw}'] = covid_chunk['content'].astype(str).str.contains(kw, na=False).astype(int)

        # 日级聚合
        daily = covid_chunk.groupby('date_only').agg(
            total_posts=('content', 'size'),
            **{f'{kw}_count': (f'has_{kw}', 'sum') for kw in KEYWORDS}
        ).reset_index()
        info_intensity.append(daily)

        del covid_chunk, chunk
        gc.collect()

if info_intensity:
    info_df = pd.concat(info_intensity, ignore_index=True)
    info_df = info_df.groupby('date_only').sum().reset_index()
    info_df['info_intensity_pct'] = (info_df['封城_count'] + info_df['武汉_count']) / info_df['total_posts'] * 100

    info_df.to_csv(f'{OUTPUT_DIR}daily_info_intensity.csv', index=False, encoding='utf-8-sig')
    print(f"\n[SAVE] daily_info_intensity.csv")
    print(f"  Date range: {info_df['date_only'].min()} ~ {info_df['date_only'].max()}")
    print(f"  Total COVID posts analyzed: {info_df['total_posts'].sum():,}")
    print(f"  Keyword mentions: {info_df['封城_count'].sum() + info_df['武汉_count'].sum():,}")
else:
    print("[ERROR] No data for info intensity")
    exit()

# %% 3. 数据合并与事件窗口截取
# ==================== 3. 合并数据 ====================
print(f"\n{'=' * 60}")
print("Phase 3: Merging penetration + intensity")
print(f"{'=' * 60}")

info_df['date'] = pd.to_datetime(info_df['date_only'])
merged = pen.merge(info_df, on='date', how='inner')

# 聚焦事件窗口
EVENT_START = pd.to_datetime('2020-01-15')
EVENT_END = pd.to_datetime('2020-01-30')
event_window = merged[(merged['date'] >= EVENT_START) & (merged['date'] <= EVENT_END)].copy()

print(f"Merged data: {len(merged)} days")
print(f"Event window (1.15-1.30): {len(event_window)} days")

# %% 4. 领先滞后相关计算
# ==================== 4. 领先-滞后相关分析 ====================
print(f"\n{'=' * 60}")
print("Phase 4: Lead-Lag Correlation Analysis")
print(f"{'=' * 60}")

lags = range(-5, 6)  # 信息强度领先渗透率-5到+5天
correlations = []

for lag in lags:
    if lag > 0:
        # 信息强度领先lag天
        merged_shift = merged.copy()
        merged_shift['intensity_shifted'] = merged_shift['info_intensity_pct'].shift(lag)
        corr = merged_shift[['high_penetration_pct', 'intensity_shifted']].corr().iloc[0, 1]
    elif lag < 0:
        # 信息强度滞后|lag|天（即渗透率领先）
        merged_shift = merged.copy()
        merged_shift['penetration_shifted'] = merged_shift['high_penetration_pct'].shift(-lag)
        corr = merged_shift[['penetration_shifted', 'info_intensity_pct']].corr().iloc[0, 1]
    else:
        # 同期
        corr = merged[['high_penetration_pct', 'info_intensity_pct']].corr().iloc[0, 1]

    correlations.append({'lag': lag, 'correlation': corr})
    direction = "intensity leads" if lag > 0 else "penetration leads" if lag < 0 else "contemporaneous"
    print(f"  lag={lag:2d} ({direction}): r={corr:.4f}")

corr_df = pd.DataFrame(correlations)
corr_df.to_csv(f'{OUTPUT_DIR}lead_lag_correlations.csv', index=False, encoding='utf-8-sig')
print(f"\n[SAVE] lead_lag_correlations.csv")

# 找出最大相关
max_corr = corr_df.loc[corr_df['correlation'].abs().idxmax()]
print(f"\n  Maximum correlation: lag={max_corr['lag']} (r={max_corr['correlation']:.4f})")

# %% 5. 格兰杰因果检验
# ==================== 5. 格兰杰因果检验 ====================
print(f"\n{'=' * 60}")
print("Phase 5: Granger Causality Tests")
print(f"{'=' * 60}")
print("H0: X does NOT Granger-cause Y")

# 准备VAR数据（需要平稳序列）
var_data = merged[['high_penetration_pct', 'info_intensity_pct']].dropna()

# 一阶差分（使序列平稳）
var_data_diff = var_data.diff().dropna()

print(f"\n  Testing: Information Intensity → Panic Penetration")
try:
    gc_result = grangercausalitytests(var_data_diff[['high_penetration_pct', 'info_intensity_pct']], maxlag=3,
                                      verbose=False)
    gc_summary = []
    for lag in range(1, 4):
        f_pvalue = gc_result[lag][0]['ssr_ftest'][1]
        chi2_pvalue = gc_result[lag][0]['ssr_chi2test'][1]
        gc_summary.append({'lag': lag, 'f_test_pvalue': f_pvalue, 'chi2_pvalue': chi2_pvalue})
        sig = "***" if f_pvalue < 0.01 else "**" if f_pvalue < 0.05 else "*" if f_pvalue < 0.1 else ""
        print(f"    lag={lag}: F-test p={f_pvalue:.4f}{sig}")
    gc_summary_df = pd.DataFrame(gc_summary)
    gc_summary_df.to_csv(f'{OUTPUT_DIR}granger_intensity_to_penetration.csv', index=False, encoding='utf-8-sig')
except Exception as e:
    print(f"    [ERROR] {e}")

print(f"\n  Testing: Panic Penetration → Information Intensity (reverse)")
try:
    gc_result_rev = grangercausalitytests(var_data_diff[['info_intensity_pct', 'high_penetration_pct']], maxlag=3,
                                          verbose=False)
    gc_summary_rev = []
    for lag in range(1, 4):
        f_pvalue = gc_result_rev[lag][0]['ssr_ftest'][1]
        gc_summary_rev.append({'lag': lag, 'f_test_pvalue': f_pvalue})
        sig = "***" if f_pvalue < 0.01 else "**" if f_pvalue < 0.05 else "*" if f_pvalue < 0.1 else ""
        print(f"    lag={lag}: F-test p={f_pvalue:.4f}{sig}")
    gc_summary_rev_df = pd.DataFrame(gc_summary_rev)
    gc_summary_rev_df.to_csv(f'{OUTPUT_DIR}granger_penetration_to_intensity.csv', index=False, encoding='utf-8-sig')
except Exception as e:
    print(f"    [ERROR] {e}")

# %% 6. 可视化绘图模块（每张图自动推送至VS Code右侧Plots面板）
# ==================== 6. 可视化 ====================
print(f"\n{'=' * 60}")
print("Phase 6: Visualization")
print(f"{'=' * 60}")

# 图1: 双轴时序图
fig, ax = plt.subplots(figsize=(14, 8))
ax2 = ax.twinx()

event_window_sorted = event_window.sort_values('date')

ax.plot(event_window_sorted['date'], event_window_sorted['high_penetration_pct'],
        'o-', color='#C0392B', linewidth=3, markersize=8, label='Panic Penetration (%)')
ax2.plot(event_window_sorted['date'], event_window_sorted['info_intensity_pct'],
         's--', color='#2980B9', linewidth=3, markersize=8, label='Info Intensity (%)')

ax.axvline(pd.to_datetime('2020-01-20'), color='orange', linestyle='--', linewidth=2, label='Broadcast (Jan 20)')
ax.axvline(pd.to_datetime('2020-01-23'), color='red', linestyle='--', linewidth=2, label='Lockdown (Jan 23)')

ax.set_xlabel('Date', fontsize=13)
ax.set_ylabel('Panic Penetration (%)', fontsize=13, color='#C0392B')
ax2.set_ylabel('Info Intensity (%)', fontsize=13, color='#2980B9')
ax.set_title('Panic Penetration vs Information Intensity (Jan 15-30, 2020)', fontsize=15, fontweight='bold')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xticks(rotation=45)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig_intensity_vs_penetration.png', dpi=300, bbox_inches='tight')
plt.show() # 推送至右侧Plots
plt.close()
print(f"[SAVE] fig_intensity_vs_penetration.png")

# 图2: 领先-滞后相关系数图
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#2980B9' if l <= 0 else '#E67E22' for l in corr_df['lag']]
bars = ax.bar(corr_df['lag'], corr_df['correlation'], color=colors, alpha=0.8, edgecolor='black')

# 标注最大值
max_idx = corr_df['correlation'].abs().idxmax()
ax.annotate(f"r={corr_df.iloc[max_idx]['correlation']:.3f}\n(lag={corr_df.iloc[max_idx]['lag']})",
            xy=(corr_df.iloc[max_idx]['lag'], corr_df.iloc[max_idx]['correlation']),
            xytext=(10, 20), textcoords='offset points', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='black'))

ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Lag (days)', fontsize=13)
ax.set_ylabel('Correlation', fontsize=13)
ax.set_title('Lead-Lag Correlation: Info Intensity vs Panic Penetration', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 添加图例
legend_elements = [Patch(facecolor='#E67E22', alpha=0.8, label='Intensity leads penetration'),
                   Patch(facecolor='#2980B9', alpha=0.8, label='Penetration leads intensity')]
ax.legend(handles=legend_elements, loc='best', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig_lead_lag_correlation.png', dpi=300, bbox_inches='tight')
plt.show() # 推送至右侧Plots
plt.close()
print(f"[SAVE] fig_lead_lag_correlation.png")

# 图3: 散点图（带时间标签）
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(event_window['info_intensity_pct'], event_window['high_penetration_pct'],
                     c=range(len(event_window)), cmap='viridis', s=100, alpha=0.7, edgecolors='black')

# 标注关键日期
for _, row in event_window.iterrows():
    if row['date'].strftime('%m-%d') in ['01-20', '01-21', '01-23', '01-25']:
        ax.annotate(row['date'].strftime('%m-%d'),
                    (row['info_intensity_pct'], row['high_penetration_pct']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Information Intensity (%)', fontsize=13)
ax.set_ylabel('Panic Penetration (%)', fontsize=13)
ax.set_title('Panic Penetration vs Information Intensity\n(colored by time, Jan 15-30)', fontsize=15, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Days from Jan 15', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig_scatter_intensity_penetration.png', dpi=300, bbox_inches='tight')
plt.show() # 推送至右侧Plots
plt.close()
print(f"[SAVE] fig_scatter_intensity_penetration.png")

print(f"\n{'=' * 60}")
print(f"✅ 信息强度中介分析完成! Output: {OUTPUT_DIR}")
print(f"{'=' * 60}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")