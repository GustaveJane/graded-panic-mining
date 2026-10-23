#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向B：稳健性检验 (Sensitivity Analysis)
==========================================
1. 替换事件窗口
2. 替换基线期
3. 替换High tier定义
VS Code右侧Plots可视化适配版
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# 全局绘图配置，解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

INPUT_DIR = '/root/autodl-tmp/group_event_study_v3d/'
OUTPUT_DIR = '/root/autodl-tmp/group_event_study_v3d/robustness/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("方向B: 稳健性检验")
print("=" * 60)

# %% 1. 读取基础数据
df = pd.read_csv(f'{INPUT_DIR}daily_panic_by_tier.csv')
df['date'] = pd.to_datetime(df['date_only'])
df_pen = pd.read_csv(f'{INPUT_DIR}daily_panic_penetration.csv')
df_pen['date'] = pd.to_datetime(df_pen['date_only'])

# %% 2. 检验1: 替换事件窗口
print("\n--- 检验1: 替换事件窗口 ---")
windows = {
    'window_1.18_1.25': ('2020-01-18', '2020-01-25'),
    'window_1.17_1.26': ('2020-01-17', '2020-01-26'),
    'window_1.19_1.24': ('2020-01-19', '2020-01-24'),
}

results_window = []
for wname, (start, end) in windows.items():
    pen_window = df_pen[(df_pen['date'] >= start) & (df_pen['date'] <= end)]
    baseline = pen_window[(pen_window['date'] >= start) & (pen_window['date'] <= '2020-01-19')]
    anticipation = pen_window[(pen_window['date'] >= '2020-01-20') & (pen_window['date'] <= '2020-01-22')]
    announcement = pen_window[(pen_window['date'] >= '2020-01-23') & (pen_window['date'] <= end)]

    b = baseline['high_penetration_pct'].mean() if len(baseline) > 0 else np.nan
    a = anticipation['high_penetration_pct'].mean() if len(anticipation) > 0 else np.nan
    n = announcement['high_penetration_pct'].mean() if len(announcement) > 0 else np.nan

    results_window.append({
        'window': wname, 'baseline': round(b, 4) if pd.notna(b) else None,
        'anticipation': round(a, 4) if pd.notna(a) else None,
        'announcement': round(n, 4) if pd.notna(n) else None,
        'delta_anticipation': round(a - b, 4) if pd.notna(a) and pd.notna(b) else None,
        'delta_announcement': round(n - a, 4) if pd.notna(n) and pd.notna(a) else None,
    })

window_df = pd.DataFrame(results_window)
window_df.to_csv(f'{OUTPUT_DIR}robustness_window.csv', index=False, encoding='utf-8-sig')
print("\n  事件窗口稳健性:")
print(window_df.to_string(index=False))

print(f"\n  方向一致性检查:")
for _, row in window_df.iterrows():
    da = row['delta_anticipation']
    dn = row['delta_announcement']
    if pd.notna(da) and pd.notna(dn):
        up = da > 0
        down = dn < 0
        print(
            f"    {row['window']}: anticipation {'↑' if up else '↓'}, announcement {'↓' if down else '↑'} {'✅' if up and down else '❌'}")

# %% 3. 检验2: 替换基线期
print("\n--- 检验2: 替换基线期 ---")
baselines = {
    'baseline_1.18_1.19': ('2020-01-18', '2020-01-19'),
    'baseline_1.15_1.19': ('2020-01-15', '2020-01-19'),
    'baseline_1.17_1.19': ('2020-01-17', '2020-01-19'),
}

results_baseline = []
for bname, (bstart, bend) in baselines.items():
    base = df_pen[(df_pen['date'] >= bstart) & (df_pen['date'] <= bend)]['high_penetration_pct'].mean()
    anti = df_pen[(df_pen['date'] >= '2020-01-20') & (df_pen['date'] <= '2020-01-22')]['high_penetration_pct'].mean()
    ann = df_pen[(df_pen['date'] >= '2020-01-23') & (df_pen['date'] <= '2020-01-25')]['high_penetration_pct'].mean()

    results_baseline.append({
        'baseline_period': bname,
        'baseline_mean': round(base, 4) if pd.notna(base) else None,
        'anticipation_mean': round(anti, 4) if pd.notna(anti) else None,
        'announcement_mean': round(ann, 4) if pd.notna(ann) else None,
    })

baseline_df = pd.DataFrame(results_baseline)
baseline_df.to_csv(f'{OUTPUT_DIR}robustness_baseline.csv', index=False, encoding='utf-8-sig')
print("\n  基线期稳健性:")
print(baseline_df.to_string(index=False))

# %% 4. 检验3: High tier定义方式
print("\n--- 检验3: High tier 定义方式 ---")
high_defs = {
    'High_3+4': [3, 4],
    'High_4_only': [4],
    'High_2+3+4': [2, 3, 4],
}

results_high_def = []
for hname, tiers in high_defs.items():
    daily_total = df.groupby('date')['post_count'].sum().reset_index(name='total')
    daily_high = df[df['gmm_tier'].isin(tiers)].groupby('date')['post_count'].sum().reset_index(name='high')
    merged_high = daily_total.merge(daily_high, on='date', how='left').fillna(0)
    merged_high['penetration'] = merged_high['high'] / merged_high['total'] * 100

    b = merged_high[(merged_high['date'] >= '2020-01-18') & (merged_high['date'] <= '2020-01-19')]['penetration'].mean()
    a = merged_high[(merged_high['date'] >= '2020-01-20') & (merged_high['date'] <= '2020-01-22')]['penetration'].mean()
    n = merged_high[(merged_high['date'] >= '2020-01-23') & (merged_high['date'] <= '2020-01-25')]['penetration'].mean()

    results_high_def.append({
        'high_definition': hname,
        'baseline': round(b, 4) if pd.notna(b) else None,
        'anticipation': round(a, 4) if pd.notna(a) else None,
        'announcement': round(n, 4) if pd.notna(n) else None,
    })

high_df = pd.DataFrame(results_high_def)
high_df.to_csv(f'{OUTPUT_DIR}robustness_high_definition.csv', index=False, encoding='utf-8-sig')
print("\n  High tier定义稳健性:")
print(high_df.to_string(index=False))

# %% 5. 可视化绘图模块（自动推送至右侧Plots）
print(f"\n{'=' * 60}")
print("Visualization")
print(f"{'=' * 60}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：不同事件窗口时序曲线
ax = axes[0, 0]
for wname, (start, end) in windows.items():
    sub = df_pen[(df_pen['date'] >= start) & (df_pen['date'] <= end)].sort_values('date')
    ax.plot(sub['date'], sub['high_penetration_pct'], 'o-', label=wname, linewidth=2)
ax.axvline(pd.to_datetime('2020-01-20'), color='orange', linestyle='--', alpha=0.5)
ax.axvline(pd.to_datetime('2020-01-23'), color='red', linestyle='--', alpha=0.5)
ax.set_title('Robustness: Event Window', fontsize=12, fontweight='bold')
ax.set_ylabel('Penetration (%)')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# 子图2：不同基线期均值对比柱状图
ax = axes[0, 1]
x = np.arange(len(baseline_df))
width = 0.25
ax.bar(x - width, baseline_df['baseline_mean'], width, label='Baseline', alpha=0.8)
ax.bar(x, baseline_df['anticipation_mean'], width, label='Anticipation', alpha=0.8)
ax.bar(x + width, baseline_df['announcement_mean'], width, label='Announcement', alpha=0.8)
ax.set_title('Robustness: Baseline Period', fontsize=12, fontweight='bold')
ax.set_ylabel('Penetration (%)')
ax.set_xticks(x)
ax.set_xticklabels(baseline_df['baseline_period'], rotation=15, ha='right', fontsize=8)
ax.legend(fontsize=8)

# 子图3：不同高风险分组定义均值对比
ax = axes[1, 0]
x = np.arange(len(high_df))
ax.bar(x - width, high_df['baseline'], width, label='Baseline', alpha=0.8)
ax.bar(x, high_df['anticipation'], width, label='Anticipation', alpha=0.8)
ax.bar(x + width, high_df['announcement'], width, label='Announcement', alpha=0.8)
ax.set_title('Robustness: High Tier Definition', fontsize=12, fontweight='bold')
ax.set_ylabel('Penetration (%)')
ax.set_xticks(x)
ax.set_xticklabels(high_df['high_definition'], fontsize=9)
ax.legend(fontsize=8)

# 子图4：效应方向一致性检验
ax = axes[1, 1]
consistency = []
for _, row in window_df.iterrows():
    da = row['delta_anticipation']
    dn = row['delta_announcement']
    consistency.append(1 if pd.notna(da) and pd.notna(dn) and da > 0 and dn < 0 else 0)
ax.barh(range(len(window_df)), consistency, color=['#27AE60' if c else '#C0392B' for c in consistency])
ax.set_yticks(range(len(window_df)))
ax.set_yticklabels(window_df['window'], fontsize=9)
ax.set_xlim(0, 1.2)
ax.set_title('Direction Consistency', fontsize=12, fontweight='bold')
for i, c in enumerate(consistency):
    ax.text(c + 0.05, i, '✅' if c else '❌', va='center', fontsize=12)

plt.suptitle('Robustness Checks Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig_robustness_summary.png', dpi=300, bbox_inches='tight')
plt.show() # 推送至VS Code右侧Plots面板
plt.close()
print(f"[SAVE] fig_robustness_summary.png")

print(f"\n✅ 方向B完成! Output: {OUTPUT_DIR}")