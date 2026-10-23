#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案C终极修复：事件研究法 — 描述统计版
============================================
不跑回归（避免日级数据过拟合），直接计算每天相对于基线期的变化。
基线期 = 1.18-1.19（事件前2天），计算每天与基线均值的差异。
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
OUTPUT_DIR = '/root/autodl-tmp/group_event_study_v3d/comparison/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% 1. 数据读取与预处理
# ==================== 数据准备 ====================
overall = pd.read_csv(f'{INPUT_DIR}daily_panic_overall.csv')
overall['date'] = pd.to_datetime(overall['date_only'])
overall = overall[['date', 'overall_panic_mean']].rename(columns={'overall_panic_mean': 'panic_mean'})

pen = pd.read_csv(f'{INPUT_DIR}daily_panic_penetration.csv')
pen['date'] = pd.to_datetime(pen['date_only'])
pen = pen[['date', 'high_penetration_pct']]

merged = pd.read_csv(f'{INPUT_DIR}daily_panic_merged_tiers.csv')
merged['date'] = pd.to_datetime(merged['date_only'])
merged_wide = merged.pivot(index='date', columns='tier_group', values='panic_mean').reset_index()
merged_wide.columns.name = None
merged_wide = merged_wide.rename(columns={'High': 'High_panic', 'Low': 'Low_panic', 'Medium': 'Medium_panic'})

topic = pd.read_csv(f'{INPUT_DIR}daily_topic_proportion.csv')
topic['date'] = pd.to_datetime(topic['date_only'])
topic_wide = topic.pivot(index='date', columns='topic', values='topic_pct').reset_index().fillna(0)

df = overall.merge(pen, on='date').merge(merged_wide, on='date').merge(topic_wide, on='date', how='left')
for col in ['P8_travel', 'P6_info_seek', 'P5_emotion_vent', 'P4_isolation', 'P7_symptom', 'P2_life_supply']:
    if col not in df.columns:
        df[col] = 0.0

C1 = pd.to_datetime('2020-01-20')
C2 = pd.to_datetime('2020-01-23')

# %% 2. 事件研究指标计算
# ==================== 事件研究法：描述统计版 ====================
print("=" * 60)
print("方案C (终极修复): 事件研究法 — 描述统计版")
print("=" * 60)
print("Method: 每天值 - 基线均值(1.18-1.19), 不跑回归, 避免过拟合")
print("-" * 60)

# 基线期
BASELINE_START = pd.to_datetime('2020-01-18')
BASELINE_END = pd.to_datetime('2020-01-19')

baseline = df[(df['date'] >= BASELINE_START) & (df['date'] <= BASELINE_END)]

outcomes = ['panic_mean', 'high_penetration_pct', 'High_panic', 'P8_travel', 'P6_info_seek', 'P5_emotion_vent', 'P4_isolation']

# 计算基线均值
baseline_means = {}
for outcome in outcomes:
    baseline_means[outcome] = baseline[outcome].mean()

print("\n基线期(1.18-1.19)均值:")
for k, v in baseline_means.items():
    print(f"  {k}: {v:.4f}")

# 事件窗口 [-10, +10] 围绕 1.20
WINDOW = 10
EVENT_DATE = pd.to_datetime('2020-01-20')

df['days_to_event'] = (df['date'] - EVENT_DATE).dt.days
es_df = df[(df['days_to_event'] >= -WINDOW) & (df['days_to_event'] <= WINDOW)].copy()

# 计算每天相对基线的变化
results = []
for _, row in es_df.iterrows():
    day = int(row['days_to_event'])
    for outcome in outcomes:
        baseline_mean = baseline_means[outcome]
        if pd.notna(baseline_mean) and baseline_mean != 0:
            # 百分比变化 (对于渗透率等有意义)
            pct_change = (row[outcome] - baseline_mean) / baseline_mean * 100
        else:
            pct_change = np.nan
        # 绝对变化
        abs_change = row[outcome] - baseline_mean
        results.append({
            'day': day, 'date': row['date'], 'outcome': outcome,
            'value': row[outcome], 'baseline_mean': baseline_mean,
            'abs_change': abs_change, 'pct_change': pct_change
        })

es_results = pd.DataFrame(results)
es_results.to_csv(f'{OUTPUT_DIR}scheme_C_event_study_descriptive.csv', index=False, encoding='utf-8-sig')
print(f"\n[SAVE] scheme_C_event_study_descriptive.csv")

# 打印关键节点
print("\n关键节点相对基线变化:")
for outcome in outcomes:
    sub = es_results[es_results['outcome'] == outcome]
    print(f"\n  {outcome}:")
    for day in [-3, -2, -1, 0, 1, 2, 3]:
        d = sub[sub['day'] == day]
        if len(d) > 0:
            print(f"    D_{day:2d}: value={d.iloc[0]['value']:.4f}, abs_change={d.iloc[0]['abs_change']:+.4f}, pct_change={d.iloc[0]['pct_change']:+.1f}%")

# %% 3. 可视化绘图模块（每张图自动弹出到右侧Plots面板）
# ==================== 可视化 ====================
print(f"\n{'='*60}")
print("Visualization")
print(f"{'='*60}")

# 图1: 绝对变化 (4个关键变量)
fig_outcomes = ['high_penetration_pct', 'High_panic', 'P8_travel', 'P6_info_seek']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, outcome in enumerate(fig_outcomes):
    ax = axes[idx]
    sub = es_results[es_results['outcome'] == outcome].sort_values('day')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(-1, color='black', linestyle='--', linewidth=2, label='Broadcast (Jan 20)')
    ax.axvline(2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Lockdown (Jan 23)')

    ax.plot(sub['day'], sub['abs_change'], 'o-', color='#2980B9', linewidth=2.5, markersize=8)
    ax.fill_between(sub['day'], 0, sub['abs_change'], alpha=0.2, color='#2980B9')

    ax.set_title(f'{outcome}\n(Absolute Change from Baseline)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Days to Event (Jan 20)', fontsize=11)
    ax.set_ylabel('Absolute Change', fontsize=11)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=10, loc='best')

plt.suptitle('Scheme C (Descriptive): Dynamic Response Relative to Baseline (Jan 18-19)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig_C_event_study_abs_change.png', dpi=300, bbox_inches='tight')
plt.show()  # 推送至VS Code右侧Plots面板
plt.close()
print(f"[SAVE] fig_C_event_study_abs_change.png")

# 图2: 百分比变化 (渗透率)
fig, ax = plt.subplots(figsize=(14, 8))
sub = es_results[es_results['outcome'] == 'high_penetration_pct'].sort_values('day')

ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(-1, color='black', linestyle='--', linewidth=2, label='Broadcast (Jan 20)')
ax.axvline(2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Lockdown (Jan 23)')

ax.plot(sub['day'], sub['pct_change'], 'o-', color='#C0392B', linewidth=3, markersize=10)
ax.fill_between(sub['day'], 0, sub['pct_change'], alpha=0.2, color='#C0392B')

# 标注关键节点
for _, row in sub.iterrows():
    if row['day'] in [-3, -1, 0, 2, 3]:
        ax.annotate(f"{row['pct_change']:+.0f}%",
                   xy=(row['day'], row['pct_change']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_title('Panic Penetration Rate: % Change from Baseline (Jan 18-19)', fontsize=14, fontweight='bold')
ax.set_xlabel('Days to Event (Jan 20)', fontsize=13)
ax.set_ylabel('% Change from Baseline', fontsize=13)
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig_C_penetration_pct_change.png', dpi=300, bbox_inches='tight')
plt.show()  # 推送至VS Code右侧Plots面板
plt.close()
print(f"[SAVE] fig_C_penetration_pct_change.png")

# 图3: 所有主题叠加 (绝对变化)
fig, ax = plt.subplots(figsize=(14, 8))
topic_outcomes = ['P8_travel', 'P6_info_seek', 'P5_emotion_vent', 'P4_isolation']
colors = {'P8_travel': '#E67E22', 'P6_info_seek': '#2980B9', 'P5_emotion_vent': '#C0392B', 'P4_isolation': '#27AE60'}

for outcome in topic_outcomes:
    sub = es_results[es_results['outcome'] == outcome].sort_values('day')
    ax.plot(sub['day'], sub['abs_change'], 'o-', color=colors[outcome], linewidth=2.5, markersize=8, label=outcome)

ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(-1, color='black', linestyle='--', linewidth=2, label='Broadcast (Jan 20)')
ax.axvline(2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Lockdown (Jan 23)')
ax.set_title('Topic Proportions: Absolute Change from Baseline', fontsize=14, fontweight='bold')
ax.set_xlabel('Days to Event (Jan 20)', fontsize=13)
ax.set_ylabel('Absolute Change (percentage points)', fontsize=13)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig_C_topics_abs_change.png', dpi=300, bbox_inches='tight')
plt.show()  # 推送至VS Code右侧Plots面板
plt.close()
print(f"[SAVE] fig_C_topics_abs_change.png")

print(f"\n{'='*60}")
print(f"✅ 方案C (描述版) 完成! Output: {OUTPUT_DIR}")
print(f"{'='*60}")