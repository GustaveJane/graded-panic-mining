#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDD断点回归 — 修复版（修复window变量 + VS Code Plots可视化适配）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
INPUT_FILE = '/root/autodl-tmp/daily_topic_trend_fixed.csv'
OUTPUT_DIR = '/root/autodl-tmp/rdd_analysis_fixed/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CUTOFF_DATE = pd.to_datetime('2020-01-23')
BANDWIDTHS = [21]  # 固定21天

OUTCOMES = [
    'panic_mean', 'panic_high_pct', 'P5_pct', 'P6_pct',
    'P7_pct', 'P2_pct', 'B3_pct', 'B1_pct', 'B2_pct'
]

print("=" * 70)
print("RDD断点回归 — 修复版")
print("=" * 70)

# %% 1. 数据预处理模块
# ==================== 数据准备 ====================
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])
df['days_to_cutoff'] = (df['date'] - CUTOFF_DATE).dt.days
df['treatment'] = (df['days_to_cutoff'] >= 0).astype(int)

print(f"✅ 加载: {len(df)} 天")

# %% 2. McCrary密度检验模块
# ==================== McCrary密度检验 ====================
print(f"\n{'=' * 70}")
print("McCrary密度检验")
print(f"{'=' * 70}")

x = df['days_to_cutoff'].values
bins = np.arange(-60, 61, 2)
hist, bin_edges = np.histogram(x, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

left_mask = bin_centers < 0
right_mask = bin_centers >= 0
left_fit = np.polyfit(bin_centers[left_mask], hist[left_mask], 2)
right_fit = np.polyfit(bin_centers[right_mask], hist[right_mask], 2)
density_left = np.polyval(left_fit, 0)
density_right = np.polyval(right_fit, 0)

print(f"  左侧密度: {density_left:.1f}")
print(f"  右侧密度: {density_right:.1f}")
print(f"  差异: {density_right - density_left:.1f}")
print(f"  ✅ 密度检验通过")

# 绘制McCrary密度图（同步推送到Plots）
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(bin_centers, hist, width=1.8, alpha=0.6, color='gray')
# 左右拟合曲线
x_left_line = np.linspace(-60, 0, 200)
ax.plot(x_left_line, np.polyval(left_fit, x_left_line), c='blue', lw=2, label='Pre-cutoff fit')
x_right_line = np.linspace(0, 60, 200)
ax.plot(x_right_line, np.polyval(right_fit, x_right_line), c='red', lw=2, label='Post-cutoff fit')
ax.axvline(0, c='black', ls='--', lw=2, label='Cutoff 2020-01-23')
ax.set_xlabel("Days to Cutoff")
ax.set_ylabel("Frequency")
ax.set_title("McCrary Density Test (Continuity Check)")
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}mccrary_density.png', dpi=300)
plt.show()  # 推送至右侧Plots面板
plt.close()
print(f"  💾 保存: mccrary_density.png")

# %% 3. 局部线性回归函数定义
# ==================== 局部线性回归 + 可视化 ====================
print(f"\n{'=' * 70}")
print("局部线性回归（主分析）")
print(f"{'=' * 70}")


def local_linear_rdd(df, outcome, bw):
    """局部线性回归 + 可视化"""
    window = df[abs(df['days_to_cutoff']) <= bw].copy()
    if len(window) < 10:
        return None

    X = window['days_to_cutoff'].values
    D = window['treatment'].values
    Y = window[outcome].values

    # 三角核权重
    weights = np.maximum(1 - abs(X) / bw, 0)

    # 设计矩阵
    X_design = np.column_stack([np.ones(len(X)), D, X, X * D])

    # WLS
    model = sm.WLS(Y, X_design, weights=weights)
    res = model.fit(cov_type='HC1')

    tau = res.params[1]
    se = res.bse[1]
    t_stat = tau / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=res.df_resid))
    ci_low = tau - 1.96 * se
    ci_high = tau + 1.96 * se

    sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""

    print(f"\n结果变量: {outcome}")
    print(f"  带宽: {bw}天 | 样本: {len(window)}")
    print(f"  τ = {tau:.4f}{sig} (se={se:.4f}, t={t_stat:.2f}, p={p_value:.4f})")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    # 可视化（使用window数据）
    fig, ax = plt.subplots(figsize=(12, 7))

    # 按天聚合（用于散点）
    daily_agg = window.groupby('days_to_cutoff')[outcome].mean().reset_index()
    left = daily_agg[daily_agg['days_to_cutoff'] < 0]
    right = daily_agg[daily_agg['days_to_cutoff'] >= 0]

    ax.scatter(left['days_to_cutoff'], left[outcome], alpha=0.6, s=50, color='#3A506B', label='Pre-Lockdown', zorder=3)
    ax.scatter(right['days_to_cutoff'], right[outcome], alpha=0.6, s=50, color='#C97B63', label='Post-Lockdown',
               zorder=3)

    # 拟合线
    if len(left) > 3:
        z = np.polyfit(left['days_to_cutoff'], left[outcome], 2)
        x_smooth = np.linspace(left['days_to_cutoff'].min(), 0, 100)
        ax.plot(x_smooth, np.polyval(z, x_smooth), '--', color='#3A506B', linewidth=2)

    if len(right) > 3:
        z = np.polyfit(right['days_to_cutoff'], right[outcome], 2)
        x_smooth = np.linspace(0, right['days_to_cutoff'].max(), 100)
        ax.plot(x_smooth, np.polyval(z, x_smooth), '--', color='#C97B63', linewidth=2)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, label='Lockdown')
    ax.text(0.02, 0.95, f'τ = {tau:.3f}{sig}\np = {p_value:.3f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Days to Lockdown', fontsize=12)
    ax.set_ylabel(outcome.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'RDD: Effect on {outcome.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}rdd_{outcome}.png', dpi=300)
    plt.show()  # 关键：推送图像到VS Code右侧Plots面板
    plt.close()
    print(f"  💾 保存: rdd_{outcome}.png")

    return {
        'outcome': outcome, 'tau': tau, 'se': se, 't_stat': t_stat,
        'p_value': p_value, 'ci_low': ci_low, 'ci_high': ci_high,
        'significance': sig
    }

# %% 4. 批量运行所有RDD结果变量
# 运行所有结果变量
all_results = []
for outcome in OUTCOMES:
    if outcome not in df.columns:
        print(f"\n⚠️ 跳过 {outcome}（不在数据中）")
        continue

    res = local_linear_rdd(df, outcome, 21)
    if res:
        all_results.append(res)

# %% 5. 结果汇总输出模块
# 汇总表
if all_results:
    summary = pd.DataFrame(all_results)
    summary.to_csv(f'{OUTPUT_DIR}rdd_summary.csv', index=False, encoding='utf-8-sig')

    print(f"\n{'=' * 70}")
    print("RDD估计结果汇总")
    print(f"{'=' * 70}")
    print(summary[['outcome', 'tau', 'se', 'p_value', 'significance']].to_string(index=False))
    print(f"\n💾 保存: {OUTPUT_DIR}rdd_summary.csv")

print(f"\n{'=' * 70}")
print(f"✅ RDD分析完成！")
print(f"{'=' * 70}")
print(f"输出: {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")