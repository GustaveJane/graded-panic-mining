#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三种断点方案对比实验
========================
方案A: 分开双断点RDD (1.20独立, 1.23独立, 各跑各的)
方案B: 联合双断点RDD (1.20和1.23同时在一个回归里)
方案C: 事件研究法 (1.20为事件日, 1.23作为D_{+3})

对比维度:
  - 系数稳定性 (不同带宽: 3/5/7天)
  - 标准误大小 (精度)
  - 显著性一致性
  - 可视化直观对比
VS Code右侧Plots可视化适配版
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# 全局绘图配置，解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

INPUT_DIR = '/root/autodl-tmp/group_event_study_v3d/'
OUTPUT_DIR = '/root/autodl-tmp/group_event_study_v3d/comparison/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% 1. 数据读取与预处理
# ==================== 数据准备 ====================
print("=" * 60)
print("Phase 0: Data Preparation")
print("=" * 60)

overall = pd.read_csv(f'{INPUT_DIR}daily_panic_overall.csv')
overall['date'] = pd.to_datetime(overall['date_only'])
overall = overall[['date', 'overall_panic_mean', 'total_posts']].rename(
    columns={'overall_panic_mean': 'panic_mean'}
)

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

print(f"Data: {len(df)} days, {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"Key outcomes: panic_mean, high_penetration_pct, High_panic, P8_travel, P6_info_seek, P5_emotion_vent, P4_isolation")

outcomes = ['panic_mean', 'high_penetration_pct', 'High_panic', 'P8_travel', 'P6_info_seek', 'P5_emotion_vent', 'P4_isolation']

# %% 2. 方案A：分开双断点RDD计算
# ==================== 方案A: 分开双断点RDD ====================
print(f"\n{'='*60}")
print("方案A: 分开双断点RDD (Separate RDDs)")
print(f"{'='*60}")

def rdd_single(df, outcome, cutoff, bw):
    """单断点局部线性回归"""
    df['days_to_cutoff'] = (df['date'] - cutoff).dt.days
    window = df[abs(df['days_to_cutoff']) <= bw].copy()
    if len(window) < 6:
        return None
    X = window['days_to_cutoff'].values
    D = (X >= 0).astype(int)
    Y = window[outcome].values
    weights = np.maximum(1 - abs(X) / bw, 0)
    X_design = np.column_stack([np.ones(len(X)), D, X, X * D])
    try:
        res = sm.WLS(Y, X_design, weights=weights).fit(cov_type='HC1')
    except:
        return None
    tau = res.params[1]
    se = res.bse[1]
    p = 2 * (1 - stats.t.cdf(abs(tau/se), df=res.df_resid)) if se > 0 else 1.0
    return {'tau': tau, 'se': se, 'pvalue': p, 'n': len(window), 'bw': bw}

results_A = []
for outcome in outcomes:
    for bw in [3, 5, 7]:
        for cutoff, label in [(C1, '1.20_Broadcast'), (C2, '1.23_Lockdown')]:
            res = rdd_single(df, outcome, cutoff, bw)
            if res:
                results_A.append({
                    'scheme': 'A_Separate', 'outcome': outcome, 'cutoff': label,
                    'bw': bw, **res
                })

df_A = pd.DataFrame(results_A)
if len(df_A) > 0:
    df_A.to_csv(f'{OUTPUT_DIR}scheme_A_separate_rdd.csv', index=False, encoding='utf-8-sig')
    print(f"[SAVE] scheme_A_separate_rdd.csv ({len(df_A)} estimates)")
    print("\n方案A 结果摘要:")
    for outcome in outcomes:
        sub = df_A[df_A['outcome'] == outcome]
        if len(sub) == 0:
            continue
        print(f"\n  {outcome}:")
        for _, row in sub.iterrows():
            sig = "***" if row['pvalue'] < 0.01 else "**" if row['pvalue'] < 0.05 else "*" if row['pvalue'] < 0.1 else ""
            print(f"    {row['cutoff']} bw={row['bw']}d: τ={row['tau']:.4f}{sig} (se={row['se']:.4f}, p={row['pvalue']:.3f}, n={row['n']})")

# %% 3. 方案B：联合双断点RDD计算
# ==================== 方案B: 联合双断点RDD ====================
print(f"\n{'='*60}")
print("方案B: 联合双断点RDD (Joint Model)")
print(f"{'='*60}")
print("Model: Y = β0 + β1·D1(1.20) + β2·D2(1.23) + β3·(X-1.20)·D1 + β4·(X-1.23)·D2 + β5·X + ε")

def rdd_joint(df, outcome, bw):
    """双断点联合模型"""
    df['days_c1'] = (df['date'] - C1).dt.days
    df['days_c2'] = (df['date'] - C2).dt.days
    # 窗口: 覆盖两个断点 ±bw
    window = df[(abs(df['days_c1']) <= bw) | (abs(df['days_c2']) <= bw)].copy()
    if len(window) < 10:
        return None

    X1 = window['days_c1'].values
    X2 = window['days_c2'].values
    D1 = (X1 >= 0).astype(int)
    D2 = (X2 >= 0).astype(int)
    Y = window[outcome].values

    # 三角核权重: 取到最近断点的距离
    dist_to_c1 = abs(X1)
    dist_to_c2 = abs(X2)
    min_dist = np.minimum(dist_to_c1, dist_to_c2)
    weights = np.maximum(1 - min_dist / bw, 0)

    X_design = np.column_stack([np.ones(len(X1)), D1, D2, X1 * D1, X2 * D2, X1])
    try:
        res = sm.WLS(Y, X_design, weights=weights).fit(cov_type='HC1')
    except Exception as e:
        return None

    tau1 = res.params[1]  # 1.20 效应
    tau2 = res.params[2]  # 1.23 效应
    se1 = res.bse[1]
    se2 = res.bse[2]
    p1 = 2 * (1 - stats.t.cdf(abs(tau1/se1), df=res.df_resid)) if se1 > 0 else 1.0
    p2 = 2 * (1 - stats.t.cdf(abs(tau2/se2), df=res.df_resid)) if se2 > 0 else 1.0

    return {
        'tau1': tau1, 'se1': se1, 'pvalue1': p1,
        'tau2': tau2, 'se2': se2, 'pvalue2': p2,
        'n': len(window), 'bw': bw, 'r2': res.rsquared
    }

results_B = []
for outcome in outcomes:
    for bw in [3, 5, 7]:
        res = rdd_joint(df, outcome, bw)
        if res:
            results_B.append({'scheme': 'B_Joint', 'outcome': outcome, **res})

df_B = pd.DataFrame(results_B)
if len(df_B) > 0:
    df_B.to_csv(f'{OUTPUT_DIR}scheme_B_joint_rdd.csv', index=False, encoding='utf-8-sig')
    print(f"[SAVE] scheme_B_joint_rdd.csv ({len(df_B)} estimates)")
    print("\n方案B 结果摘要:")
    for outcome in outcomes:
        sub = df_B[df_B['outcome'] == outcome]
        if len(sub) == 0:
            continue
        print(f"\n  {outcome}:")
        for _, row in sub.iterrows():
            s1 = "***" if row['pvalue1'] < 0.01 else "**" if row['pvalue1'] < 0.05 else "*" if row['pvalue1'] < 0.1 else ""
            s2 = "***" if row['pvalue2'] < 0.01 else "**" if row['pvalue2'] < 0.05 else "*" if row['pvalue2'] < 0.1 else ""
            print(f"    bw={row['bw']}d: τ1(1.20)={row['tau1']:.4f}{s1} (p={row['pvalue1']:.3f}), τ2(1.23)={row['tau2']:.4f}{s2} (p={row['pvalue2']:.3f}), n={row['n']}, R²={row['r2']:.3f}")

# %% 4. 方案C：事件研究法计算
# ==================== 方案C: 事件研究法 (Event Study) ====================
print(f"\n{'='*60}")
print("方案C: 事件研究法 (Event Study)")
print(f"{'='*60}")

EVENT_DATE = pd.to_datetime('2020-01-20')
WINDOW = 10

df['days_to_event'] = (df['date'] - EVENT_DATE).dt.days
es_df = df[(df['days_to_event'] >= -WINDOW) & (df['days_to_event'] <= WINDOW)].copy()
es_df['event_time'] = es_df['days_to_event'].astype(int)
dummy_df = pd.get_dummies(es_df['event_time'], prefix='D')
if -1 in dummy_df.columns:
    dummy_df = dummy_df.drop(columns=[-1])
es_df = pd.concat([es_df, dummy_df], axis=1)
dummy_cols = [c for c in es_df.columns if str(c).startswith('D_')]
es_df['linear_trend'] = es_df['days_to_event']

results_C = []
for outcome in outcomes:
    if outcome not in es_df.columns:
        continue
    valid = es_df[es_df[outcome].notna()].copy()
    if len(valid) < 15:
        continue
    X = sm.add_constant(valid[['linear_trend'] + dummy_cols])
    y = valid[outcome].values
    model = sm.OLS(y, X).fit(cov_type='HC1')

    for col in dummy_cols:
        day = int(col.replace('D_', ''))
        coef = model.params[col]
        se = model.bse[col]
        pval = model.pvalues[col]
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        results_C.append({
            'scheme': 'C_EventStudy', 'outcome': outcome, 'day': day,
            'coef': coef, 'se': se, 'pvalue': pval, 'ci_low': ci_low, 'ci_high': ci_high, 'significance': sig
        })

df_C = pd.DataFrame(results_C)
if len(df_C) > 0:
    df_C.to_csv(f'{OUTPUT_DIR}scheme_C_event_study.csv', index=False, encoding='utf-8-sig')
    print(f"[SAVE] scheme_C_event_study.csv ({len(df_C)} coefficients)")

    # 关键节点提取
    print("\n方案C 关键节点系数:")
    for outcome in outcomes:
        sub = df_C[(df_C['outcome'] == outcome) & (df_C['day'].isin([-3, -2, -1, 0, 1, 2, 3]))]
        if len(sub) == 0:
            continue
        print(f"\n  {outcome}:")
        for _, row in sub.iterrows():
            sig = "***" if row['pvalue'] < 0.01 else "**" if row['pvalue'] < 0.05 else "*" if row['pvalue'] < 0.1 else ""
            print(f"    D_{row['day']:2d}: coef={row['coef']:+.4f}{sig} (p={row['pvalue']:.3f})")

# %% 5. 综合数值诊断对比
# ==================== 综合对比 ====================
print(f"\n{'='*60}")
print("Phase 4: 综合对比与诊断")
print(f"{'='*60}")

# 1. 方案A vs 方案B 的 τ 一致性对比
print("\n【诊断1】方案A vs 方案B 的系数一致性 (bw=7d)")
print("-" * 60)
for outcome in outcomes:
    a1 = df_A[(df_A['outcome']==outcome) & (df_A['cutoff']=='1.20_Broadcast') & (df_A['bw']==7)]
    a2 = df_A[(df_A['outcome']==outcome) & (df_A['cutoff']=='1.23_Lockdown') & (df_A['bw']==7)]
    b = df_B[(df_B['outcome']==outcome) & (df_B['bw']==7)]
    if len(a1)==0 or len(a2)==0 or len(b)==0:
        continue
    print(f"\n  {outcome}:")
    print(f"    A-1.20: τ={a1.iloc[0]['tau']:.4f} (p={a1.iloc[0]['pvalue']:.3f})")
    print(f"    B-1.20: τ={b.iloc[0]['tau1']:.4f} (p={b.iloc[0]['pvalue1']:.3f})")
    print(f"    A-1.23: τ={a2.iloc[0]['tau']:.4f} (p={a2.iloc[0]['pvalue']:.3f})")
    print(f"    B-1.23: τ={b.iloc[0]['tau2']:.4f} (p={b.iloc[0]['pvalue2']:.3f})")
    diff1 = abs(a1.iloc[0]['tau'] - b.iloc[0]['tau1'])
    diff2 = abs(a2.iloc[0]['tau'] - b.iloc[0]['tau2'])
    print(f"    → 1.20差异: {diff1:.4f} | 1.23差异: {diff2:.4f}")
    if diff1 > 0.01 or diff2 > 0.01:
        print(f"    ⚠️ 系数差异较大，说明断点间存在污染/共线性问题")
    else:
        print(f"    ✅ 系数基本一致，结果稳健")

# 2. 方案A 带宽敏感性
print(f"\n{'='*60}")
print("【诊断2】方案A 带宽敏感性 (同一断点, 不同带宽)")
print("-" * 60)
for outcome in outcomes:
    for cutoff in ['1.20_Broadcast', '1.23_Lockdown']:
        sub = df_A[(df_A['outcome']==outcome) & (df_A['cutoff']==cutoff)].sort_values('bw')
        if len(sub) < 2:
            continue
        taus = sub['tau'].values
        max_diff = max(taus) - min(taus)
        print(f"\n  {outcome} @ {cutoff}: τ(bw3)={taus[0]:.4f}, τ(bw5)={taus[1]:.4f}, τ(bw7)={taus[2]:.4f}, max_diff={max_diff:.4f}")
        if max_diff > 0.02:
            print(f"    ⚠️ 带宽敏感，结果不稳定")
        else:
            print(f"    ✅ 带宽稳健")

# 3. 方案B 自由度诊断
print(f"\n{'='*60}")
print("【诊断3】方案B 自由度与模型拟合")
print("-" * 60)
for outcome in outcomes:
    sub = df_B[df_B['outcome']==outcome].sort_values('bw')
    if len(sub) == 0:
        continue
    print(f"\n  {outcome}:")
    for _, row in sub.iterrows():
        print(f"    bw={row['bw']}d: n={row['n']}, R²={row['r2']:.3f}")
    # 检查R²是否异常高(过拟合)或异常低(模型失效)
    r2s = sub['r2'].values
    if max(r2s) > 0.95:
        print(f"    ⚠️ R²过高，可能存在过拟合")
    elif min(r2s) < 0.01:
        print(f"    ⚠️ R²过低，模型解释力不足")
    else:
        print(f"    ✅ R²在合理范围")

# %% 6. 可视化绘图模块（每张图自动渲染到VS Code右侧Plots面板）
# ==================== 可视化对比 ====================
print(f"\n{'='*60}")
print("Phase 5: Visualization")
print(f"{'='*60}")

# 图1: 方案A vs B 效应系数横向对比图
fig_outcomes = ['high_penetration_pct', 'P8_travel', 'P6_info_seek']
fig, axes = plt.subplots(3, 1, figsize=(16, 18))

for idx, outcome in enumerate(fig_outcomes):
    ax = axes[idx]

    # 方案A: 分开RDD (bw=7)
    a1 = df_A[(df_A['outcome']==outcome) & (df_A['cutoff']=='1.20_Broadcast') & (df_A['bw']==7)]
    a2 = df_A[(df_A['outcome']==outcome) & (df_A['cutoff']=='1.23_Lockdown') & (df_A['bw']==7)]
    if len(a1) > 0:
        ax.barh(2, a1.iloc[0]['tau'], height=0.3, color='#E67E22', alpha=0.7, label=f"A-Sep 1.20: τ={a1.iloc[0]['tau']:.4f}")
    if len(a2) > 0:
        ax.barh(1, a2.iloc[0]['tau'], height=0.3, color='#C0392B', alpha=0.7, label=f"A-Sep 1.23: τ={a2.iloc[0]['tau']:.4f}")

    # 方案B: 联合RDD (bw=7)
    b = df_B[(df_B['outcome']==outcome) & (df_B['bw']==7)]
    if len(b) > 0:
        ax.barh(0.5, b.iloc[0]['tau1'], height=0.2, color='#F39C12', alpha=0.9, label=f"B-Joint 1.20: τ={b.iloc[0]['tau1']:.4f}")
        ax.barh(0, b.iloc[0]['tau2'], height=0.2, color='#E74C3C', alpha=0.9, label=f"B-Joint 1.23: τ={b.iloc[0]['tau2']:.4f}")

    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks([0, 0.5, 1, 2])
    ax.set_yticklabels(['B-1.23', 'B-1.20', 'A-1.23', 'A-1.20'])
    ax.set_xlabel('Treatment Effect (τ)', fontsize=12)
    ax.set_title(f'{outcome}: Separate vs Joint RDD (bw=7d)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('Scheme A vs B: Treatment Effect Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig_comparison_A_vs_B.png', dpi=300, bbox_inches='tight')
plt.show() # 推送至右侧Plots
plt.close()
print(f"[SAVE] fig_comparison_A_vs_B.png")

# 图2: 方案C 事件研究动态效应时序图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
plot_outcomes = ['high_penetration_pct', 'High_panic', 'P8_travel', 'P6_info_seek']
for idx, outcome in enumerate(plot_outcomes):
    ax = axes[idx]
    sub = df_C[df_C['outcome'] == outcome].sort_values('day')
    if len(sub) == 0:
        continue
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(-1, color='black', linestyle='--', linewidth=1.5, label='Event (Jan 20)')
    ax.axvline(2, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Lockdown (Jan 23)')
    colors = ['#C0392B' if s != '' else '#2980B9' for s in sub['significance']]
    ax.scatter(sub['day'], sub['coef'], c=colors, s=50, zorder=3)
    ax.plot(sub['day'], sub['coef'], color='#2980B9', linewidth=1.5, alpha=0.7)
    ax.fill_between(sub['day'], sub['ci_low'], sub['ci_high'], alpha=0.2, color='#2980B9')
    ax.set_title(outcome, fontsize=12, fontweight='bold')
    ax.set_xlabel('Days to Event (Jan 20)', fontsize=10)
    ax.set_ylabel('Coefficient', fontsize=10)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=9, loc='upper left')
plt.suptitle('Scheme C: Event Study Dynamic Effects', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig_comparison_C_event_study.png', dpi=300, bbox_inches='tight')
plt.show() # 推送至右侧Plots
plt.close()
print(f"[SAVE] fig_comparison_C_event_study.png")

# 图3: 显著性热力对比图
if len(df_A) > 0 and len(df_B) > 0:
    fig, ax = plt.subplots(figsize=(14, 8))
    # 构造对比矩阵
    comp_data = []
    for outcome in outcomes:
        for bw in [3, 5, 7]:
            a1 = df_A[(df_A['outcome']==outcome) & (df_A['cutoff']=='1.20_Broadcast') & (df_A['bw']==bw)]
            a2 = df_A[(df_A['outcome']==outcome) & (df_A['cutoff']=='1.23_Lockdown') & (df_A['bw']==bw)]
            b = df_B[(df_B['outcome']==outcome) & (df_B['bw']==bw)]
            row = {'outcome': outcome, 'bw': bw}
            row['A_1.20_sig'] = 1 if len(a1)>0 and a1.iloc[0]['pvalue']<0.1 else 0
            row['A_1.23_sig'] = 1 if len(a2)>0 and a2.iloc[0]['pvalue']<0.1 else 0
            row['B_1.20_sig'] = 1 if len(b)>0 and b.iloc[0]['pvalue1']<0.1 else 0
            row['B_1.23_sig'] = 1 if len(b)>0 and b.iloc[0]['pvalue2']<0.1 else 0
            comp_data.append(row)
    comp_df = pd.DataFrame(comp_data)
    comp_pivot = comp_df.pivot(index='outcome', columns='bw', values=['A_1.20_sig', 'A_1.23_sig', 'B_1.20_sig', 'B_1.23_sig'])

    # 简化热力图: 只画bw=7
    comp7 = comp_df[comp_df['bw']==7][['outcome', 'A_1.20_sig', 'A_1.23_sig', 'B_1.20_sig', 'B_1.23_sig']].set_index('outcome')
    sns.heatmap(comp7, annot=True, fmt='d', cmap='RdYlGn', cbar_kws={'label': 'Significant (1=Yes, 0=No)'}, ax=ax)
    ax.set_title('Significance Comparison (bw=7d): A-Separate vs B-Joint', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Outcome', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig_comparison_significance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show() # 推送至右侧Plots
    plt.close()
    print(f"[SAVE] fig_comparison_significance_heatmap.png")

print(f"\n{'='*60}")
print(f"✅ 三种方案对比完成! Output: {OUTPUT_DIR}")
print(f"{'='*60}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")