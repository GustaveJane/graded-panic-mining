#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案B：标准版 — 全量行为基线 + 封城前COVID≥1 + 常态基线≥3
"""

import matplotlib

matplotlib.use('module://backend_interagg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import defaultdict

# ==================== 配置 ====================
DATA_DIR = '/root/autodl-tmp/'
OUTPUT_DIR = '/root/autodl-tmp/bertopic_analysis_v7b/planB_standard/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_LIST = [
    'temp19.csv', 'temp1.csv', 'temp2.csv', 'temp3.csv',
    'temp4.csv', 'temp5.csv', 'temp6.csv', 'temp7.csv',
    'temp8.csv', 'temp9.csv', 'temp10.csv', 'temp11.csv', 'temp12.csv'
]

LOCKDOWN = pd.Timestamp('2020-01-23')
CHUNK_SIZE = 100000

GMM_THRESHOLDS = [0.4339, 0.6174, 0.8155, 0.9584]
TIER_MAP = {0: 'low', 1: 'low', 2: 'medium', 3: 'high', 4: 'high'}
TIER_COLORS = {'low': '#85CDCA', 'medium': '#E8A87C', 'high': '#C97B63'}

print(f"[{datetime.now().strftime('%H:%M:%S')}] 方案B：标准版分析开始...")

# ==================== 第一步：全量数据提取用户基线 ====================
print("\n【Step 1】全量数据提取三层基线...")

user_base = defaultdict(lambda: {
    'total_posts': 0,
    'unique_months': set(),
    'pre_posts': 0,
    'pre_covid': 0,
    'pre_noncovid': 0,
    'has_panic': False,
})

for fname in FILE_LIST:
    fp = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fp):
        continue
    print(f"  扫描 {fname}...")

    for chunk in pd.read_csv(fp, chunksize=CHUNK_SIZE, on_bad_lines='skip', low_memory=False):
        uid_col = None
        for c in ['user_id', 'uid', 'userId', 'userid']:
            if c in chunk.columns:
                uid_col = c
                break
        if uid_col is None or 'created_at' not in chunk.columns:
            continue

        chunk['date'] = pd.to_datetime(chunk['created_at'], errors='coerce')
        chunk = chunk[chunk['date'].notna()]

        has_covid = 'covid_related' in chunk.columns
        has_panic = 'panic_prob' in chunk.columns

        for _, row in chunk.iterrows():
            uid = str(row[uid_col])
            d = row['date']
            month = d.strftime('%Y-%m')

            s = user_base[uid]
            s['total_posts'] += 1
            s['unique_months'].add(month)

            # 封城前
            if d < LOCKDOWN:
                s['pre_posts'] += 1
                if has_covid:
                    is_covid = pd.to_numeric(row.get('covid_related', 0), errors='coerce') == 1
                    if is_covid:
                        s['pre_covid'] += 1
                    else:
                        s['pre_noncovid'] += 1

            if has_panic and pd.notna(pd.to_numeric(row.get('panic_prob'), errors='coerce')):
                s['has_panic'] = True

# 转为DataFrame
base_records = []
for uid, s in user_base.items():
    base_records.append({
        'user_id': uid,
        'total_posts': s['total_posts'],
        'active_months': len(s['unique_months']),
        'pre_posts': s['pre_posts'],
        'pre_covid': s['pre_covid'],
        'pre_noncovid': s['pre_noncovid'],
        'has_panic': s['has_panic'],
    })

base_df = pd.DataFrame(base_records)
print(f"\n  全量用户: {len(base_df):,}")
print(f"  有panic_prob: {base_df['has_panic'].sum():,}")

# ==================== 第二步：方案B筛选 ====================
print("\n【Step 2】方案B筛选...")

# B: 全量≥10 + 活跃≥2个月 + 封城前COVID≥1 + 封城前非COVID≥3
planB = base_df[
    (base_df['total_posts'] >= 10) &
    (base_df['active_months'] >= 2) &
    (base_df['pre_covid'] >= 1) &
    (base_df['pre_noncovid'] >= 3)
    ]

print(f"  方案B有效用户: {len(planB):,} 人")
print(f"    全量发帖均值: {planB['total_posts'].mean():.1f}")
print(f"    活跃月份均值: {planB['active_months'].mean():.1f}")
print(f"    封城前COVID均值: {planB['pre_covid'].mean():.1f}")
print(f"    封城前非COVID均值: {planB['pre_noncovid'].mean():.1f}")

planB_uids = set(planB['user_id'])
print(f"    唯一UID数: {len(planB_uids):,}")

# ==================== 第三步：提取方案B用户的COVID帖子（原始数据） ====================
print("\n【Step 3】提取方案B用户的COVID帖子...")

covid_posts = []
for fname in FILE_LIST:
    fp = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fp):
        continue
    print(f"  提取 {fname}...")

    for chunk in pd.read_csv(fp, chunksize=CHUNK_SIZE, on_bad_lines='skip', low_memory=False):
        uid_col = None
        for c in ['user_id', 'uid', 'userId', 'userid']:
            if c in chunk.columns:
                uid_col = c
                break
        if uid_col is None:
            continue

        chunk[uid_col] = chunk[uid_col].astype(str)
        chunk = chunk[chunk[uid_col].isin(planB_uids)]
        if len(chunk) == 0:
            continue

        if 'covid_related' not in chunk.columns or 'panic_prob' not in chunk.columns:
            continue

        chunk['covid_related'] = pd.to_numeric(chunk['covid_related'], errors='coerce').fillna(0)
        chunk['panic_prob'] = pd.to_numeric(chunk['panic_prob'], errors='coerce')

        covid_chunk = chunk[(chunk['covid_related'] == 1) & (chunk['panic_prob'].notna())].copy()
        if len(covid_chunk) == 0:
            continue

        covid_chunk['date'] = pd.to_datetime(covid_chunk['created_at'], errors='coerce')
        covid_chunk = covid_chunk[covid_chunk['date'].notna()]
        covid_posts.append(covid_chunk[[uid_col, 'date', 'panic_prob', 'content']].rename(columns={uid_col: 'user_id'}))

if len(covid_posts) == 0:
    raise ValueError("未找到COVID帖子，请检查数据")

df = pd.concat(covid_posts, ignore_index=True)
df = df.sort_values(['user_id', 'date'])
print(f"  COVID帖子: {len(df):,} 条")
print(f"  覆盖用户: {df['user_id'].nunique():,}")

# ==================== 第四步：GMM分层 + 计算变化量 ====================
print("\n【Step 4】GMM分层，计算变化量...")


def assign_gmm_tier(p):
    if pd.isna(p):
        return 0
    for i, t in enumerate(GMM_THRESHOLDS):
        if p < t:
            return i
    return 4


df['gmm_tier'] = df['panic_prob'].apply(assign_gmm_tier)
df['tier_label'] = df['gmm_tier'].map(TIER_MAP)

# 检查分层分布
tier_dist = df['tier_label'].value_counts()
print(f"  帖子分层: {tier_dist.to_dict()}")


# 用户级变化
def calc_change(uid_df):
    uid_df = uid_df.sort_values('date')
    pre = uid_df[uid_df['date'] < LOCKDOWN]
    post = uid_df[uid_df['date'] >= LOCKDOWN]

    if len(pre) < 1 or len(post) < 1:
        return pd.Series({
            'tier': uid_df['tier_label'].mode()[0] if len(uid_df) > 0 else np.nan,
            'pre_panic': np.nan, 'post_panic': np.nan, 'delta': np.nan,
            'pre_posts': len(pre), 'post_posts': len(post), 'total_posts': len(uid_df),
        })

    return pd.Series({
        'tier': uid_df['tier_label'].mode()[0],
        'pre_panic': pre['panic_prob'].mean(),
        'post_panic': post['panic_prob'].mean(),
        'delta': post['panic_prob'].mean() - pre['panic_prob'].mean(),
        'pre_posts': len(pre),
        'post_posts': len(post),
        'total_posts': len(uid_df),
    })


user_df = df.groupby('user_id').apply(calc_change).reset_index()
user_df = user_df[user_df['delta'].notna()]
print(f"  可计算变化量: {len(user_df):,} 人")

# 检查用户级分层
user_tier_dist = user_df['tier'].value_counts()
print(f"  用户分层: {user_tier_dist.to_dict()}")

# 如果还是全部low，用三分位数
if len(user_tier_dist) == 1 and 'low' in user_tier_dist.index:
    print("\n  ⚠️ GMM阈值在此样本失效，启用数据驱动三分位数...")
    q33 = user_df['pre_panic'].quantile(0.33)
    q66 = user_df['pre_panic'].quantile(0.66)
    print(f"     33%={q33:.4f}, 66%={q66:.4f}")


    def quantile_tier(p):
        if p <= q33:
            return 'low'
        elif p <= q66:
            return 'medium'
        else:
            return 'high'


    user_df['tier'] = user_df['pre_panic'].apply(quantile_tier)
    print(f"     重分层: {user_df['tier'].value_counts().to_dict()}")

# ==================== 第五步：分层描述统计 ====================
print("\n【Step 5】分层描述统计...")

summary = []
for tier in ['low', 'medium', 'high']:
    subset = user_df[user_df['tier'] == tier]
    if len(subset) == 0:
        continue
    summary.append({
        'tier': tier,
        'n': len(subset),
        'pre_panic': subset['pre_panic'].mean(),
        'post_panic': subset['post_panic'].mean(),
        'delta': subset['delta'].mean(),
        'median_delta': subset['delta'].median(),
        'std_delta': subset['delta'].std(),
        'rise_pct': (subset['delta'] > 0).mean() * 100,
        'drop_pct': (subset['delta'] < 0).mean() * 100,
        'avg_pre_posts': subset['pre_posts'].mean(),
        'avg_post_posts': subset['post_posts'].mean(),
    })

summary_df = pd.DataFrame(summary)
print("\n" + "=" * 70)
print("【方案B：分层广播效应】")
print("=" * 70)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'planB_summary.csv'), index=False, encoding='utf-8-sig')

# ==================== 第六步：可视化 ====================
print("\n【Step 6】可视化...")

available_tiers = [t for t in ['low', 'medium', 'high'] if t in user_df['tier'].values]

if len(available_tiers) > 0 and len(user_df) >= 10:
    # 图1：变化量分布
    fig, ax = plt.subplots(figsize=(12, 6))
    data_to_plot = [user_df[user_df['tier'] == t]['delta'].values for t in available_tiers]
    bp = ax.boxplot(data_to_plot, tick_labels=[t.capitalize() for t in available_tiers],
                    patch_artist=True, widths=0.5)
    for patch, tier in zip(bp['boxes'], available_tiers):
        patch.set_facecolor(TIER_COLORS[tier])
        patch.set_alpha(0.7)

    for i, tier in enumerate(available_tiers):
        y = user_df[user_df['tier'] == tier]['delta'].values
        x = np.random.normal(i + 1, 0.08, size=len(y))
        ax.scatter(x, y, alpha=0.15, s=12, color='black')
        mean_val = user_df[user_df['tier'] == tier]['delta'].mean()
        ax.text(i + 1, mean_val + 0.005, f'μ={mean_val:+.4f}', ha='center', fontsize=11,
                fontweight='bold', color=TIER_COLORS[tier])

    ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
    ax.set_ylabel('Delta Panic (Post - Pre)', fontsize=12)
    ax.set_xlabel('Tier', fontsize=12)
    ax.set_title(
        f'Plan B: Broadcast Effect by Tier (n={len(user_df)})\nStandard: total≥10, months≥2, pre-COVID≥1, pre-nonCOVID≥3',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'P1_planB_delta.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  图1已保存")

    # 图2：前后连线
    fig, ax = plt.subplots(figsize=(10, 6))
    for tier in available_tiers:
        subset = user_df[user_df['tier'] == tier]
        pre_mean = subset['pre_panic'].mean()
        post_mean = subset['post_panic'].mean()
        ax.plot([0, 1], [pre_mean, post_mean], 'o-', color=TIER_COLORS[tier],
                linewidth=3, markersize=10, label=f'{tier.capitalize()} (n={len(subset)})')
        ax.annotate('', xy=(1, post_mean), xytext=(0, pre_mean),
                    arrowprops=dict(arrowstyle='->', color=TIER_COLORS[tier], lw=2, alpha=0.5))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-Lockdown', 'Post-Lockdown'], fontsize=12)
    ax.set_ylabel('Mean Panic Probability', fontsize=12)
    ax.set_title('Plan B: Pre/Post Panic by Tier', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'P2_planB_prepost.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  图2已保存")

    # 图3：基线vs变化
    fig, ax = plt.subplots(figsize=(10, 7))
    for tier in available_tiers:
        subset = user_df[user_df['tier'] == tier]
        if len(subset) > 2:
            ax.scatter(subset['pre_panic'], subset['delta'],
                       c=TIER_COLORS[tier], alpha=0.5, s=40, edgecolors='none', label=tier.capitalize())
            z = np.polyfit(subset['pre_panic'], subset['delta'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['pre_panic'].min(), subset['pre_panic'].max(), 100)
            ax.plot(x_line, p(x_line), '--', color=TIER_COLORS[tier], linewidth=2, alpha=0.8)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Pre-Lockdown Panic Baseline', fontsize=12)
    ax.set_ylabel('Delta Panic (Post - Pre)', fontsize=12)
    ax.set_title('Plan B: Baseline vs. Change', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'P3_planB_baseline.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  图3已保存")

user_df.to_csv(os.path.join(OUTPUT_DIR, 'planB_users.csv'), index=False, encoding='utf-8-sig')

print(f"\n{'=' * 70}")
print("✅ 方案B分析完成！")
print(f"{'=' * 70}")
print(f"筛选条件:")
print(f"  - 全量发帖≥10（行为稳定）")
print(f"  - 活跃≥2个月（持续存在）")
print(f"  - 封城前COVID≥1（能算pre）")
print(f"  - 封城前非COVID≥3（常态基线）")
print(f"有效用户: {len(user_df):,} 人")
if len(summary_df) > 0:
    for _, row in summary_df.iterrows():
        print(f"  {row['tier'].upper()}: n={row['n']}, ΔPanic={row['delta']:+.4f}, 上升={row['rise_pct']:.1f}%")
print(f"\n输出: {OUTPUT_DIR}")