#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
师姐方向：封城广播效应 + 数据交叉分析
修复版：兼容 v6b 的 0-4 tier + 支持 user_id + 全英文标签
"""

import matplotlib

matplotlib.use('module://backend_interagg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ==================== 配置 ====================
INPUT_FILE = '/root/autodl-tmp/bertopic_analysis_v7b/sampled_manual_framework.csv'
OUTPUT_DIR = '/root/autodl-tmp/bertopic_analysis_v7b/weekly_report_figs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCKDOWN_DATE = pd.Timestamp('2020-01-23')

# v6b tier 映射：0,1=Mild->low, 2=Moderate->medium, 3,4=Severe->high
TIER_MAP = {0: 'low', 1: 'low', 2: 'medium', 3: 'high', 4: 'high'}
TIER_ORDER = ['low', 'medium', 'high']
TIER_COLORS = {'low': '#85CDCA', 'medium': '#E8A87C', 'high': '#C97B63'}

NATURE_COLORS = {
    'A_original_emotion': '#C97B63',
    'B_news_forward': '#3A506B',
    'C_help_support': '#85CDCA',
    'D_daily_misc': '#E8A87C'
}

# ==================== 1. 数据加载 ====================
print("=" * 60)
print("[1] Loading data...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Total samples: {len(df):,}")

# 日期处理
df['date'] = pd.to_datetime(df['date'])
df['days_to_lockdown'] = (df['date'] - LOCKDOWN_DATE).dt.days


# 时期划分
def classify_period(d):
    if d < -7:
        return 'pre_early'
    elif d < 0:
        return 'pre_imminent'
    elif d == 0:
        return 'd_day'
    elif d <= 7:
        return 'post_week1'
    elif d <= 14:
        return 'post_week2'
    elif d <= 30:
        return 'post_month1'
    else:
        return 'post_later'


df['period'] = df['days_to_lockdown'].apply(classify_period)
PERIOD_ORDER = ['pre_early', 'pre_imminent', 'd_day', 'post_week1', 'post_week2', 'post_month1', 'post_later']
PERIOD_LABELS = ['>7d Before', '7d Before', 'D-Day', 'Week 1', 'Week 2', 'Month 1', '>1 Month']

# 修复 tier：v6b 输出是 0-4 整数，映射为 low/medium/high
if df['gmm_tier'].dtype in [np.int64, np.float64, 'int32', 'float32']:
    print(f"Detected numeric gmm_tier (0-4), mapping to low/medium/high...")
    df['gmm_tier'] = df['gmm_tier'].map(TIER_MAP)
else:
    print(f"gmm_tier type: {df['gmm_tier'].dtype}, values: {df['gmm_tier'].unique()}")

# 检测 user_id
uid_candidates = ['user_id', 'uid', 'userId', 'userid']
USER_COL = None
for c in uid_candidates:
    if c in df.columns:
        USER_COL = c
        print(f"Found user column: {USER_COL}")
        break
if USER_COL is None:
    print("No user ID found, skipping individual-level analyses")

# ==================== 2. 图1：封城广播效应 ====================
print("\n[2] Generating P0-1: Broadcast Effect...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 子图1：每日恐慌均值（RDD风格）---
ax1 = axes[0, 0]
daily_panic = df.groupby(['days_to_lockdown', 'gmm_tier'])['panic_prob'].mean().unstack()
for tier in TIER_ORDER:
    if tier in daily_panic.columns:
        ax1.plot(daily_panic.index, daily_panic[tier], label=tier,
                 color=TIER_COLORS[tier], linewidth=2)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Days to Lockdown', fontsize=11)
ax1.set_ylabel('Mean Panic Probability', fontsize=11)
ax1.set_title('P0-1a: Broadcast Effect — Daily Panic by Tier', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_xlim(-30, 60)

# --- 子图2：时期聚合柱状图 ---
ax2 = axes[0, 1]
period_panic = df.groupby(['period', 'gmm_tier'])['panic_prob'].mean().unstack()
period_panic = period_panic.reindex(PERIOD_ORDER)
period_panic = period_panic[[c for c in TIER_ORDER if c in period_panic.columns]]
period_panic.plot(kind='bar', ax=ax2, color=[TIER_COLORS[c] for c in period_panic.columns], width=0.8)
ax2.set_xticklabels(PERIOD_LABELS, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('Mean Panic Probability', fontsize=11)
ax2.set_title('P0-1b: Panic by Period & Tier', fontsize=13, fontweight='bold')
ax2.legend(title='GMM Tier', fontsize=9)

# --- 子图3：封城前后变化量（ΔPanic）---
ax3 = axes[1, 0]
pre = df[df['period'] == 'pre_imminent'].groupby('gmm_tier')['panic_prob'].mean()
post = df[df['period'] == 'post_week1'].groupby('gmm_tier')['panic_prob'].mean()
delta = (post - pre).reindex(TIER_ORDER)
colors_delta = ['#C97B63' if v > 0 else '#85CDCA' for v in delta.values]
bars = ax3.bar(range(len(delta)), delta.values, color=colors_delta, edgecolor='black', alpha=0.8)
ax3.set_xticks(range(len(delta)))
ax3.set_xticklabels(['High', 'Medium', 'Low'], fontsize=10)
ax3.set_ylabel('Delta Panic (PostW1 - Pre7d)', fontsize=11)
ax3.set_title('P0-1c: Broadcast Shock Magnitude by Tier', fontsize=13, fontweight='bold')
ax3.axhline(y=0, color='black', linewidth=1)
for bar, val in zip(bars, delta.values):
    offset = 0.005 if val > 0 else -0.01
    va = 'bottom' if val > 0 else 'top'
    ax3.text(bar.get_x() + bar.get_width() / 2, val + offset,
             f'{val:+.3f}', ha='center', va=va, fontsize=10, fontweight='bold')

# --- 子图4：信息类型占比变化 ---
ax4 = axes[1, 1]
nature_period = pd.crosstab(df['period'], df['content_nature'], normalize='index') * 100
nature_period = nature_period.reindex(PERIOD_ORDER)
nature_cols = [c for c in ['A_original_emotion', 'B_news_forward', 'C_help_support', 'D_daily_misc'] if
               c in nature_period.columns]
nature_period[nature_cols].plot(kind='bar', stacked=True, ax=ax4,
                                color=[NATURE_COLORS.get(c, '#999') for c in nature_cols], width=0.8)
ax4.set_xticklabels(PERIOD_LABELS, rotation=30, ha='right', fontsize=9)
ax4.set_ylabel('Percentage (%)', fontsize=11)
ax4.set_title('P0-1d: Content Nature Shift (Broadcast Response)', fontsize=13, fontweight='bold')
ax4.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.02, 1))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'P0_1_broadcast_effect.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Done")

# ==================== 3. 图2：主题迁移流向图 ====================
print("\n[3] Generating P0-2: Topic Migration...")

pre_topics = df[df['days_to_lockdown'] < 0]['manual_topic'].value_counts(normalize=True)
post_topics = df[df['days_to_lockdown'] >= 0]['manual_topic'].value_counts(normalize=True)
topic_shift = pd.DataFrame({'pre': pre_topics, 'post': post_topics}).fillna(0)
topic_shift['delta'] = topic_shift['post'] - topic_shift['pre']
topic_shift = topic_shift.sort_values('delta', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

ax = axes[0]
colors_t = ['#C97B63' if v > 0 else '#3A506B' for v in topic_shift['delta']]
bars = ax.barh(range(len(topic_shift)), topic_shift['delta'] * 100, color=colors_t, edgecolor='black', alpha=0.85)
ax.set_yticks(range(len(topic_shift)))
ax.set_yticklabels(topic_shift.index, fontsize=9)
ax.set_xlabel('Delta Share (Post - Pre) %', fontsize=11)
ax.set_title('P0-2a: Topic Migration After Broadcast', fontsize=13, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlim(-15, 15)
for bar, val in zip(bars, topic_shift['delta'] * 100):
    ax.text(val + 0.5 if val > 0 else val - 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val:+.1f}%', ha='left' if val > 0 else 'right', va='center', fontsize=9)

ax = axes[1]
x = np.arange(len(topic_shift))
width = 0.35
ax.bar(x - width / 2, topic_shift['pre'] * 100, width, label='Pre-Lockdown', color='#3A506B', alpha=0.8)
ax.bar(x + width / 2, topic_shift['post'] * 100, width, label='Post-Lockdown', color='#C97B63', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(topic_shift.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Share (%)', fontsize=11)
ax.set_title('P0-2b: Topic Distribution Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'P0_2_topic_migration.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Done")

# ==================== 4. 图3：双轴时间序列 ====================
print("\n[4] Generating P0-3: Dual-Axis Timeline...")

df['week'] = df['date'].dt.to_period('W').dt.start_time
weekly_nature = pd.crosstab(df['week'], df['content_nature'], normalize='index') * 100

fig, ax1 = plt.subplots(figsize=(14, 6))

nature_labels = {
    'A_original_emotion': 'Original Emotion',
    'B_news_forward': 'News Forward',
    'C_help_support': 'Mutual Help',
    'D_daily_misc': 'Daily Misc'
}
for nature in ['A_original_emotion', 'B_news_forward', 'C_help_support']:
    if nature in weekly_nature.columns:
        ax1.plot(weekly_nature.index, weekly_nature[nature],
                 label=nature_labels.get(nature, nature),
                 color=NATURE_COLORS[nature], linewidth=2.5, marker='o', markersize=4)

ax1.axvline(x=LOCKDOWN_DATE, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(LOCKDOWN_DATE, ax1.get_ylim()[1] * 0.95, ' Lockdown', fontsize=11, color='black', fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Content Nature Share (%)', fontsize=12, color='#333')
ax1.tick_params(axis='y', labelcolor='#333')
ax1.legend(loc='upper left', fontsize=10)

ax2 = ax1.twinx()
weekly_panic = df.groupby('week')['panic_prob'].mean()
ax2.plot(weekly_panic.index, weekly_panic.values, color='#E63946', linewidth=3, linestyle='--', label='Avg Panic Prob',
         alpha=0.8)
ax2.set_ylabel('Mean Panic Probability', fontsize=12, color='#E63946')
ax2.tick_params(axis='y', labelcolor='#E63946')
ax2.legend(loc='upper right', fontsize=10)

plt.title('P0-3: Information Type vs. Panic Over Time', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'P0_3_dual_axis_timeline.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Done")

# ==================== 5. 图4：交叉热力图 ====================
print("\n[5] Generating P0-4: Cross Heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cross = pd.crosstab(df['manual_topic'], df['gmm_tier'])
cross = cross[[c for c in TIER_ORDER if c in cross.columns]]
sns.heatmap(cross, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0], linewidths=0.5)
axes[0].set_title('P0-4a: Topic x Tier (Count)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('GMM Tier', fontsize=11)
axes[0].set_ylabel('Manual Topic', fontsize=11)

cross_pct = pd.crosstab(df['manual_topic'], df['gmm_tier'], normalize='index') * 100
cross_pct = cross_pct[[c for c in TIER_ORDER if c in cross_pct.columns]]
sns.heatmap(cross_pct, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[1], linewidths=0.5)
axes[1].set_title('P0-4b: Topic x Tier (Row %)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('GMM Tier', fontsize=11)
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'P0_4_cross_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Done")

# ==================== 6. 图5：个体轨迹（需要 user_id） ====================
print("\n[6] Generating P1-5: Individual Trajectories...")

if USER_COL:
    user_counts = df[USER_COL].value_counts()
    valid_users = user_counts[user_counts >= 3].index
    pre_users = set(df[df['days_to_lockdown'] < 0][USER_COL].unique())
    post_users = set(df[df['days_to_lockdown'] >= 0][USER_COL].unique())
    cross_users = list(pre_users & post_users & set(valid_users))

    if len(cross_users) >= 10:
        sampled_users = np.random.choice(cross_users, size=min(25, len(cross_users)), replace=False)
        user_df = df[df[USER_COL].isin(sampled_users)].copy()

        fig, ax = plt.subplots(figsize=(14, 7))
        for uid in sampled_users[:20]:
            u_data = user_df[user_df[USER_COL] == uid].sort_values('date')
            if len(u_data) >= 2:
                tier = u_data['gmm_tier'].mode()[0]
                ax.plot(u_data['days_to_lockdown'], u_data['panic_prob'],
                        color=TIER_COLORS.get(tier, '#999'), alpha=0.35, linewidth=1.5)

        mean_panic = df.groupby('days_to_lockdown')['panic_prob'].mean()
        ax.plot(mean_panic.index, mean_panic.values, color='black', linewidth=3,
                linestyle='--', label='Population Mean', zorder=10)
        ax.axvline(x=0, color='red', linestyle='-', linewidth=2, alpha=0.7, label='Lockdown')
        ax.set_xlabel('Days to Lockdown', fontsize=12)
        ax.set_ylabel('Panic Probability', fontsize=12)
        ax.set_title('P1-5: Individual Trajectories After Broadcast (n=20 sampled)', fontsize=14, fontweight='bold')
        ax.set_xlim(-30, 60)

        from matplotlib.lines import Line2D

        legend_elements = [Line2D([0], [0], color=TIER_COLORS['high'], lw=2, label='High Tier'),
                           Line2D([0], [0], color=TIER_COLORS['medium'], lw=2, label='Medium Tier'),
                           Line2D([0], [0], color=TIER_COLORS['low'], lw=2, label='Low Tier')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'P1_5_individual_trajectories.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print("Done")
    else:
        print(f"Only {len(cross_users)} users cross lockdown, skipping")
else:
    print("No user_id, skipping")

# ==================== 7. 图6：认知更新散点图 ====================
print("\n[7] Generating P1-6: Cognitive Update...")

if USER_COL:
    user_stats = df.groupby(USER_COL).agg({
        'panic_prob': 'mean',
        'manual_topic': lambda x: (x == 'P6_info_seek').sum() / len(x) * 100,
        'gmm_tier': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'medium'
    }).rename(columns={'manual_topic': 'info_seek_pct'})
else:
    user_stats = df.groupby('week').agg({
        'panic_prob': 'mean',
        'manual_topic': lambda x: (x == 'P6_info_seek').sum() / len(x) * 100,
    }).rename(columns={'manual_topic': 'info_seek_pct'})
    user_stats['gmm_tier'] = 'all'

fig, ax = plt.subplots(figsize=(10, 7))
for tier in user_stats['gmm_tier'].unique():
    subset = user_stats[user_stats['gmm_tier'] == tier]
    color = TIER_COLORS.get(tier, '#666')
    ax.scatter(subset['info_seek_pct'], subset['panic_prob'],
               c=color, alpha=0.6, s=80, edgecolors='black', linewidth=0.5,
               label=f'{tier} (n={len(subset)})')

# 回归线
if len(user_stats) > 2:
    z = np.polyfit(user_stats['info_seek_pct'], user_stats['panic_prob'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(user_stats['info_seek_pct'].min(), user_stats['info_seek_pct'].max(), 100)
    r = np.corrcoef(user_stats['info_seek_pct'], user_stats['panic_prob'])[0, 1]
    ax.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2, label=f'Linear fit (r={r:.3f})')

ax.set_xlabel('Information Seeking (P6) Share (%)', fontsize=12)
ax.set_ylabel('Mean Panic Probability', fontsize=12)
ax.set_title('P1-6: Does Info-Seeking Reduce Panic?', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'P1_6_cognitive_update.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Done")

# ==================== 8. 图7：阈值触发箱线图（需要 user_id） ====================
print("\n[8] Generating P1-7: Threshold Trigger...")

if USER_COL:
    user_features = df.groupby(USER_COL).agg({
        'panic_prob': 'mean',
        'content_nature': 'nunique',
        'date': 'count',
        'manual_topic': lambda x: (x == 'B_news_forward').sum() / len(x) * 100,
        'gmm_tier': lambda x: x.mode()[0]
    }).rename(columns={'content_nature': 'topic_diversity', 'date': 'post_volume', 'manual_topic': 'news_dependency'})
    user_features = user_features[user_features['post_volume'] >= 3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    features = [('post_volume', 'Post Volume'), ('topic_diversity', 'Topic Diversity'),
                ('news_dependency', 'News Dependency (%)')]

    for ax, (feat, title) in zip(axes, features):
        sns.boxplot(data=user_features, x='gmm_tier', y=feat, order=TIER_ORDER,
                    palette=[TIER_COLORS[t] for t in TIER_ORDER], ax=ax)
        ax.set_title(f'P1-7: {title} by Tier', fontsize=12, fontweight='bold')
        ax.set_xlabel('GMM Tier', fontsize=11)
        ax.set_ylabel(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'P1_7_threshold_trigger.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("Done")
else:
    print("No user_id, skipping")

# ==================== 9. 汇总统计（修复版） ====================
print("\n[9] Summary Statistics...")

summary = []
for tier in TIER_ORDER:
    pre = df[(df['gmm_tier'] == tier) & (df['period'] == 'pre_imminent')]['panic_prob'].mean()
    post = df[(df['gmm_tier'] == tier) & (df['period'] == 'post_week1')]['panic_prob'].mean()
    summary.append({
        'Tier': tier,
        'Pre_Panic': round(pre, 4) if pd.notna(pre) else None,
        'Post_Panic': round(post, 4) if pd.notna(post) else None,
        'Delta': round(post - pre, 4) if pd.notna(pre) and pd.notna(post) else None,
        'Pct_Change': round((post - pre) / pre * 100, 1) if pd.notna(pre) and pre > 0 else None
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'broadcast_effect_summary.csv'), index=False, encoding='utf-8-sig')
print("\nSummary:")
print(summary_df.to_string(index=False))

# 周级相关性
if 'P6_info_seek' in df['manual_topic'].values:
    weekly_info = df.groupby('week').agg({
        'panic_prob': 'mean',
        'manual_topic': lambda x: (x == 'P6_info_seek').sum() / len(x)
    })
    if len(weekly_info) > 2:
        corr = weekly_info.corr().iloc[0, 1]
        print(f"\nInfo-Seeking(P6) vs Panic correlation: r = {corr:.4f}")

print(f"\n{'=' * 60}")
print(f"All done! Output: {OUTPUT_DIR}")
print(f"{'=' * 60}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
