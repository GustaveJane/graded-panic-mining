##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向C补充：群体级Tier分布追踪
================================
不追踪个体用户，只看"每天发帖的用户群体中，各tier占比怎么变"
解决"沉默≠不恐慌"的方法论缺陷
VS Code右侧Plots可视化适配版
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 全局绘图配置，解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = '/root/autodl-tmp/group_event_study_v3d/user_transition/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVENT_DATES = ['2020-01-18', '2020-01-19', '2020-01-20', '2020-01-21',
               '2020-01-22', '2020-01-23', '2020-01-24', '2020-01-25']

# %% 1. 读取分层日度数据
# 简化版：直接读取daily_panic_by_tier.csv计算每天发帖群体的tier分布
df = pd.read_csv('/root/autodl-tmp/group_event_study_v3d/daily_panic_by_tier.csv')
df['date'] = pd.to_datetime(df['date_only'])

print("=" * 60)
print("方向C补充: 群体级Tier分布追踪")
print("=" * 60)

# %% 2. 计算事件窗口每日高恐慌群体占比
results = []
for date_str in EVENT_DATES:
    date = pd.to_datetime(date_str)
    sub = df[df['date'] == date]
    if len(sub) == 0:
        continue

    total = sub['post_count'].sum()
    tier3_4 = sub[sub['gmm_tier'].isin([3, 4])]['post_count'].sum()
    tier3_4_pct = tier3_4 / total * 100 if total > 0 else 0

    results.append({
        'date': date_str,
        'total_posts': total,
        'tier3_4_posts': tier3_4,
        'tier3_4_pct': round(tier3_4_pct, 4)
    })
    print(f"{date_str}: 总帖={total:,}, Tier3+4={tier3_4:,}, 占比={tier3_4_pct:.2f}%")

results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUTPUT_DIR}group_level_tier_distribution.csv', index=False, encoding='utf-8-sig')

# %% 3. 绘图模块，图表自动渲染至VS Code右侧Plots面板
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(results_df['date'], results_df['tier3_4_pct'], 'o-', color='#C0392B', linewidth=3, markersize=10)
ax.axvline('2020-01-20', color='orange', linestyle='--', linewidth=2, label='Broadcast (Jan 20)')
ax.axvline('2020-01-23', color='red', linestyle='--', linewidth=2, label='Lockdown (Jan 23)')
ax.set_xlabel('Date', fontsize=13)
ax.set_ylabel('Tier3+4 Percentage (%)', fontsize=13)
ax.set_title('Group-Level Panic Distribution: Daily Poster Tier3+4 %', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig_group_level_tier_distribution.png', dpi=300, bbox_inches='tight')
plt.show() # 推送至VS Code右侧Plots可视化面板
plt.close()

print(f"\n[SAVE] group_level_tier_distribution.csv")
print(f"[SAVE] fig_group_level_tier_distribution.png")