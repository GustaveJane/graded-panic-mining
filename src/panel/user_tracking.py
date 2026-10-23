#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户追踪 v4.2 - pre>=10, post>=1
"""

import os
import sys
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from datetime import datetime
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 配置
# ==============================================================================

DATA_DIR = '/root/autodl-tmp/'
OUTPUT_DIR = '/root/autodl-tmp/output_v4_2_pre10_post1/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CUTOFF_PRE_END = pd.Timestamp('2020-01-22 23:59:59')
CUTOFF_POST_START = pd.Timestamp('2020-01-23')
CUTOFF_POST_END = pd.Timestamp('2020-03-24 23:59:59')

MIN_PRE_POSTS = 10     # 封城前>=10帖
MIN_POST_POSTS = 1     # 封城后>=1帖
MIN_PRE_PANIC = 0.01   # 封城前平均恐慌>0.01
RANDOM_SEED = 42

MONTH_FILES = [
    ('2019-12', 'temp19.csv'),
    ('2020-01', 'temp1.csv'),
    ('2020-02', 'temp2.csv'),
    ('2020-03', 'temp3.csv'),
]

CHUNK_SIZE = 1_000_000
N_WORKERS = min(4, cpu_count())

print("=" * 70)
print(f"用户追踪 v4.2 | pre>={MIN_PRE_POSTS}帖 | post>={MIN_POST_POSTS}帖 | pre_panic>{MIN_PRE_PANIC}")
print(f"Workers: {N_WORKERS}")
print("=" * 70)

# ==============================================================================
# 处理单个月份
# ==============================================================================

def process_month(args):
    month_key, filename, data_dir = args
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        return {}, 0, f"{month_key}: 不存在"

    file_size = os.path.getsize(filepath) / (1024**3)

    pre_posts = defaultdict(int)
    post_posts = defaultdict(int)
    pre_panic = defaultdict(float)
    post_panic = defaultdict(float)
    pre_covid = defaultdict(int)
    post_covid = defaultdict(int)

    total_rows = 0

    for chunk in pd.read_csv(
        filepath,
        chunksize=CHUNK_SIZE,
        usecols=['user_id', 'created_at', 'panic_prob', 'covid_related'],
        dtype={'user_id': 'str', 'created_at': 'str', 'panic_prob': 'float32', 'covid_related': 'float32'},
        low_memory=False,
    ):
        dates = chunk['created_at'].str[:10]

        if month_key == '2019-12':
            is_pre = pd.Series(True, index=chunk.index)
            is_post = pd.Series(False, index=chunk.index)
        elif month_key == '2020-01':
            is_pre = dates <= '2020-01-22'
            is_post = dates >= '2020-01-23'
        elif month_key == '2020-02':
            is_pre = pd.Series(False, index=chunk.index)
            is_post = pd.Series(True, index=chunk.index)
        elif month_key == '2020-03':
            is_pre = pd.Series(False, index=chunk.index)
            is_post = dates <= '2020-03-24'
        else:
            is_pre = pd.Series(False, index=chunk.index)
            is_post = pd.Series(False, index=chunk.index)

        valid = chunk['panic_prob'].notna() & chunk['user_id'].notna()
        chunk = chunk[valid]
        is_pre = is_pre[valid]
        is_post = is_post[valid]

        if len(chunk) == 0:
            continue

        uids = chunk['user_id'].values
        panic = chunk['panic_prob'].fillna(0).values.astype(np.float32)
        covid = chunk['covid_related'].fillna(0).values.astype(np.int32)
        pre_mask = is_pre.values
        post_mask = is_post.values

        unique_uids = np.unique(uids)
        uid_to_idx = {uid: i for i, uid in enumerate(unique_uids)}
        n = len(unique_uids)

        pre_p_arr = np.zeros(n, dtype=np.int32)
        post_p_arr = np.zeros(n, dtype=np.int32)
        pre_pan_arr = np.zeros(n, dtype=np.float64)
        post_pan_arr = np.zeros(n, dtype=np.float64)
        pre_cov_arr = np.zeros(n, dtype=np.int32)
        post_cov_arr = np.zeros(n, dtype=np.int32)

        for i in range(len(uids)):
            idx = uid_to_idx[uids[i]]
            if pre_mask[i]:
                pre_p_arr[idx] += 1
                pre_pan_arr[idx] += panic[i]
                pre_cov_arr[idx] += (covid[i] == 1)
            elif post_mask[i]:
                post_p_arr[idx] += 1
                post_pan_arr[idx] += panic[i]
                post_cov_arr[idx] += (covid[i] == 1)

        for i, uid in enumerate(unique_uids):
            pre_posts[uid] += pre_p_arr[i]
            post_posts[uid] += post_p_arr[i]
            pre_panic[uid] += pre_pan_arr[i]
            post_panic[uid] += post_pan_arr[i]
            pre_covid[uid] += pre_cov_arr[i]
            post_covid[uid] += post_cov_arr[i]

        total_rows += len(chunk)
        del chunk, uids, panic, covid, pre_mask, post_mask
        del pre_p_arr, post_p_arr, pre_pan_arr, post_pan_arr, pre_cov_arr, post_cov_arr
        gc.collect()

    result = {
        'pre_posts': dict(pre_posts),
        'post_posts': dict(post_posts),
        'pre_panic': dict(pre_panic),
        'post_panic': dict(post_panic),
        'pre_covid': dict(pre_covid),
        'post_covid': dict(post_covid),
    }

    n_users = len(set(list(pre_posts.keys()) + list(post_posts.keys())))
    return result, total_rows, f"{month_key}: {total_rows:,} 条, {n_users:,} 用户 ({file_size:.2f}GB)"

# ==============================================================================
# 第一阶段: 并行处理
# ==============================================================================

print("\n第一阶段: 并行收集统计...")
start_time = datetime.now()

args_list = [(mk, fn, DATA_DIR) for mk, fn in MONTH_FILES]

if N_WORKERS > 1:
    with Pool(N_WORKERS) as pool:
        results = pool.map(process_month, args_list)
else:
    results = [process_month(a) for a in args_list]

global_stats = defaultdict(lambda: {
    'pre_posts': 0, 'post_posts': 0,
    'pre_panic_sum': 0.0, 'post_panic_sum': 0.0,
    'pre_covid_posts': 0, 'post_covid_posts': 0,
})

total_rows_all = 0
for result, rows, msg in results:
    print(f"   ✅ {msg}")
    total_rows_all += rows
    for uid, v in result['pre_posts'].items():
        global_stats[uid]['pre_posts'] += v
        global_stats[uid]['pre_panic_sum'] += result['pre_panic'][uid]
        global_stats[uid]['pre_covid_posts'] += result['pre_covid'][uid]
    for uid, v in result['post_posts'].items():
        global_stats[uid]['post_posts'] += v
        global_stats[uid]['post_panic_sum'] += result['post_panic'][uid]
        global_stats[uid]['post_covid_posts'] += result['post_covid'][uid]

elapsed = (datetime.now() - start_time).total_seconds()
print(f"\n{'='*70}")
print(f"总计: {total_rows_all:,} 条, {len(global_stats):,} 唯一用户")
print(f"耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")

# ==============================================================================
# 第二阶段: 清洗 (pre>=10, post>=1, pre_panic>0.01)
# ==============================================================================

print("\n" + "="*70)
print(f"第二阶段: 清洗 (pre>={MIN_PRE_POSTS}, post>={MIN_POST_POSTS}, pre_panic>{MIN_PRE_PANIC})")
print("="*70)

records = []
for uid, s in global_stats.items():
    records.append({
        'user_id': uid,
        'pre_posts': s['pre_posts'],
        'post_posts': s['post_posts'],
        'pre_panic_sum': s['pre_panic_sum'],
        'post_panic_sum': s['post_panic_sum'],
        'pre_covid_posts': s['pre_covid_posts'],
        'post_covid_posts': s['post_covid_posts'],
    })

df_all = pd.DataFrame(records)
print(f"清洗前: {len(df_all):,} 人")

df_all['pre_avg_panic'] = df_all['pre_panic_sum'] / df_all['pre_posts'].replace(0, np.nan)

mask = (
    (df_all['pre_posts'] >= MIN_PRE_POSTS) &
    (df_all['post_posts'] >= MIN_POST_POSTS) &
    (df_all['pre_avg_panic'] > MIN_PRE_PANIC)
)
df_clean = df_all[mask].copy()

df_clean['post_avg_panic'] = df_clean['post_panic_sum'] / df_clean['post_posts']
df_clean['panic_change'] = df_clean['post_avg_panic'] - df_clean['pre_avg_panic']
df_clean['panic_change_pct'] = df_clean['panic_change'] / df_clean['pre_avg_panic']

print(f"pre>={MIN_PRE_POSTS}, post>={MIN_POST_POSTS}: {len(df_all[(df_all['pre_posts']>=MIN_PRE_POSTS)&(df_all['post_posts']>=MIN_POST_POSTS)]):,} 人")
print(f"再筛选 pre_panic>{MIN_PRE_PANIC}: {len(df_clean):,} 人")
print(f"\n恐慌变化统计:")
print(df_clean['panic_change'].describe())

df_clean.to_csv(os.path.join(OUTPUT_DIR, 'panickers_only.csv'), index=False)

# ==============================================================================
# 第三阶段: 定义恢复轨迹
# ==============================================================================

print("\n第三阶段: 定义恢复轨迹")
print("="*70)

def classify_trajectory(row):
    change = row['panic_change']
    pre = row['pre_avg_panic']
    rel_change = change / pre if pre > 0 else 0

    if rel_change < -0.5:
        return 'Recovery_Strong'
    elif rel_change < -0.1:
        return 'Recovery_Moderate'
    elif abs(rel_change) <= 0.1:
        return 'Stabilized'
    elif rel_change > 0.5:
        return 'Deterioration'
    else:
        return 'Worsening'

df_clean['trajectory'] = df_clean.apply(classify_trajectory, axis=1)

print("\n恢复轨迹分布:")
traj_counts = df_clean['trajectory'].value_counts()
for traj, count in traj_counts.items():
    sub = df_clean[df_clean['trajectory'] == traj]
    print(f"  {traj}: {count:,}人 ({count/len(df_clean):.1%})")
    print(f"    pre: {sub['pre_avg_panic'].mean():.4f} → post: {sub['post_avg_panic'].mean():.4f}")
    print(f"    平均变化: {sub['panic_change'].mean():.4f} ({sub['panic_change_pct'].mean():.1%})")

# ==============================================================================
# 第四阶段: 可视化
# ==============================================================================

print("\n第四阶段: 可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Panic Recovery (pre>={MIN_PRE_POSTS}, post>={MIN_POST_POSTS}) | n={len(df_clean):,}',
             fontsize=12, fontweight='bold')

ax = axes[0, 0]
colors_traj = {'Recovery_Strong': '#2ecc71', 'Recovery_Moderate': '#3498db',
               'Stabilized': '#f39c12', 'Worsening': '#e74c3c', 'Deterioration': '#8e44ad'}
for traj in colors_traj:
    if traj in df_clean['trajectory'].values:
        sub = df_clean[df_clean['trajectory'] == traj]
        ax.scatter(sub['pre_avg_panic'], sub['post_avg_panic'],
                  alpha=0.5, c=colors_traj[traj], s=30, label=traj)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('Pre-Lockdown Panic')
ax.set_ylabel('Post-Lockdown Panic')
ax.set_title('Pre vs Post by Trajectory')
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.hist(df_clean['panic_change'], bins=50, color='teal', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
ax.axvline(df_clean['panic_change'].mean(), color='orange', linestyle='-', linewidth=2,
           label=f'Mean={df_clean["panic_change"].mean():.4f}')
ax.axvline(df_clean['panic_change'].median(), color='purple', linestyle='-.', linewidth=2,
           label=f'Median={df_clean["panic_change"].median():.4f}')
ax.set_xlabel('Panic Change')
ax.set_ylabel('Count')
ax.set_title('Panic Change Distribution')
ax.legend()

ax = axes[1, 0]
traj_order = ['Recovery_Strong', 'Recovery_Moderate', 'Stabilized', 'Worsening', 'Deterioration']
traj_counts_ordered = [traj_counts.get(t, 0) for t in traj_order]
traj_colors = [colors_traj.get(t, 'gray') for t in traj_order]
bars = ax.bar(range(len(traj_order)), traj_counts_ordered, color=traj_colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(traj_order)))
ax.set_xticklabels(traj_order, rotation=15, ha='right')
ax.set_ylabel('Count')
ax.set_title('Trajectory Distribution')
for bar, c in zip(bars, traj_counts_ordered):
    if c > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(traj_counts_ordered)*0.02,
                str(c), ha='center', va='bottom', fontsize=10)

ax = axes[1, 1]
ax.scatter(df_clean['pre_posts'], df_clean['panic_change'], alpha=0.4, c='steelblue', s=20)
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Pre-Lockdown Posts')
ax.set_ylabel('Panic Change')
ax.set_title('Pre Posts vs Panic Change')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_trajectories.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 图1已保存")

# ==============================================================================
# 第五阶段: 回归分析
# ==============================================================================

print("\n第五阶段: 回归分析")
print("="*70)

df_clean['recovered'] = (df_clean['panic_change'] < -0.1).astype(int)

X = sm.add_constant(df_clean[['pre_avg_panic', 'pre_posts', 'post_posts', 'pre_covid_posts']])
y = df_clean['post_avg_panic']
model1 = sm.OLS(y, X).fit()
print(f"\n模型1: post_panic ~ pre_panic + pre_posts + post_posts + pre_covid_posts")
print(f"样本: {len(df_clean):,}, R²: {model1.rsquared:.4f}")
print(model1.summary().tables[1])

X2 = sm.add_constant(df_clean[['pre_avg_panic', 'pre_posts', 'post_posts']])
y2 = df_clean['recovered']
model2 = sm.Logit(y2, X2).fit(disp=0)
print(f"\n模型2: Logit(Recovered) ~ pre_panic + pre_posts + post_posts")
print(f"Pseudo R²: {model2.prsquared:.4f}")
print(model2.summary().tables[1])

df_clean['pre_panic_sq'] = df_clean['pre_avg_panic'] ** 2
X3 = sm.add_constant(df_clean[['pre_avg_panic', 'pre_panic_sq', 'pre_posts', 'post_posts']])
model3 = sm.OLS(y, X3).fit()
print(f"\n模型3: 加入 pre_panic²")
print(f"R²: {model3.rsquared:.4f}")
print(model3.summary().tables[1])

pd.DataFrame({
    'model': ['OLS_post_panic', 'Logit_recovered', 'OLS_nonlinear'],
    'n_obs': [model1.nobs, model2.nobs, model3.nobs],
    'r_squared': [model1.rsquared, model2.prsquared, model3.rsquared],
}).to_csv(os.path.join(OUTPUT_DIR, 'regression_results.csv'), index=False)

# ==============================================================================
# 第六阶段: RDD
# ==============================================================================

print("\n第六阶段: RDD断点回归")
print("="*70)

daily_path = os.path.join(DATA_DIR, 'daily_stats_detail_2019_2020.csv')
if os.path.exists(daily_path):
    daily_df = pd.read_csv(daily_path)
    time_col = 'date' if 'date' in daily_df.columns else 'created_at'
    daily_df['date'] = pd.to_datetime(daily_df[time_col])
    daily_df = daily_df[daily_df['date'] <= CUTOFF_POST_END]
    daily_df['days_from_cutoff'] = (daily_df['date'] - CUTOFF_POST_START).dt.days

    outcome_col = None
    for c in ['panic_rate', 'panic_mean', 'avg_panic', 'mean_panic']:
        if c in daily_df.columns:
            outcome_col = c
            break

    if outcome_col:
        if daily_df[outcome_col].max() > 1:
            print(f"   检测到恐慌率>1，自动除以100")
            daily_df[outcome_col] = daily_df[outcome_col] / 100

        bandwidth = 45
        rdd_data = daily_df[abs(daily_df['days_from_cutoff']) <= bandwidth].copy()

        rdd_data['treatment'] = (rdd_data['days_from_cutoff'] >= 0).astype(int)
        rdd_data['centered'] = rdd_data['days_from_cutoff']
        rdd_data['centered_x_treat'] = rdd_data['centered'] * rdd_data['treatment']

        X_rdd = sm.add_constant(rdd_data[['treatment', 'centered', 'centered_x_treat']])
        y_rdd = rdd_data[outcome_col]
        model_rdd = sm.OLS(y_rdd, X_rdd).fit()

        print(f"\nRDD: 带宽±{bandwidth}天, {len(rdd_data)}天")
        print(f"断点效应: {model_rdd.params['treatment']:.6f} (p={model_rdd.pvalues['treatment']:.4f})")
        print(model_rdd.summary().tables[1])

        fig, ax = plt.subplots(figsize=(12, 6))
        pre_data = rdd_data[rdd_data['days_from_cutoff'] < 0].copy()
        post_data = rdd_data[rdd_data['days_from_cutoff'] >= 0].copy()

        ax.scatter(pre_data['days_from_cutoff'], pre_data[outcome_col], alpha=0.6, c='blue', s=50, label='Pre')
        ax.scatter(post_data['days_from_cutoff'], post_data[outcome_col], alpha=0.6, c='red', s=50, label='Post')

        beta = model_rdd.params
        pre_data['pred'] = beta['const'] + pre_data['centered'] * beta['centered']
        post_data['pred'] = (beta['const'] + beta['treatment'] +
                            post_data['centered'] * beta['centered'] +
                            post_data['centered_x_treat'] * beta['centered_x_treat'])

        pre_sorted = pre_data.sort_values('days_from_cutoff')
        post_sorted = post_data.sort_values('days_from_cutoff')
        ax.plot(pre_sorted['days_from_cutoff'], pre_sorted['pred'], 'b-', linewidth=2)
        ax.plot(post_sorted['days_from_cutoff'], post_sorted['pred'], 'r-', linewidth=2)

        ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Lockdown (Jan 23)')
        ax.set_xlabel('Days from Lockdown')
        ax.set_ylabel(f'{outcome_col} (proportion)')
        ax.set_title(f'RDD: Lockdown Effect (±{bandwidth} days)')
        ax.legend()

        fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_rdd.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 图2 (RDD) 已保存")

        pd.DataFrame({
            'bandwidth': [bandwidth],
            'treatment_effect': [model_rdd.params['treatment']],
            'treatment_pvalue': [model_rdd.pvalues['treatment']],
            'r_squared': [model_rdd.rsquared],
        }).to_csv(os.path.join(OUTPUT_DIR, 'rdd_results.csv'), index=False)
    else:
        print("⚠️ 未找到恐慌率列")
else:
    print(f"⚠️ 未找到: {daily_path}")

# ==============================================================================
# 总结
# ==============================================================================

total_time = (datetime.now() - start_time).total_seconds()
print("\n" + "="*70)
print("✅ 完成！")
print("="*70)
print(f"总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
print(f"\n【样本】")
print(f"封城前已有恐慌的用户: {len(df_clean):,} 人")
print(f"  pre>={MIN_PRE_POSTS}帖, post>={MIN_POST_POSTS}帖, pre_panic>{MIN_PRE_PANIC}")
print(f"\n【恢复轨迹】")
for traj, count in traj_counts.items():
    sub = df_clean[df_clean['trajectory'] == traj]
    print(f"  {traj}: {count:,}人 ({count/len(df_clean):.1%}) | 变化: {sub['panic_change'].mean():.4f}")
print(f"\n【特征】")
print(f"封城前恐慌: {df_clean['pre_avg_panic'].mean():.4f} | 后: {df_clean['post_avg_panic'].mean():.4f}")
print(f"恐慌变化: {df_clean['panic_change'].mean():.4f} (中位数: {df_clean['panic_change'].median():.4f})")
print(f"恢复率(下降>10%): {df_clean['recovered'].mean():.1%}")
print(f"\n【输出】{OUTPUT_DIR}")