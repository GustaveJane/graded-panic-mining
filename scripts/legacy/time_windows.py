#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Weibo COVID-19 Analysis: Panic Rate Spike & Drop Detection + Event Windows
================================================================================
功能：
1. 自动检测恐慌率剧烈波动点（骤升/骤降）
2. 标注对应事件
3. 生成事件窗口对比图（±7天/±14天）
4. 输出波动点详细分析表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/root/autodl-tmp'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES_CONFIG = [
    {'path': '/root/autodl-tmp/temp19_1.csv', 'panic_col': 'panic_pred'},
    {'path': '/root/autodl-tmp/temp1.csv', 'panic_col': 'panic'},
    {'path': '/root/autodl-tmp/temp2.csv', 'panic_col': 'panic'},
    {'path': '/root/autodl-tmp/temp3.csv', 'panic_col': 'panic'},
    {'path': '/root/autodl-tmp/temp4.csv', 'panic_col': 'panic'},
    {'path': '/root/autodl-tmp/temp5.csv', 'panic_col': 'panic'},
    {'path': '/root/autodl-tmp/temp6.csv', 'panic_col': 'panic'},
]

COLOR_PRIMARY = '#3A506B'
COLOR_ACCENT = '#C97B63'
COLOR_RED = '#C0392B'
COLOR_GREEN = '#27AE60'
COLOR_GRAY = '#7F8C8D'

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("PANIC RATE SPIKE & DROP ANALYSIS")
print("=" * 70)

# ==============================
# 1. LOAD DATA
# ==============================
print("\nLoading data...")

all_data = []
for cfg in FILES_CONFIG:
    path = cfg['path']
    panic_col = cfg['panic_col']
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, usecols=['created_at', 'covid_related', panic_col])
    df = df.rename(columns={panic_col: 'panic'})
    all_data.append(df)
    print(f"✅ {os.path.basename(path)} | {len(df):,}")

df = pd.concat(all_data, ignore_index=True)
df['date'] = pd.to_datetime(df['created_at'], format='mixed', dayfirst=False).dt.date
df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'] >= '2019-12-01') & (df['date'] <= '2020-06-30')]

all_dates = pd.date_range('2019-12-01', '2020-06-30', freq='D')
daily = df.groupby('date').agg(
    total_posts=('covid_related', 'count'),
    covid_posts=('covid_related', 'sum'),
    panic_posts=('panic', 'sum')
).reindex(all_dates, fill_value=0)

daily['panic_rate'] = np.where(daily['covid_posts'] > 0,
                               daily['panic_posts'] / daily['covid_posts'], 0)
daily = daily.reset_index().rename(columns={'index': 'date'})

print(f"\n📊 Total: {len(df):,} | Days: {len(daily)}")

# ==============================
# 2. DETECT SPIKES & DROPS
# ==============================
print("\n" + "=" * 70)
print("STEP 2: DETECTING PANIC RATE SPIKES & DROPS")
print("=" * 70)

# 计算多种变化指标
daily['panic_rate_diff'] = daily['panic_rate'].diff()
daily['panic_rate_diff_next'] = daily['panic_rate'].diff(-1)  # 次日变化（用于找骤降前的高点）
daily['panic_rate_pct_change'] = daily['panic_rate'].pct_change()

# 7日移动平均和标准差（用于识别异常波动）
daily['panic_rate_ma7'] = daily['panic_rate'].rolling(window=7, center=True, min_periods=1).mean()
daily['panic_rate_std7'] = daily['panic_rate'].rolling(window=7, center=True, min_periods=1).std()
daily['z_score'] = (daily['panic_rate'] - daily['panic_rate_ma7']) / daily['panic_rate_std7'].replace(0, np.nan)

# 定义波动阈值
ABS_THRESHOLD = 0.01  # 绝对变化阈值（1个百分点）
Z_THRESHOLD = 2.0  # Z-score阈值

# 找出骤升点（当日比前一日大幅上升）
spikes = daily[
    (daily['panic_rate_diff'] > ABS_THRESHOLD) |
    (daily['z_score'] > Z_THRESHOLD)
    ].copy()
spikes = spikes[spikes['panic_rate'] > 0.005]  # 过滤掉从0开始的噪声
spikes['type'] = 'SPIKE'

# 找出骤降点（当日比前一日大幅下降）
drops = daily[
    (daily['panic_rate_diff'] < -ABS_THRESHOLD) |
    (daily['z_score'] < -Z_THRESHOLD)
    ].copy()
drops = drops[drops['panic_rate'].shift(1) > 0.005]  # 前一日有一定基数
drops['type'] = 'DROP'

# 合并并排序
volatility_events = pd.concat([spikes, drops]).sort_values('date')
volatility_events = volatility_events.drop_duplicates(subset=['date'])

print(f"\n📈 Detected {len(spikes)} SPIKES")
print(f"📉 Detected {len(drops)} DROPS")
print(f"📊 Total volatility events: {len(volatility_events)}")

# 打印Top波动
print("\n" + "-" * 70)
print("TOP 10 STRONGEST VOLATILITY EVENTS")
print("-" * 70)
top_vol = \
volatility_events.nlargest(10, 'panic_rate_diff' if 'panic_rate_diff' in volatility_events.columns else 'panic_rate')[
    ['date', 'panic_rate', 'panic_rate_diff', 'type']]
print(top_vol.to_string())

# ==============================
# 3. MANUAL EVENT ANNOTATIONS
# ==============================
print("\n" + "=" * 70)
print("STEP 3: KEY EVENT ANNOTATIONS")
print("=" * 70)

# 手动标注已知关键事件（基于历史记录和波动检测结果）
KEY_EVENTS = [
    # 格式: (日期, 事件名, 类型[SPIKE/DROP], 描述, 预期影响)
    ('2019-12-30', 'First COVID Info Leak', 'SPIKE',
     '武汉不明原因肺炎信息首次在社交媒体流传', '信息真空期的极度恐慌'),
    ('2019-12-31', 'Official First Alert', 'DROP',
     '武汉卫健委首次正式通报27例不明原因肺炎', '官方信息释放，不确定性降低'),
    ('2020-01-20', 'Human-to-Human Confirmed', 'SPIKE',
     '钟南山院士确认新冠病毒人传人', '权威确认引发恐慌升级'),
    ('2020-01-23', 'Wuhan Lockdown', 'DROP',
     '武汉正式封城，公共交通暂停', '强制性政策提供确定性，恐慌骤降'),
    ('2020-01-25', 'Spring Festival + Lockdown Extended', 'DROP',
     '大年初一，全国多地启动一级响应', '节日+政策双重镇静'),
    ('2020-02-02', 'Wuhan Hospital Crisis', 'SPIKE',
     '医疗资源挤兑，方舱医院开始建设', '短期恐慌反弹'),
    ('2020-03-24', 'Hubei Reopening Notice', 'NEUTRAL',
     '湖北省发布解除离鄂通道管控通告', '预期已消化，波动不大'),
    ('2020-04-04', 'National Mourning Day', 'DROP',
     '全国性哀悼活动，降半旗', '仪式性情绪宣泄'),
    ('2020-04-08', 'Wuhan Reopening', 'NEUTRAL',
     '武汉正式解除离汉离鄂通道管控', '利好出尽，情绪平稳'),
    ('2020-05-14', 'Jilin Shulan Outbreak', 'SPIKE',
     '吉林舒兰市出现聚集性疫情，封城管理', '二次反弹恐慌'),
    ('2020-05-17', 'Two Sessions Confirmed', 'DROP',
     '全国政协十三届三次会议召开日期确定', '政治信号稳定预期'),
    ('2020-06-13', 'Beijing Xinfadi Outbreak', 'SPIKE',
     '北京新发地市场疫情复发', '局部反弹引发短期恐慌'),
]

events_df = pd.DataFrame(KEY_EVENTS, columns=['date', 'event_name', 'event_type', 'description', 'mechanism'])
events_df['date'] = pd.to_datetime(events_df['date'])

print("\nAnnotated Key Events:")
print(events_df[['date', 'event_name', 'event_type']].to_string(index=False))

# ==============================
# 4. PLOT 1: PANIC RATE WITH ANNOTATIONS
# ==============================
print("\n" + "=" * 70)
print("STEP 4: GENERATING ANNOTATED TREND PLOT")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})

# Top: COVID Posts + Panic Posts (log)
ax1_twin = ax1.twinx()
ax1.set_ylabel('COVID Posts (log)', color=COLOR_PRIMARY, fontsize=12, fontweight='bold')
ax1.set_yscale('log')
ax1.set_ylim(1, daily['covid_posts'].max() * 1.5)
ax1.plot(daily['date'], daily['covid_posts'], color=COLOR_PRIMARY, linewidth=1.5, alpha=0.8, label='COVID Posts')
ax1.tick_params(axis='y', labelcolor=COLOR_PRIMARY)
ax1.grid(True, alpha=0.2, which='both')

ax1_twin.set_ylabel('Panic Posts', color=COLOR_ACCENT, fontsize=12, fontweight='bold')
ax1_twin.plot(daily['date'], daily['panic_posts'], color=COLOR_ACCENT, linewidth=1.5, alpha=0.8, label='Panic Posts')
ax1_twin.tick_params(axis='y', labelcolor=COLOR_ACCENT)
ax1.set_title('Daily COVID Posts & Panic Posts', fontsize=13, fontweight='bold', pad=10)

# Bottom: Panic Rate with annotations
ax2.set_ylabel('Panic Rate (log scale)', color=COLOR_RED, fontsize=12, fontweight='bold')

# 使用对数坐标显示恐慌率
epsilon = 1e-6
ax2.set_yscale('log')
ax2.set_ylim(epsilon, daily['panic_rate'].max() * 2)

ax2.fill_between(daily['date'], daily['panic_rate'], alpha=0.25, color=COLOR_RED)
ax2.plot(daily['date'], daily['panic_rate'], color=COLOR_RED, linewidth=1.8, label='Panic Rate', zorder=3)

# 7-day MA
ax2.plot(daily['date'], daily['panic_rate_ma7'], color='black', linewidth=1.2,
         linestyle='--', alpha=0.5, label='7-day MA', zorder=4)

# 标注关键事件
for _, evt in events_df.iterrows():
    evt_date = evt['date']
    if daily['date'].min() <= evt_date <= daily['date'].max():
        # 根据事件类型选择颜色
        if evt['event_type'] == 'SPIKE':
            color = COLOR_RED
            marker = '▲'
        elif evt['event_type'] == 'DROP':
            color = COLOR_GREEN
            marker = '▼'
        else:
            color = COLOR_GRAY
            marker = '●'

        # 画垂直线
        ax2.axvline(x=evt_date, color=color, linestyle='--', alpha=0.5, linewidth=1.5, zorder=2)

        # 找对应的恐慌率值用于标注位置
        idx = (daily['date'] - evt_date).abs().idxmin()
        y_pos = daily.loc[idx, 'panic_rate'] if daily.loc[idx, 'panic_rate'] > epsilon else epsilon * 10

        # 标注事件名（旋转90度）
        ax2.text(evt_date, daily['panic_rate'].max() * 1.3,
                 f"{marker} {evt['event_name']}",
                 rotation=90, ha='right', va='top', fontsize=8,
                 color=color, fontweight='bold', zorder=5)

# X-axis
ax2.set_xlim(pd.Timestamp('2019-12-01'), pd.Timestamp('2020-06-30'))
month_ticks = pd.date_range('2019-12-01', '2020-06-01', freq='MS')
ax2.set_xticks(month_ticks)
ax2.set_xticklabels(['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], rotation=0, fontsize=11)
ax2.set_xlabel('Date', fontsize=12)

ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.2, which='both')
ax2.set_title('Panic Rate with Key Events (▲=Spike, ▼=Drop, ●=Neutral)',
              fontsize=13, fontweight='bold', pad=10)

plt.suptitle('Panic Rate Trend with Volatility Event Annotations\n(Dec 2019 - Jun 2020)',
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/panic_rate_annotated.png', dpi=300, bbox_inches='tight')
print(f"\n💾 Saved: {OUTPUT_DIR}/panic_rate_annotated.png")
plt.show()

# ==============================
# 5. PLOT 2: EVENT WINDOWS FOR KEY SPIKES/DROPS
# ==============================
print("\n" + "=" * 70)
print("STEP 5: EVENT WINDOW COMPARISON FOR VOLATILITY EVENTS")
print("=" * 70)


def plot_detailed_event_window(event_date, event_name, event_type, description, window=7, ax=None):
    """绘制单个事件的详细窗口图，包含更多信息"""
    event_ts = pd.Timestamp(event_date)
    start = event_ts - pd.Timedelta(days=window)
    end = event_ts + pd.Timedelta(days=window)

    window_data = daily[(daily['date'] >= start) & (daily['date'] <= end)].copy()
    window_data['days_to_event'] = (window_data['date'] - event_ts).dt.days

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # 主Y轴：COVID Posts（柱状）
    colors = [COLOR_PRIMARY if d < 0 else (COLOR_ACCENT if d > 0 else COLOR_RED)
              for d in window_data['days_to_event']]
    bars = ax.bar(window_data['days_to_event'], window_data['covid_posts'],
                  color=colors, alpha=0.6, width=0.8, label='COVID Posts')

    # 标记事件日
    event_bar_idx = window_data[window_data['days_to_event'] == 0].index
    if len(event_bar_idx) > 0:
        bars[list(window_data.index).index(event_bar_idx[0])].set_color(COLOR_RED)
        bars[list(window_data.index).index(event_bar_idx[0])].set_alpha(1.0)

    ax.set_ylabel('COVID Posts', color=COLOR_PRIMARY, fontsize=10, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=COLOR_PRIMARY)

    # 次Y轴：Panic Rate（折线）
    ax_twin = ax.twinx()

    # 根据事件类型选择颜色
    line_color = COLOR_RED if event_type == 'SPIKE' else COLOR_GREEN

    ax_twin.plot(window_data['days_to_event'], window_data['panic_rate'],
                 color=line_color, linewidth=3, marker='o', markersize=5,
                 label=f'Panic Rate ({event_type})', zorder=5)

    # 标注事件日的恐慌率值
    event_day = window_data[window_data['days_to_event'] == 0]
    if len(event_day) > 0:
        pr_val = event_day['panic_rate'].values[0]
        ax_twin.scatter([0], [pr_val], color='black', s=100, zorder=6, marker='*')
        ax_twin.annotate(f'{pr_val:.4f}', xy=(0, pr_val),
                         xytext=(2, pr_val * 1.2 if pr_val > 0 else 0.01),
                         fontsize=10, fontweight='bold', color='black')

    ax_twin.set_ylabel('Panic Rate', color=line_color, fontsize=10, fontweight='bold')
    ax_twin.tick_params(axis='y', labelcolor=line_color)

    # 自动设置Y轴范围
    pr_max = window_data['panic_rate'].max()
    ax_twin.set_ylim(0, pr_max * 1.4)

    # 事件线
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2.5, label='Event Day')

    # 标题包含描述
    ax.set_xlabel('Days to Event', fontsize=10)
    ax.set_title(f'{event_name}\n({event_date}) | {description}',
                 fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, alpha=0.2)

    # 图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    return window_data


# 选择最关键的6个波动事件做详细窗口
KEY_VOLATILITY_EVENTS = [
    ('2019-12-30', 'First COVID Info Leak', 'SPIKE', '信息真空期恐慌'),
    ('2019-12-31', 'Official First Alert', 'DROP', '官方通报镇静'),
    ('2020-01-20', 'Human-to-Human Confirmed', 'SPIKE', '权威确认引发恐慌'),
    ('2020-01-23', 'Wuhan Lockdown', 'DROP', '封城政策镇静效应'),
    ('2020-05-14', 'Jilin Shulan Outbreak', 'SPIKE', '二次反弹恐慌'),
    ('2020-05-17', 'Two Sessions Confirmed', 'DROP', '政治信号稳定预期'),
]

fig, axes = plt.subplots(3, 2, figsize=(18, 16))
axes = axes.flatten()

for ax, (date, name, etype, desc) in zip(axes, KEY_VOLATILITY_EVENTS):
    plot_detailed_event_window(date, name, etype, desc, window=7, ax=ax)

plt.suptitle('Key Volatility Events: Detailed Window Analysis (±7 days)\n▲=Spike (Panic Rises) | ▼=Drop (Panic Falls)',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/volatility_event_windows.png', dpi=300, bbox_inches='tight')
print(f"\n💾 Saved: {OUTPUT_DIR}/volatility_event_windows.png")
plt.show()

# ==============================
# 6. SUMMARY TABLE
# ==============================
print("\n" + "=" * 70)
print("STEP 6: VOLATILITY EVENT SUMMARY TABLE")
print("=" * 70)

summary_data = []
for date_str, name, etype, desc in KEY_VOLATILITY_EVENTS:
    event_ts = pd.Timestamp(date_str)

    # ±3天窗口
    pre3 = daily[(daily['date'] >= event_ts - pd.Timedelta(days=3)) & (daily['date'] < event_ts)]
    post3 = daily[(daily['date'] >= event_ts) & (daily['date'] <= event_ts + pd.Timedelta(days=3))]

    # ±7天窗口
    pre7 = daily[(daily['date'] >= event_ts - pd.Timedelta(days=7)) & (daily['date'] < event_ts)]
    post7 = daily[(daily['date'] >= event_ts) & (daily['date'] <= event_ts + pd.Timedelta(days=7))]

    event_day = daily[daily['date'] == event_ts]
    pr_event = event_day['panic_rate'].values[0] if len(event_day) > 0 else np.nan

    summary_data.append({
        'Event': name,
        'Date': date_str,
        'Type': etype,
        'PanicRate_EventDay': f"{pr_event:.4f}" if not np.isnan(pr_event) else 'N/A',
        'Pre3_PR_mean': f"{pre3['panic_rate'].mean():.4f}" if len(pre3) > 0 else 'N/A',
        'Post3_PR_mean': f"{post3['panic_rate'].mean():.4f}" if len(post3) > 0 else 'N/A',
        'Change_3d': f"{post3['panic_rate'].mean() - pre3['panic_rate'].mean():.4f}" if len(pre3) > 0 and len(
            post3) > 0 else 'N/A',
        'Pre7_PR_mean': f"{pre7['panic_rate'].mean():.4f}" if len(pre7) > 0 else 'N/A',
        'Post7_PR_mean': f"{post7['panic_rate'].mean():.4f}" if len(post7) > 0 else 'N/A',
        'Change_7d': f"{post7['panic_rate'].mean() - pre7['panic_rate'].mean():.4f}" if len(pre7) > 0 and len(
            post7) > 0 else 'N/A',
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# 保存CSV
summary_df.to_csv(f'{OUTPUT_DIR}/volatility_events_summary.csv', index=False, encoding='utf-8-sig')
print(f"\n💾 Summary CSV saved: {OUTPUT_DIR}/volatility_events_summary.csv")

# ==============================
# 7. PLOT 3: SPIKE vs DROP COMPARISON
# ==============================
print("\n" + "=" * 70)
print("STEP 7: SPIKE vs DROP PATTERN COMPARISON")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: All spikes overlay
ax1.set_title('Panic Rate Spikes: Overlay Pattern', fontsize=13, fontweight='bold')
spike_events = [e for e in KEY_VOLATILITY_EVENTS if e[2] == 'SPIKE']

for date_str, name, etype, desc in spike_events:
    event_ts = pd.Timestamp(date_str)
    window_data = daily[(daily['date'] >= event_ts - pd.Timedelta(days=7)) &
                        (daily['date'] <= event_ts + pd.Timedelta(days=7))].copy()
    window_data['days_to_event'] = (window_data['date'] - event_ts).dt.days

    # 标准化到事件前1天为基准
    baseline = window_data[window_data['days_to_event'] == -1]['panic_rate'].values
    if len(baseline) > 0 and baseline[0] > 0:
        window_data['pr_normalized'] = window_data['panic_rate'] / baseline[0]
    else:
        window_data['pr_normalized'] = window_data['panic_rate']

    ax1.plot(window_data['days_to_event'], window_data['pr_normalized'],
             marker='o', markersize=3, linewidth=2, label=name[:20], alpha=0.8)

ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Days to Event', fontsize=11)
ax1.set_ylabel('Normalized Panic Rate\n(Event Day-1 = 1)', fontsize=11)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.2)

# Right: All drops overlay
ax2.set_title('Panic Rate Drops: Overlay Pattern', fontsize=13, fontweight='bold')
drop_events = [e for e in KEY_VOLATILITY_EVENTS if e[2] == 'DROP']

for date_str, name, etype, desc in drop_events:
    event_ts = pd.Timestamp(date_str)
    window_data = daily[(daily['date'] >= event_ts - pd.Timedelta(days=7)) &
                        (daily['date'] <= event_ts + pd.Timedelta(days=7))].copy()
    window_data['days_to_event'] = (window_data['date'] - event_ts).dt.days

    baseline = window_data[window_data['days_to_event'] == -1]['panic_rate'].values
    if len(baseline) > 0 and baseline[0] > 0:
        window_data['pr_normalized'] = window_data['panic_rate'] / baseline[0]
    else:
        window_data['pr_normalized'] = window_data['panic_rate']

    ax2.plot(window_data['days_to_event'], window_data['pr_normalized'],
             marker='s', markersize=3, linewidth=2, label=name[:20], alpha=0.8)

ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Days to Event', fontsize=11)
ax2.set_ylabel('Normalized Panic Rate\n(Event Day-1 = 1)', fontsize=11)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.2)

plt.suptitle('Spike vs Drop: Normalized Panic Rate Patterns', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/spike_drop_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n💾 Saved: {OUTPUT_DIR}/spike_drop_comparison.png")
plt.show()

print("\n" + "=" * 70)
print("ALL ANALYSES COMPLETE")
print("=" * 70)
print(f"\nOutput files:")
print(f"  1. {OUTPUT_DIR}/panic_rate_annotated.png")
print(f"  2. {OUTPUT_DIR}/volatility_event_windows.png")
print(f"  3. {OUTPUT_DIR}/spike_drop_comparison.png")
print(f"  4. {OUTPUT_DIR}/volatility_events_summary.csv")