#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weibo COVID-19 & Panic Daily Trend (Dec 2019 - Jun 2020)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

# ==============================
# 1. Read data
# ==============================
files_config = [
    {'path': '/root/autodl-tmp/temp19_1.csv', 'panic_col': 'panic_pred'},
    {'path': '/root/autodl-tmp/temp1.csv', 'panic_col': 'panic'},
    {'path': '/root/autodl-tmp/temp2.csv', 'panic_col': 'panic'},
    {'path': '/root/autodl-tmp/temp3.csv', 'panic_col': 'panic'},
    {'path': '/root/autodl-tmp/temp4.csv', 'panic_col': 'panic'},
    {'path': '/root/autodl-tmp/temp5.csv', 'panic_col': 'panic'},
    {'path': '/root/autodl-tmp/temp6.csv', 'panic_col': 'panic'},
]

all_data = []
for cfg in files_config:
    path = cfg['path']
    panic_col = cfg['panic_col']
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, usecols=['created_at', 'covid_related', panic_col])
    df = df.rename(columns={panic_col: 'panic'})
    print(f"✅ {path} | rows: {len(df):,}")
    all_data.append(df)

df = pd.concat(all_data, ignore_index=True)
print(f"\n📊 Total: {len(df):,}")

# ==============================
# 2. Parse dates & daily stats
# ==============================
df['date'] = pd.to_datetime(df['created_at'], format='mixed', dayfirst=False).dt.date
df['date'] = pd.to_datetime(df['date'])

# Filter to exact range
df = df[(df['date'] >= '2019-12-01') & (df['date'] <= '2020-06-30')]

all_dates = pd.date_range(start='2019-12-01', end='2020-06-30', freq='D')

daily_covid = df[df['covid_related'] == 1].groupby('date').size().reindex(all_dates, fill_value=0)
daily_panic = df[df['panic'] == 1].groupby('date').size().reindex(all_dates, fill_value=0)

print(f"📅 Range: {all_dates.min().date()} ~ {all_dates.max().date()}")
print(f"📅 Days: {len(all_dates)}")

# ==============================
# 3. Plot
# ==============================
fig, ax1 = plt.subplots(figsize=(18, 7))

# Left: COVID posts (log scale)
color_covid = '#3A506B'
ax1.set_ylabel('COVID-related Posts (log scale)', color=color_covid, fontsize=12, fontweight='bold')
ax1.set_yscale('log')
ax1.set_ylim(1, daily_covid.max() * 1.5)
line1 = ax1.plot(all_dates, daily_covid, color=color_covid, linewidth=1.8,
                  marker='o', markersize=2.5, label='COVID-related Posts', alpha=0.85)
ax1.tick_params(axis='y', labelcolor=color_covid)
ax1.grid(True, alpha=0.3, which='both')

# Right: Panic posts (linear)
ax2 = ax1.twinx()
color_panic = '#C97B63'
ax2.set_ylabel('Panic Posts', color=color_panic, fontsize=12, fontweight='bold')
line2 = ax2.plot(all_dates, daily_panic, color=color_panic, linewidth=1.8,
                  marker='s', markersize=2.5, label='Panic Posts', alpha=0.85)
ax2.tick_params(axis='y', labelcolor=color_panic)

# ==============================
# X-axis: STRICT fix
# ==============================
# Must set xlim BEFORE any tick operations
ax1.set_xlim(pd.Timestamp('2019-12-01'), pd.Timestamp('2020-06-30'))

# Manual month ticks - no auto locator
month_ticks = pd.date_range('2019-12-01', '2020-06-01', freq='MS')
ax1.set_xticks(month_ticks)
ax1.set_xticklabels(
    ['2019-12', '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06'],
    rotation=45, ha='right', fontsize=11
)

# Remove any margins that could bleed into Nov
ax1.margins(x=0)

# ==============================
# Events (all English)
# ==============================
events = [
    ('2019-12-31', '1st COVID alert'),
    ('2020-01-20', 'Human-to-human'),
    ('2020-01-23', 'Wuhan lockdown'),
    ('2020-03-24', 'Hubei reopening'),
]
for d, label in events:
    ts = pd.Timestamp(d)
    if all_dates.min() <= ts <= all_dates.max():
        ax1.axvline(x=ts, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        # Place text at top of COVID axis
        y_pos = daily_covid.max() * 1.2
        ax1.text(ts, y_pos, label, rotation=90, ha='right', va='top',
                fontsize=8, color='gray', alpha=0.8)

# Legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=11, framealpha=0.9)

plt.title('Weibo Daily COVID-19 & Panic Posts Trend\n(Dec 2019 - Jun 2020)',
          fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/root/autodl-tmp/covid_panic_daily_fixed.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: /root/autodl-tmp/covid_panic_daily_fixed.png")
plt.show()

# Export
daily_df = pd.DataFrame({
    'date': all_dates,
    'covid_posts': daily_covid.values,
    'panic_posts': daily_panic.values
})
daily_df.to_csv('/root/autodl-tmp/daily_stats_detail.csv', index=False, encoding='utf-8-sig')
print("💾 CSV saved")