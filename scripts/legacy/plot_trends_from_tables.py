import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==================== 1. 读取数据并强制过滤范围 ====================
file_path = '/root/autodl-tmp/daily_stats_detail_2019_2020.csv'
daily_df = pd.read_csv(file_path)
daily_df['date'] = pd.to_datetime(daily_df['date'])

# 只保留 2019-12-01 到 2020-12-31 的数据，其他全部删掉
daily_df = daily_df[(daily_df['date'] >= '2019-12-01') & (daily_df['date'] <= '2020-12-31')].reset_index(drop=True)

# ==================== 2. 初始化画布 ====================
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax1 = plt.subplots(figsize=(16, 8))

# ==================== 3. 画主图 ====================
# 恐慌率曲线
color1 = '#3A506B'
ax1.plot(daily_df['date'], daily_df['panic_rate'],
         color=color1, linewidth=1.5, alpha=0.9, label='Panic Rate (%)')
ax1.fill_between(daily_df['date'], daily_df['panic_rate'],
                 alpha=0.15, color=color1)

ax1.set_ylabel('Panic Rate (%)', color=color1, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, daily_df['panic_rate'].max() * 1.2)

# 发帖量柱状图
ax2 = ax1.twinx()
color2 = '#C97B63'
ax2.bar(daily_df['date'], daily_df['total_posts'],
        alpha=0.3, color=color2, width=1.0, label='Daily Post Count')
ax2.set_ylabel('Daily Post Count', color=color2, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)

# ==================== 4. 关键：手动控制X轴，彻底干掉11月 ====================
# 强制设置X轴范围，Matplotlib无法再自动扩展
ax1.set_xlim([pd.to_datetime('2019-12-01'), pd.to_datetime('2021-01-05')])

# 手动生成你要的刻度（从2019-12到2020-12，每月1号）
ticks = pd.date_range(start='2019-12-01', end='2020-12-31', freq='MS')
ax1.set_xticks(ticks)
ax1.set_xticklabels([d.strftime('%b %Y') for d in ticks])

# 强制隐藏所有在数据范围外的刻度（双重保险）
for label in ax1.get_xticklabels():
    if label.get_text() == 'Nov 2019':
        label.set_visible(False)

# 调整标签角度
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ==================== 5. 事件标注和美化 ====================
EVENTS = {
    '2020-01-20': ('Dr. Zhong Nanshan\nconfirms H2H transmission', '#C97B63'),
    '2020-01-23': ('Wuhan Lockdown', '#3A506B'),
    '2020-03-24': ('Hubei reopening\nannouncement', '#6B8E23'),
    '2020-04-08': ('Wuhan officially\nreopens', '#8B4513'),
}
for date_str, (label, color) in EVENTS.items():
    event_date = pd.to_datetime(date_str)
    ax1.axvline(x=event_date, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.annotate(label, xy=(event_date, ax1.get_ylim()[1] * 0.95),
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=9, color=color, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

ax1.set_title('COVID-19 Related Weibo Posts: Panic Rate & Volume Trend\n(Dec 2019 - Dec 2020)',
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='-')

# 图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('/root/autodl-tmp/FINAL_FIXED.png', dpi=300, bbox_inches='tight')
plt.show()