import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import glob
import os

# ==================== Configuration ====================
DATA_DIR = '/root/autodl-tmp/'
OUTPUT_PATH = '/root/autodl-tmp/covid_panic_daily_trend.png'
DAILY_STATS_PATH = '/root/autodl-tmp/daily_stats_detail_2019_2020.csv'

# File list (from your terminal output)
FILE_LIST = [
    'temp19.csv', 'temp1.csv', 'temp2.csv', 'temp3.csv',
    'temp4.csv', 'temp5.csv', 'temp6.csv', 'temp7.csv',
    'temp8.csv', 'temp9.csv', 'temp10.csv', 'temp11.csv', 'temp12.csv'
]

# Key events (from Chinese labels in your original plot)
EVENTS = {
    '2020-01-20': ('Dr. Zhong Nanshan\nconfirms H2H transmission', '#C97B63'),
    '2020-01-23': ('Wuhan Lockdown', '#3A506B'),
    '2020-03-24': ('Hubei reopening\nannouncement', '#6B8E23'),
    '2020-04-08': ('Wuhan officially\nreopens', '#8B4513'),
}


# ==================== Data Loading ====================
def load_data():
    """Load and combine all temp CSV files"""
    all_data = []

    for fname in FILE_LIST:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"⚠️  File not found: {fpath}")
            continue

        df = pd.read_csv(fpath)

        # Detect panic column name
        if 'panic_pred' in df.columns:
            panic_col = 'panic_pred'
        elif 'panic' in df.columns:
            panic_col = 'panic'
        else:
            print(f"⚠️  No panic column in {fname}")
            continue

        # Ensure date column exists (assuming 'date' or 'created_at')
        date_col = None
        for col in ['date', 'created_at', 'time', 'publish_time']:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            print(f"⚠️  No date column in {fname}")
            continue

        # ✅ 自动兼容所有日期格式，容错解析
        df['date'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce').dt.date
        df['is_panic'] = df[panic_col].astype(int)

        # Keep only necessary columns
        all_data.append(df[['date', 'is_panic']])

        print(f"✅  Loaded: {fpath} | Rows: {len(df):,} | Panic col: {panic_col}")

    if not all_data:
        raise ValueError("❌ No valid data loaded! Check your file paths and column names.")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.dropna(subset=['date'])
    print(f"\n📊  Total records: {len(combined):,}")
    return combined


# ==================== Daily Aggregation ====================
def compute_daily_stats(df):
    """Compute daily panic rate and post counts"""
    daily = df.groupby('date').agg(
        total_posts=('is_panic', 'count'),
        panic_posts=('is_panic', 'sum')
    ).reset_index()

    daily['date'] = pd.to_datetime(daily['date'])
    daily['panic_rate'] = daily['panic_posts'] / daily['total_posts'] * 100

    # ==================== ✅ 固定从 2019-12-01 到 2020-12-31 ====================
    start_date = '2019-12-01'
    end_date = '2020-12-31'

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily = daily.set_index('date').reindex(date_range).reset_index()
    daily.columns = ['date', 'total_posts', 'panic_posts', 'panic_rate']
    daily[['total_posts', 'panic_posts']] = daily[['total_posts', 'panic_posts']].fillna(0).astype(int)
    daily['panic_rate'] = daily['panic_rate'].fillna(0)

    return daily


# ==================== Plotting ====================
def plot_trend(daily_df):
    """Generate the panic trend chart (The Economist style)"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Primary axis: Panic rate
    color1 = '#3A506B'
    ax1.plot(daily_df['date'], daily_df['panic_rate'],
             color=color1, linewidth=1.5, alpha=0.9, label='Panic Rate (%)')
    ax1.fill_between(daily_df['date'], daily_df['panic_rate'],
                     alpha=0.15, color=color1)

    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Panic Rate (%)', color=color1, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(daily_df['panic_rate']) * 1.2)

    # Secondary axis: Total posts
    ax2 = ax1.twinx()
    color2 = '#C97B63'
    ax2.bar(daily_df['date'], daily_df['total_posts'],
            alpha=0.3, color=color2, width=0.8, label='Daily Post Count')
    ax2.set_ylabel('Daily Post Count', color=color2, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Event annotations
    for date_str, (label, color) in EVENTS.items():
        event_date = pd.to_datetime(date_str)
        if event_date >= daily_df['date'].min() and event_date <= daily_df['date'].max():
            ax1.axvline(x=event_date, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            ax1.annotate(label, xy=(event_date, ax1.get_ylim()[1] * 0.95),
                         xytext=(10, 0), textcoords='offset points',
                         fontsize=9, color=color, fontweight='bold',
                         va='top', ha='left',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor=color, alpha=0.8))

    # Formatting
    ax1.set_title('COVID-19 Related Weibo Posts: Panic Rate & Volume Trend\n'
                  '(Dec 2019 - Dec 2020)',
                  fontsize=14, fontweight='bold', pad=20)

    # ✅ 关键修复：强制X轴范围 + 调整月份显示，避免自动往前对齐
    start_date_plot = pd.to_datetime('2019-12-01')
    end_date_plot = pd.to_datetime('2020-12-31')
    ax1.set_xlim(start_date_plot, end_date_plot)

    # 调整月份定位器，让标签从第一个月开始显示
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1, bymonthday=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax1.grid(True, alpha=0.3, linestyle='-')

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n💾  Chart saved: {OUTPUT_PATH}")

    daily_df.to_csv(DAILY_STATS_PATH, index=False)
    print(f"💾  Daily stats saved: {DAILY_STATS_PATH}")

    plt.show()
    return daily_df


# ==================== Main ====================
if __name__ == '__main__':
    print("Starting data processing...")
    df = load_data()

    print("\nProcessing dates and computing daily statistics...")
    daily_stats = compute_daily_stats(df)

    print(f"\n📅  Time span: {daily_stats['date'].min().date()} to {daily_stats['date'].max().date()}")
    print(f"📅  Total days: {len(daily_stats)}")

    print("\nGenerating chart...")
    result = plot_trend(daily_stats)

    print("\n📋  Daily stats preview (first 10 rows):")
    print(result.head(10).to_string(index=False))