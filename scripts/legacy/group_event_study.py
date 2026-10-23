#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Event Study — Daily Aggregation v3d (Strategy A: Located-only for geo)
============================================================================
Geo analysis: ONLY posts with valid geo_info OR explicit text location mention.
All other analyses (penetration, tier curves, topics): use FULL sample.
VS Code Plots右侧可视化适配版
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import gc
import re
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== 全局绘图配置（解决中文乱码） ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== CONFIG ====================
DATA_DIR = '/root/autodl-tmp/'
OUTPUT_DIR = '/root/autodl-tmp/group_event_study_v3d/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_LIST = [
    'temp19.csv', 'temp1.csv', 'temp2.csv', 'temp3.csv',
    'temp4.csv', 'temp5.csv', 'temp6.csv', 'temp7.csv',
    'temp8.csv', 'temp9.csv', 'temp10.csv', 'temp11.csv', 'temp12.csv'
]

CHUNK_SIZE = 100000

GMM_THRESHOLDS = [0.4339, 0.6174, 0.8155, 0.9584]
TIER_LABELS = {
    0: 'Tier0-Low',
    1: 'Tier1-Low',
    2: 'Tier2-Medium',
    3: 'Tier3-High',
    4: 'Tier4-High'
}

EVENT_START = pd.to_datetime('2020-01-18')
EVENT_END   = pd.to_datetime('2020-01-25')
CUTOFF_DT   = pd.to_datetime('2020-01-23')

# Geo bounds for coordinate-based classification
GEO_BOUNDS = {
    'Wuhan':    {'lon': (113.5, 114.5), 'lat': (30.0, 31.0)},
    'Beijing':  {'lon': (115.5, 117.5), 'lat': (39.5, 41.5)},
    'Shanghai': {'lon': (120.5, 122.5), 'lat': (30.5, 32.5)},
    'Guangzhou':{'lon': (112.5, 114.5), 'lat': (22.5, 24.5)},
    'Shenzhen': {'lon': (113.5, 114.8), 'lat': (22.2, 22.9)},
}

# Hubei province bounds (Wuhan excluded to avoid double-counting)
HUBEI_BOUNDS = {'lon': (108.0, 116.5), 'lat': (29.0, 33.5)}

# Text-based city patterns (fallback for geo_info missing)
TEXT_CITY_PATTERNS = {
    'Wuhan':    ['武汉', '汉口', '武昌', '汉阳', '江夏', '光谷', '东湖'],
    'Beijing':  ['北京', '帝都', '京城'],
    'Shanghai': ['上海', '魔都'],
    'Guangzhou':['广州', '羊城', '花城'],
    'Shenzhen': ['深圳', '鹏城'],
    'Hubei':    ['湖北', '襄阳', '宜昌', '荆州', '十堰', '黄石', '鄂州', '荆门', '黄冈', '孝感', '咸宁', '随州', '恩施', '天门', '仙桃', '潜江'],
}

TOPIC_KWS = {
    'P8_travel': ['航班', '高铁', '火车', '地铁', '返乡', '回家', '滞留', '出境', '机票', '车站', '封路', '停运'],
    'P6_info_seek': ['数据', '真相', '到底', '究竟', '为什么', '怎么回事', '求科普', '什么情况', '最新', '通报', '发布'],
    'P7_symptom': ['发烧', '咳嗽', '嗓子', '头晕', '乏力', '检测', '核酸', '阳性', '症状', '感染', '确诊', '体温'],
    'P2_life_supply': ['口罩', '买菜', '物资', '粮食', '蔬菜', '超市', '快递', '外卖', '缺货', '囤货', '酒精'],
    'P5_emotion_vent': ['害怕', '恐惧', '担心', '焦虑', '崩溃', '想哭', '想死', '压抑', '绝望', '无助', '心慌', '发抖'],
    'P1_medical_resource': ['医院', '床位', '透析', '化疗', '手术', '急诊', 'ICU', '救护车', '医护', '医生', '护士'],
    'P3_work_school': ['上班', '开学', '复工', '交通', '地铁', '公交', '工资', '失业', '停课', '远程', '办公'],
    'P4_isolation': ['居家', '隔离', '封控', '小区', '出入', '通行证', '健康码', '封闭', '封锁', '管控'],
}

GEO_CANDIDATES = ['geo_info', 'get_info', 'location', 'city', 'geo', '定位', 'place', 'region', 'area', 'province', 'addr']

# ==================== FUNCTIONS ====================

def assign_gmm_tier(panic_prob):
    if pd.isna(panic_prob):
        return np.nan
    pp = float(panic_prob)
    for i, thresh in enumerate(GMM_THRESHOLDS):
        if pp < thresh:
            return i
    return 4


def detect_geo_column(columns):
    cols_lower = [c.lower() for c in columns]
    for cand in GEO_CANDIDATES:
        if cand.lower() in cols_lower:
            idx = cols_lower.index(cand.lower())
            return columns[idx]
    return None


def parse_coords(geo_str):
    """Parse 'lon,lat' string into (lon, lat) tuple."""
    if pd.isna(geo_str):
        return None, None
    s = str(geo_str).strip()
    # Remove common wrappers
    s = s.strip('"').strip("'").strip()
    # Match patterns like "114.3055,30.5928" or "114.3055, 30.5928"
    match = re.match(r'^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$', s)
    if match:
        try:
            lon = float(match.group(1))
            lat = float(match.group(2))
            return lon, lat
        except ValueError:
            return None, None
    return None, None


def classify_by_coords(lon, lat):
    """Classify region by coordinates."""
    if lon is None or lat is None:
        return None
    # Check specific cities first
    for city, bounds in GEO_BOUNDS.items():
        if bounds['lon'][0] <= lon <= bounds['lon'][1] and bounds['lat'][0] <= lat <= bounds['lat'][1]:
            if city == 'Wuhan':
                return 'Wuhan'
            elif city in ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen']:
                return 'Tier1'
    # Check Hubei province (excluding Wuhan already matched above)
    if HUBEI_BOUNDS['lon'][0] <= lon <= HUBEI_BOUNDS['lon'][1] and \
       HUBEI_BOUNDS['lat'][0] <= lat <= HUBEI_BOUNDS['lat'][1]:
        return 'Hubei'
    return 'Other'


def classify_by_text(text):
    """Classify region by text mentions."""
    if pd.isna(text):
        return None
    text = str(text)
    for region, patterns in TEXT_CITY_PATTERNS.items():
        for pat in patterns:
            if pat in text:
                if region == 'Wuhan':
                    return 'Wuhan'
                elif region == 'Hubei':
                    return 'Hubei'
                elif region in ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen']:
                    return 'Tier1'
    return None


def classify_region(row, geo_col_name):
    """Unified region classifier: coords first, then text, then Unknown."""
    # Strategy A: only use location if geo_info is present OR text explicitly mentions location
    geo_val = row.get(geo_col_name, None)
    lon, lat = parse_coords(geo_val)
    if lon is not None and lat is not None:
        region = classify_by_coords(lon, lat)
        if region is not None:
            return region
    # Fallback: text-based (only if geo_info is missing but text has explicit mention)
    text_region = classify_by_text(row.get('content', ''))
    if text_region is not None:
        return text_region
    return 'Unknown'


def classify_topic(text):
    if pd.isna(text):
        return 'other'
    text = str(text)
    scores = {}
    for topic, kws in TOPIC_KWS.items():
        scores[topic] = sum(1 for kw in kws if kw in text)
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return 'other'

# %% 1. 数据读取与聚合主函数
# ==================== MAIN ====================

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Group Event Study v3d started")
    print(f"Strategy A: Located-only for geo analysis; full sample for all else")
    print(f"Output directory: {OUTPUT_DIR}")

    geo_col_name = None

    # ------------------------------------------------------------------
    # Phase 1: Read raw temp files
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 1: Reading raw temp files (full population)")
    print(f"{'='*60}")

    all_records = []
    total_posts = total_covid = total_valid = 0

    for fname in FILE_LIST:
        fp = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fp):
            print(f"  [SKIP] {fname} not found")
            continue

        print(f"\n  Processing {fname} ...")
        file_posts = file_covid = file_valid = 0

        for chunk in pd.read_csv(
            fp, chunksize=CHUNK_SIZE, on_bad_lines='skip', low_memory=False
        ):
            req_cols = ['content', 'created_at', 'covid_related', 'panic_prob']
            if not all(c in chunk.columns for c in req_cols):
                missing = [c for c in req_cols if c not in chunk.columns]
                print(f"    [SKIP] chunk missing: {missing}")
                continue

            total_posts += len(chunk)
            file_posts += len(chunk)

            chunk['date'] = pd.to_datetime(chunk['created_at'], errors='coerce')
            n_bad = chunk['date'].isna().sum()
            if n_bad > 0:
                print(f"    [WARN] {n_bad:,} unparseable dates skipped")
            chunk = chunk[chunk['date'].notna()]

            chunk['covid_related'] = pd.to_numeric(chunk['covid_related'], errors='coerce').fillna(0)
            covid_chunk = chunk[chunk['covid_related'] == 1].copy()
            total_covid += len(covid_chunk)
            file_covid += len(covid_chunk)

            if len(covid_chunk) == 0:
                continue

            covid_chunk['panic_prob'] = pd.to_numeric(covid_chunk['panic_prob'], errors='coerce')
            valid_chunk = covid_chunk[covid_chunk['panic_prob'].notna()].copy()
            total_valid += len(valid_chunk)
            file_valid += len(valid_chunk)

            if len(valid_chunk) == 0:
                continue

            # Auto-detect geo column on first valid chunk
            if geo_col_name is None:
                detected = detect_geo_column(list(valid_chunk.columns))
                if detected:
                    geo_col_name = detected
                    print(f"    [INFO] Geo column auto-detected: '{geo_col_name}'")
                else:
                    print(f"    [INFO] No geo column detected. Candidates: {GEO_CANDIDATES}")
                    geo_col_name = '__NONE__'

            # GMM tier
            valid_chunk['gmm_tier'] = valid_chunk['panic_prob'].apply(assign_gmm_tier)
            valid_chunk = valid_chunk[valid_chunk['gmm_tier'].notna()]

            valid_chunk['date_only'] = pd.to_datetime(valid_chunk['date'].dt.date)

            # Region classification (Strategy A: coords + text fallback)
            if geo_col_name and geo_col_name != '__NONE__':
                valid_chunk['region'] = valid_chunk.apply(
                    lambda row: classify_region(row, geo_col_name), axis=1
                )
            else:
                valid_chunk['region'] = valid_chunk['content'].apply(classify_by_text).fillna('Unknown')

            # Topic classification
            valid_chunk['topic'] = valid_chunk['content'].apply(classify_topic)

            # Mark if post has explicit location (for Strategy A)
            valid_chunk['has_location'] = valid_chunk['region'] != 'Unknown'

            keep_cols = ['date_only', 'gmm_tier', 'panic_prob', 'region', 'topic', 'has_location']
            all_records.append(valid_chunk[keep_cols].copy())

            del valid_chunk, covid_chunk, chunk
            gc.collect()

        print(f"    Posts: {file_posts:,} | COVID: {file_covid:,} | Valid: {file_valid:,}")

    # ------------------------------------------------------------------
    # Phase 2: Concatenate & build FULL sample + LOCATED subset
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 2: Concatenating & building FULL + LOCATED samples")
    print(f"{'='*60}")

    if len(all_records) == 0:
        print("[ERROR] No valid data found!")
        return

    df = pd.concat(all_records, ignore_index=True)
    print(f"Total valid COVID posts: {len(df):,}")
    print(f"  Date range: {df['date_only'].min()} ~ {df['date_only'].max()}")
    print(f"  Unique dates: {df['date_only'].nunique()}")
    print(f"  GMM tier distribution:")
    print(df['gmm_tier'].value_counts().sort_index().to_string())
    print(f"  Region distribution (FULL sample):")
    print(df['region'].value_counts().to_string())
    print(f"  Located posts: {df['has_location'].sum():,} ({df['has_location'].mean()*100:.2f}%)")
    print(f"  Topic distribution (top 10):")
    print(df['topic'].value_counts().head(10).to_string())

    # Strategy A: build located-only subset for geo analysis
    df_located = df[df['has_location'] == True].copy()
    print(f"\n  LOCATED subset for geo analysis: {len(df_located):,} posts")
    print(f"  Located region distribution:")
    print(df_located['region'].value_counts().to_string())

    del all_records
    gc.collect()

    # --- 2.1 Global daily by tier (FULL) ---
    global_agg = df.groupby(['date_only', 'gmm_tier']).agg(
        panic_mean=('panic_prob', 'mean'),
        post_count=('panic_prob', 'size')
    ).reset_index()
    global_agg['tier_label'] = global_agg['gmm_tier'].map(TIER_LABELS)
    global_agg.to_csv(f'{OUTPUT_DIR}daily_panic_by_tier.csv', index=False, encoding='utf-8-sig')
    print(f"  [SAVE] daily_panic_by_tier.csv (FULL)")

    # --- 2.2 Overall daily (FULL) ---
    overall_daily = df.groupby('date_only').agg(
        overall_panic_mean=('panic_prob', 'mean'),
        total_posts=('panic_prob', 'size')
    ).reset_index()
    overall_daily.to_csv(f'{OUTPUT_DIR}daily_panic_overall.csv', index=False, encoding='utf-8-sig')
    print(f"  [SAVE] daily_panic_overall.csv (FULL)")

    # --- 2.3 Regional daily by tier (LOCATED ONLY) ---
    regional_agg = df_located.groupby(['date_only', 'region', 'gmm_tier']).agg(
        panic_mean=('panic_prob', 'mean'),
        post_count=('panic_prob', 'size')
    ).reset_index()
    regional_agg['tier_label'] = regional_agg['gmm_tier'].map(TIER_LABELS)
    regional_agg.to_csv(f'{OUTPUT_DIR}daily_panic_by_region_tier.csv', index=False, encoding='utf-8-sig')
    print(f"  [SAVE] daily_panic_by_region_tier.csv (LOCATED ONLY)")

    # --- 2.4 Topic daily (FULL) ---
    topic_daily = df.groupby(['date_only', 'topic']).size().reset_index(name='topic_count')
    topic_total = df.groupby('date_only').size().reset_index(name='total_count')
    topic_daily = topic_daily.merge(topic_total, on='date_only')
    topic_daily['topic_pct'] = topic_daily['topic_count'] / topic_daily['total_count'] * 100
    topic_daily.to_csv(f'{OUTPUT_DIR}daily_topic_proportion.csv', index=False, encoding='utf-8-sig')
    print(f"  [SAVE] daily_topic_proportion.csv (FULL)")

    # --- 2.5 Event window (FULL) ---
    event_mask = (df['date_only'] >= EVENT_START) & (df['date_only'] <= EVENT_END)
    event_df = df[event_mask].copy()
    print(f"\n  Event window (FULL): {len(event_df):,} posts")

    event_global = None
    event_topic = None
    if len(event_df) > 0:
        event_global = event_df.groupby(['date_only', 'gmm_tier']).agg(
            panic_mean=('panic_prob', 'mean'),
            post_count=('panic_prob', 'size')
        ).reset_index()
        event_global['tier_label'] = event_global['gmm_tier'].map(TIER_LABELS)
        event_global.to_csv(f'{OUTPUT_DIR}event_window_panic_by_tier.csv', index=False, encoding='utf-8-sig')
        print(f"  [SAVE] event_window_panic_by_tier.csv (FULL)")

        event_topic = event_df.groupby(['date_only', 'topic']).size().reset_index(name='topic_count')
        event_topic_total = event_df.groupby('date_only').size().reset_index(name='total_count')
        event_topic = event_topic.merge(event_topic_total, on='date_only')
        event_topic['topic_pct'] = event_topic['topic_count'] / event_topic['total_count'] * 100
        event_topic.to_csv(f'{OUTPUT_DIR}event_window_topic_proportion.csv', index=False, encoding='utf-8-sig')
        print(f"  [SAVE] event_window_topic_proportion.csv (FULL)")

    # --- 2.6 Event window (LOCATED ONLY) ---
    event_located = df_located[
        (df_located['date_only'] >= EVENT_START) & (df_located['date_only'] <= EVENT_END)
    ].copy()
    print(f"  Event window (LOCATED): {len(event_located):,} posts")
    if len(event_located) > 0:
        event_located_global = event_located.groupby(['date_only', 'gmm_tier']).agg(
            panic_mean=('panic_prob', 'mean'),
            post_count=('panic_prob', 'size')
        ).reset_index()
        event_located_global['tier_label'] = event_located_global['gmm_tier'].map(TIER_LABELS)
        event_located_global.to_csv(f'{OUTPUT_DIR}event_window_panic_by_tier_located.csv', index=False, encoding='utf-8-sig')
        print(f"  [SAVE] event_window_panic_by_tier_located.csv (LOCATED)")

    # --- 2.7 Exclude-Tier0 daily (FULL) ---
    df_nonzero = df[df['gmm_tier'] >= 1].copy()
    print(f"\n  Non-Tier0 posts (FULL): {len(df_nonzero):,} ({len(df_nonzero)/len(df)*100:.2f}%)")
    nonzero_daily = df_nonzero.groupby('date_only').agg(
        nonzero_panic_mean=('panic_prob', 'mean'),
        nonzero_posts=('panic_prob', 'size')
    ).reset_index()
    nonzero_daily.to_csv(f'{OUTPUT_DIR}daily_panic_nonzero_tier.csv', index=False, encoding='utf-8-sig')
    print(f"  [SAVE] daily_panic_nonzero_tier.csv (FULL)")

    # --- 2.8 Panic penetration rate (FULL) — 口径A ---
    daily_total = df.groupby('date_only').size().reset_index(name='total_posts')
    daily_high = df[df['gmm_tier'].isin([3,4])].groupby('date_only').size().reset_index(name='high_posts')
    penetration = daily_total.merge(daily_high, on='date_only', how='left').fillna(0)
    penetration['high_penetration_pct'] = penetration['high_posts'] / penetration['total_posts'] * 100
    penetration.to_csv(f'{OUTPUT_DIR}daily_panic_penetration.csv', index=False, encoding='utf-8-sig')
    print(f"  [SAVE] daily_panic_penetration.csv (FULL)")

    # --- 2.9 Merged High/Med/Low daily (FULL) ---
    df['tier_group'] = df['gmm_tier'].map({0: 'Low', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'High'})
    merged_daily = df.groupby(['date_only', 'tier_group']).agg(
        panic_mean=('panic_prob', 'mean'),
        post_count=('panic_prob', 'size')
    ).reset_index()
    merged_daily.to_csv(f'{OUTPUT_DIR}daily_panic_merged_tiers.csv', index=False, encoding='utf-8-sig')
    print(f"  [SAVE] daily_panic_merged_tiers.csv (FULL)")

    # %% 2. 可视化绘图模块（每张图自动弹出到右侧Plots）
    # ------------------------------------------------------------------
    # Phase 3: Visualization
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 3: Visualization")
    print(f"{'='*60}")

    sns.set_style('whitegrid')
    palette = sns.color_palette('tab10', 5)
    tier_group_palette = {'Low': '#2980B9', 'Medium': '#F39C12', 'High': '#C0392B'}

    # --- Fig 1: Five-tier curves (FULL) ---
    if len(event_df) > 0 and event_global is not None:
        fig, ax = plt.subplots(figsize=(14, 8))
        for tier in sorted(event_global['gmm_tier'].unique()):
            tier_data = event_global[event_global['gmm_tier'] == tier].sort_values('date_only')
            label = TIER_LABELS.get(tier, f'Tier{tier}')
            ax.plot(tier_data['date_only'], tier_data['panic_mean'],
                    marker='o', linewidth=2.5, markersize=8, label=label, color=palette[tier])
        ax.axvline(CUTOFF_DT, color='black', linestyle='--', linewidth=2,
                   label='Lockdown Announcement (Jan 23)')
        ax.set_xlabel('Date', fontsize=13)
        ax.set_ylabel('Mean Panic Probability', fontsize=13)
        ax.set_title('Daily Panic Curves by GMM Tier (Jan 18–25, 2020)', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}fig1_tier_panic_curves.png', dpi=300, bbox_inches='tight')
        plt.show()  # 推送至右侧Plots面板
        plt.close()
        print(f"  [SAVE] fig1_tier_panic_curves.png (FULL)")

    # --- Fig 2: Heterogeneous response (FULL) ---
    if len(event_df) > 0 and event_global is not None:
        fig, ax = plt.subplots(figsize=(14, 8))
        event_overall = event_df.groupby('date_only')['panic_prob'].mean().reset_index()
        ax.plot(event_overall['date_only'], event_overall['panic_prob'],
                marker='s', linewidth=3, markersize=10, color='black', label='Overall Mean', zorder=5)
        high_tier = event_global[event_global['gmm_tier'].isin([3,4])].groupby('date_only')['panic_mean'].mean().reset_index()
        ax.plot(high_tier['date_only'], high_tier['panic_mean'],
                marker='^', linewidth=2.5, markersize=8, color='#C0392B', label='High Tier (3+4)', zorder=4)
        low_tier = event_global[event_global['gmm_tier'].isin([0,1])].groupby('date_only')['panic_mean'].mean().reset_index()
        ax.plot(low_tier['date_only'], low_tier['panic_mean'],
                marker='v', linewidth=2.5, markersize=8, color='#2980B9', label='Low Tier (0+1)', zorder=4)
        ax.axvline(CUTOFF_DT, color='black', linestyle='--', linewidth=2)
        ax.text(CUTOFF_DT, ax.get_ylim()[1] * 0.95, 'Lockdown', fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.set_xlabel('Date', fontsize=13)
        ax.set_ylabel('Mean Panic Probability', fontsize=13)
        ax.set_title('Heterogeneous Panic Response to Lockdown Announcement', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}fig2_heterogeneous_response.png', dpi=300, bbox_inches='tight')
        plt.show()  # 推送至右侧Plots面板
        plt.close()
        print(f"  [SAVE] fig2_heterogeneous_response.png (FULL)")

    # --- Fig 2b: Merged High/Med/Low (FULL) ---
    if len(event_df) > 0:
        event_df['tier_group'] = event_df['gmm_tier'].map({0: 'Low', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'High'})
        merged_event = event_df.groupby(['date_only', 'tier_group'])['panic_prob'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(14, 8))
        for group in ['High', 'Medium', 'Low']:
            gdata = merged_event[merged_event['tier_group'] == group].sort_values('date_only')
            ax.plot(gdata['date_only'], gdata['panic_prob'],
                    marker='o', linewidth=3, markersize=10, label=f'{group} Tier', color=tier_group_palette[group])
        ax.axvline(CUTOFF_DT, color='black', linestyle='--', linewidth=2, label='Lockdown')
        ax.set_xlabel('Date', fontsize=13)
        ax.set_ylabel('Mean Panic Probability', fontsize=13)
        ax.set_title('Merged Tier Response (High/Medium/Low) — Jan 18-25', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}fig2b_merged_tier_response.png', dpi=300, bbox_inches='tight')
        plt.show()  # 推送至右侧Plots面板
        plt.close()
        print(f"  [SAVE] fig2b_merged_tier_response.png (FULL)")

    # --- Fig 2c: Panic penetration rate (FULL) ---
    event_pen = penetration[(penetration['date_only'] >= EVENT_START) & (penetration['date_only'] <= EVENT_END)].copy()
    if len(event_pen) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(event_pen['date_only'], event_pen['high_penetration_pct'],
                marker='o', linewidth=3, markersize=10, color='#8E44AD')
        ax.axvline(CUTOFF_DT, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Date', fontsize=13)
        ax.set_ylabel('High-Tier Penetration (%)', fontsize=13)
        ax.set_title('Panic Penetration Rate (Tier3+4 / Total Posts) — Jan 18-25', fontsize=15, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}fig2c_panic_penetration.png', dpi=300, bbox_inches='tight')
        plt.show()  # 推送至右侧Plots面板
        plt.close()
        print(f"  [SAVE] fig2c_panic_penetration.png (FULL)")

    # --- Fig 3: Regional (LOCATED ONLY) ---
    regions_of_interest = ['Wuhan', 'Hubei', 'Tier1', 'Other']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, region in enumerate(regions_of_interest):
        ax = axes[idx]
        region_data = event_located[event_located['region'] == region] if len(event_located) > 0 else pd.DataFrame()
        if len(region_data) == 0:
            ax.set_title(f'{region} (No Data)', fontsize=13)
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
            continue
        region_agg = region_data.groupby(['date_only', 'gmm_tier'])['panic_prob'].mean().reset_index()
        for tier in sorted(region_agg['gmm_tier'].unique()):
            tier_data = region_agg[region_agg['gmm_tier'] == tier].sort_values('date_only')
            label = TIER_LABELS.get(tier, f'Tier{tier}')
            ax.plot(tier_data['date_only'], tier_data['panic_prob'],
                    marker='o', linewidth=2, markersize=6, label=label)
        ax.axvline(CUTOFF_DT, color='black', linestyle='--', linewidth=1.5)
        ax.set_title(region, fontsize=13, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Panic Probability', fontsize=11)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.suptitle('Regional Heterogeneity — LOCATED Posts Only (Jan 18–25)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig3_regional_heterogeneity.png', dpi=300, bbox_inches='tight')
    plt.show()  # 推送至右侧Plots面板
    plt.close()
    print(f"  [SAVE] fig3_regional_heterogeneity.png (LOCATED)")

    # --- Fig 4: Topic trends (FULL) ---
    if len(event_df) > 0 and event_topic is not None:
        key_topics = ['P8_travel', 'P6_info_seek', 'P7_symptom', 'P2_life_supply', 'P5_emotion_vent']
        fig, ax = plt.subplots(figsize=(14, 8))
        for topic in key_topics:
            topic_data = event_topic[event_topic['topic'] == topic].sort_values('date_only')
            if len(topic_data) > 0:
                ax.plot(topic_data['date_only'], topic_data['topic_pct'],
                        marker='o', linewidth=2.5, markersize=8, label=topic)
        ax.axvline(CUTOFF_DT, color='black', linestyle='--', linewidth=2, label='Lockdown')
        ax.set_xlabel('Date', fontsize=13)
        ax.set_ylabel('Topic Proportion (%)', fontsize=13)
        ax.set_title('Daily Topic Proportion Shifts (Jan 18–25)', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}fig4_topic_proportion_trends.png', dpi=300, bbox_inches='tight')
        plt.show()  # 推送至右侧Plots面板
        plt.close()
        print(f"  [SAVE] fig4_topic_proportion_trends.png (FULL)")

    # --- Fig 5: Stacked area (FULL) ---
    if len(event_df) > 0 and event_topic is not None:
        topic_pivot = event_topic.pivot(index='date_only', columns='topic', values='topic_pct').fillna(0)
        top_topics = topic_pivot.sum().nlargest(8).index.tolist()
        topic_pivot_top = topic_pivot[top_topics]
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.stackplot(topic_pivot_top.index,
                     *[topic_pivot_top[col] for col in topic_pivot_top.columns],
                     labels=topic_pivot_top.columns, alpha=0.8)
        ax.axvline(CUTOFF_DT, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Date', fontsize=13)
        ax.set_ylabel('Topic Proportion (%)', fontsize=13)
        ax.set_title('Topic Composition Dynamics (Jan 18–25)', fontsize=15, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}fig5_topic_composition_area.png', dpi=300, bbox_inches='tight')
        plt.show()  # 推送至右侧Plots面板
        plt.close()
        print(f"  [SAVE] fig5_topic_composition_area.png (FULL)")

    # %% 3. 三期统计汇总模块
    # ------------------------------------------------------------------
    # Phase 4: Three-phase summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 4: Three-phase Summary")
    print(f"{'='*60}")

    baseline_s = pd.to_datetime('2020-01-18')
    baseline_e = pd.to_datetime('2020-01-19')
    anticipation_s = pd.to_datetime('2020-01-20')
    anticipation_e = pd.to_datetime('2020-01-22')
    announcement_s = pd.to_datetime('2020-01-23')
    announcement_e = pd.to_datetime('2020-01-25')

    # --- 4.1 FULL sample: by 5 tiers ---
    baseline_df = df[(df['date_only'] >= baseline_s) & (df['date_only'] <= baseline_e)]
    anticipation_df = df[(df['date_only'] >= anticipation_s) & (df['date_only'] <= anticipation_e)]
    announcement_df = df[(df['date_only'] >= announcement_s) & (df['date_only'] <= announcement_e)]

    print(f"  FULL sample — Baseline: {len(baseline_df):,} | Anticipation: {len(anticipation_df):,} | Announcement: {len(announcement_df):,}")

    summary_rows = []
    for tier in range(5):
        b = baseline_df[baseline_df['gmm_tier'] == tier]['panic_prob'].mean()
        a = anticipation_df[anticipation_df['gmm_tier'] == tier]['panic_prob'].mean()
        n = announcement_df[announcement_df['gmm_tier'] == tier]['panic_prob'].mean()
        summary_rows.append({
            'tier': TIER_LABELS[tier],
            'baseline_mean': round(b, 4) if pd.notna(b) else None,
            'anticipation_mean': round(a, 4) if pd.notna(a) else None,
            'announcement_mean': round(n, 4) if pd.notna(n) else None,
            'delta_anticipation': round(a - b, 4) if pd.notna(a) and pd.notna(b) else None,
            'delta_announcement': round(n - a, 4) if pd.notna(n) and pd.notna(a) else None,
            'baseline_n': len(baseline_df[baseline_df['gmm_tier'] == tier]),
            'anticipation_n': len(anticipation_df[anticipation_df['gmm_tier'] == tier]),
            'announcement_n': len(announcement_df[announcement_df['gmm_tier'] == tier]),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{OUTPUT_DIR}three_phase_summary_by_tier.csv', index=False, encoding='utf-8-sig')
    print("\n  --- FULL: By 5 Tiers ---")
    print(summary_df.to_string(index=False))

    # --- 4.2 FULL: merged High/Med/Low ---
    for d in [baseline_df, anticipation_df, announcement_df]:
        d['tier_group'] = d['gmm_tier'].map({0: 'Low', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'High'})
    merged_rows = []
    for group in ['Low', 'Medium', 'High']:
        b = baseline_df[baseline_df['tier_group'] == group]['panic_prob'].mean()
        a = anticipation_df[anticipation_df['tier_group'] == group]['panic_prob'].mean()
        n = announcement_df[announcement_df['tier_group'] == group]['panic_prob'].mean()
        merged_rows.append({
            'tier_group': group,
            'baseline_mean': round(b, 4) if pd.notna(b) else None,
            'anticipation_mean': round(a, 4) if pd.notna(a) else None,
            'announcement_mean': round(n, 4) if pd.notna(n) else None,
            'delta_anticipation': round(a - b, 4) if pd.notna(a) and pd.notna(b) else None,
            'delta_announcement': round(n - a, 4) if pd.notna(n) and pd.notna(a) else None,
            'baseline_n': len(baseline_df[baseline_df['tier_group'] == group]),
            'anticipation_n': len(anticipation_df[anticipation_df['tier_group'] == group]),
            'announcement_n': len(announcement_df[announcement_df['tier_group'] == group]),
        })
    merged_summary = pd.DataFrame(merged_rows)
    merged_summary.to_csv(f'{OUTPUT_DIR}three_phase_summary_merged_tiers.csv', index=False, encoding='utf-8-sig')
    print("\n  --- FULL: Merged High/Med/Low ---")
    print(merged_summary.to_string(index=False))

    # --- 4.3 FULL: panic penetration ---
    def calc_penetration(d):
        total = len(d)
        high = len(d[d['gmm_tier'].isin([3,4])])
        return round(high / total * 100, 4) if total > 0 else None
    pen_rows = [{
        'phase': 'baseline', 'penetration_pct': calc_penetration(baseline_df),
        'total_n': len(baseline_df), 'high_n': len(baseline_df[baseline_df['gmm_tier'].isin([3,4])])
    }, {
        'phase': 'anticipation', 'penetration_pct': calc_penetration(anticipation_df),
        'total_n': len(anticipation_df), 'high_n': len(anticipation_df[anticipation_df['gmm_tier'].isin([3,4])])
    }, {
        'phase': 'announcement', 'penetration_pct': calc_penetration(announcement_df),
        'total_n': len(announcement_df), 'high_n': len(announcement_df[announcement_df['gmm_tier'].isin([3,4])])
    }]
    pen_df = pd.DataFrame(pen_rows)
    pen_df.to_csv(f'{OUTPUT_DIR}three_phase_panic_penetration.csv', index=False, encoding='utf-8-sig')
    print("\n  --- FULL: Panic Penetration ---")
    print(pen_df.to_string(index=False))

    # --- 4.4 LOCATED ONLY: by region ---
    baseline_loc = df_located[(df_located['date_only'] >= baseline_s) & (df_located['date_only'] <= baseline_e)]
    anticipation_loc = df_located[(df_located['date_only'] >= anticipation_s) & (df_located['date_only'] <= anticipation_e)]
    announcement_loc = df_located[(df_located['date_only'] >= announcement_s) & (df_located['date_only'] <= announcement_e)]

    print(f"\n  LOCATED sample — Baseline: {len(baseline_loc):,} | Anticipation: {len(anticipation_loc):,} | Announcement: {len(announcement_loc):,}")

    reg_summary = []
    for region in ['Wuhan', 'Hubei', 'Tier1', 'Other']:
        b = baseline_loc[baseline_loc['region'] == region]['panic_prob'].mean()
        a = anticipation_loc[anticipation_loc['region'] == region]['panic_prob'].mean()
        n = announcement_loc[announcement_loc['region'] == region]['panic_prob'].mean()
        bn = len(baseline_loc[baseline_loc['region'] == region])
        an = len(anticipation_loc[anticipation_loc['region'] == region])
        nn = len(announcement_loc[announcement_loc['region'] == region])
        reg_summary.append({
            'region': region,
            'baseline_mean': round(b, 4) if pd.notna(b) else None,
            'anticipation_mean': round(a, 4) if pd.notna(a) else None,
            'announcement_mean': round(n, 4) if pd.notna(n) else None,
            'delta_anticipation': round(a - b, 4) if pd.notna(a) and pd.notna(b) else None,
            'delta_announcement': round(n - a, 4) if pd.notna(n) and pd.notna(a) else None,
            'baseline_n': bn, 'anticipation_n': an, 'announcement_n': nn,
        })
    reg_summary_df = pd.DataFrame(reg_summary)
    reg_summary_df.to_csv(f'{OUTPUT_DIR}three_phase_summary_by_region.csv', index=False, encoding='utf-8-sig')
    print("\n  --- LOCATED: By Region ---")
    print(reg_summary_df.to_string(index=False))

    # %% 4. 程序收尾
    # ------------------------------------------------------------------
    # Phase 5: Wrap-up
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"✅ All done! Output: {OUTPUT_DIR}")
    print(f"{'='*60}")
    print("生成全部文件清单：")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")

if __name__ == '__main__':
    main()