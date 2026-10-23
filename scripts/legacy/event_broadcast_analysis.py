#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
event_broadcast_analysis.py

封城事件广播效应分析：
- 风声期：1/20 - 1/22（“人传人”认知转折）
- 公告期：1/23 00:00 - 10:00（封城正式广播）
- 执行期：1/23 10:00 - 1/26
- 稳定期：1/27 - 2/10（可调整）

支持三类分析：
1) 总体样本：按日/阶段看恐慌与主题结构变化
2) 平衡用户面板：识别个体恐慌是否真实下降
3) 主题迁移：识别恐慌结构如何变化

输入依赖：
- topic_outputs/doc_taxonomy_mapping.csv
- topic_outputs/topic_taxonomy_mapping.csv
- topic_outputs/topic_representative_docs.csv（可选）

输出：
- event_analysis_outputs/*
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(os.environ.get('TOPIC_BASE_DIR', Path(__file__).resolve().parent))
INPUT_DIR = Path(os.environ.get('TOPIC_OUTPUT_DIR', BASE_DIR / 'topic_outputs'))
OUTPUT_DIR = Path(os.environ.get('EVENT_OUTPUT_DIR', BASE_DIR / 'event_analysis_outputs'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 事件时间窗口
# -----------------------------
BASELINE_START = pd.Timestamp('2019-12-01')
BASELINE_END = pd.Timestamp('2020-01-19 23:59:59')
SIGNAL_START = pd.Timestamp('2020-01-20')
SIGNAL_END = pd.Timestamp('2020-01-22 23:59:59')
ANNOUNCE_START = pd.Timestamp('2020-01-23 00:00:00')
ANNOUNCE_END = pd.Timestamp('2020-01-23 09:59:59')
IMPLEMENT_START = pd.Timestamp('2020-01-23 10:00:00')
IMPLEMENT_END = pd.Timestamp('2020-01-26 23:59:59')
STABLE_START = pd.Timestamp('2020-01-27 00:00:00')
STABLE_END = pd.Timestamp('2020-02-10 23:59:59')

WINDOW_LABELS = ['baseline', 'signal', 'announce', 'implement', 'stable']
WINDOW_CN = {
    'baseline': '基线期',
    'signal': '风声期',
    'announce': '公告期',
    'implement': '执行期',
    'stable': '稳定期',
}

GMM_ORDER = ['非恐慌', '低恐慌', '中恐慌', '高恐慌']
PANIC_ORDER = ['医疗资源求助', '生活物资焦虑', '复工/复学焦虑', '隔离/封控影响', '情绪宣泄', '信息寻求', '症状/确诊担忧', '出行/交通焦虑']
LEVEL1_ORDER = ['原创情绪表达', '新闻转发/通报', '互助/求助', '闲谈/生活吐槽']
LEVEL2_ORDER = ['国内疫情数据', '国际疫情动态', '政策干预信息', '医疗/科普信息']

# -----------------------------
# 工具函数
# -----------------------------
def load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'未找到 {path}，请先运行 topic_model.py')
    return pd.read_csv(path)


def to_dt(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors='coerce')
    return dt


def assign_window(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return 'unknown'
    if BASELINE_START <= ts <= BASELINE_END:
        return 'baseline'
    if SIGNAL_START <= ts <= SIGNAL_END:
        return 'signal'
    if ANNOUNCE_START <= ts <= ANNOUNCE_END:
        return 'announce'
    if IMPLEMENT_START <= ts <= IMPLEMENT_END:
        return 'implement'
    if STABLE_START <= ts <= STABLE_END:
        return 'stable'
    return 'outside'


def safe_mean(x):
    return float(np.nanmean(x)) if len(x) else np.nan


def save_lineplot(df: pd.DataFrame, x: str, y: str, title: str, path: Path, hue: str | None = None, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    if hue is None:
        sns.lineplot(data=df, x=x, y=y, marker='o', ax=ax)
    else:
        sns.lineplot(data=df, x=x, y=y, hue=hue, marker='o', ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, title: str, path: Path, figsize=(12, 6), fmt='.1f'):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=fmt, cmap='YlOrRd', ax=ax, linewidths=0.5, cbar_kws={'label': '占比/均值'})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.div(df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


@dataclass
class AnalysisConfig:
    min_user_posts: int = 5
    min_days_spanned: int = 3
    max_posts_per_user_per_day: int = 3
    min_before_posts: int = 2
    min_after_posts: int = 2


# -----------------------------
# 主流程
# -----------------------------
def main():
    print('=' * 80)
    print('event_broadcast_analysis.py — 封城事件广播效应分析')
    print('=' * 80)

    doc_path = INPUT_DIR / 'doc_taxonomy_mapping.csv'
    tax_path = INPUT_DIR / 'topic_taxonomy_mapping.csv'

    df = load_required(doc_path)
    tax = load_required(tax_path)

    if 'created_at' in df.columns:
        df['date'] = to_dt(df['created_at'])
    elif 'date' in df.columns:
        df['date'] = to_dt(df['date'])
    elif 'parsed_time' in df.columns:
        df['date'] = to_dt(df['parsed_time'])
    else:
        raise ValueError('缺少时间列：需要 created_at / date / parsed_time 之一')

    if 'content' not in df.columns:
        raise ValueError('缺少 content 列')
    if 'user_id' not in df.columns:
        raise ValueError('缺少 user_id 列，请先确保输入数据带 user_id')
    if 'panic_prob' not in df.columns:
        raise ValueError('缺少 panic_prob 列')

    df = df.dropna(subset=['date']).copy()
    df['window'] = df['date'].apply(assign_window)
    df = df[df['window'] != 'outside'].copy()

    # 只保留较有用的列
    keep_cols = ['user_id', 'date', 'window', 'content', 'panic_prob', 'topic', 'content_nature_cn', 'nature_detail_cn', 'panic_type_cn', 'risk_group_name']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # 保存原始分阶段统计
    print('\n【Step 1】总体样本阶段分布 ...')
    phase_order = ['baseline', 'signal', 'announce', 'implement', 'stable']
    phase_df = df[df['window'].isin(phase_order)].copy()
    overall_summary = []
    for w in phase_order:
        sub = phase_df[phase_df['window'] == w].copy()
        if len(sub) == 0:
            continue
        overall_summary.append({
            'window': WINDOW_CN[w],
            'n_posts': len(sub),
            'n_users': sub['user_id'].nunique(),
            'mean_panic': sub['panic_prob'].mean(),
            'median_panic': sub['panic_prob'].median(),
            'high_panic_share': (sub['panic_prob'] >= 0.8).mean(),
            'mid_high_panic_share': (sub['panic_prob'] >= 0.6).mean(),
            'help_share': (sub['panic_type_cn'] == '医疗资源求助').mean() if 'panic_type_cn' in sub.columns else np.nan,
            'emotion_share': (sub['panic_type_cn'] == '情绪宣泄').mean() if 'panic_type_cn' in sub.columns else np.nan,
        })
    overall_summary_df = pd.DataFrame(overall_summary)
    overall_summary_df.to_csv(OUTPUT_DIR / 'overall_phase_summary.csv', index=False, encoding='utf-8-sig')

    # 按天统计
    print('\n【Step 2】日度趋势 ...')
    phase_df['day'] = phase_df['date'].dt.floor('D')
    daily = phase_df.groupby('day').agg(
        n_posts=('content', 'size'),
        n_users=('user_id', 'nunique'),
        mean_panic=('panic_prob', 'mean'),
        median_panic=('panic_prob', 'median'),
        high_panic_share=('panic_prob', lambda s: (s >= 0.8).mean()),
        mid_high_panic_share=('panic_prob', lambda s: (s >= 0.6).mean()),
    ).reset_index()
    daily.to_csv(OUTPUT_DIR / 'daily_panic_trend.csv', index=False, encoding='utf-8-sig')

    save_lineplot(daily, 'day', 'mean_panic', '日均恐慌概率趋势', OUTPUT_DIR / 'daily_mean_panic.png', figsize=(13, 5.5))
    save_lineplot(daily, 'day', 'high_panic_share', '高恐慌文本比例趋势', OUTPUT_DIR / 'daily_high_panic_share.png', figsize=(13, 5.5))
    save_lineplot(daily, 'day', 'n_posts', '日发帖量趋势', OUTPUT_DIR / 'daily_post_volume.png', figsize=(13, 5.5))
    save_lineplot(daily, 'day', 'n_users', '日活跃用户数趋势', OUTPUT_DIR / 'daily_active_users.png', figsize=(13, 5.5))

    # 阶段主题结构
    print('\n【Step 3】总体样本阶段主题结构 ...')
    def phase_topic_table(col, order=None):
        tmp = phase_df[phase_df[col].notna()].copy() if col in phase_df.columns else pd.DataFrame()
        if len(tmp) == 0:
            return pd.DataFrame()
        tab = pd.crosstab(tmp['window'], tmp[col], normalize='index') * 100
        tab = tab.reindex(index=phase_order, fill_value=0)
        if order is not None:
            tab = tab.reindex(columns=order, fill_value=0)
        return tab

    if 'content_nature_cn' in phase_df.columns:
        l1_phase = phase_topic_table('content_nature_cn', LEVEL1_ORDER)
        l1_phase.to_csv(OUTPUT_DIR / 'phase_level1_distribution.csv', encoding='utf-8-sig')
        save_heatmap(l1_phase, '阶段 × 内容性质（%）', OUTPUT_DIR / 'phase_level1_heatmap.png', figsize=(11, 4.8))

    if 'panic_type_cn' in phase_df.columns:
        p_phase = phase_topic_table('panic_type_cn', PANIC_ORDER)
        p_phase.to_csv(OUTPUT_DIR / 'phase_panic_distribution.csv', encoding='utf-8-sig')
        save_heatmap(p_phase, '阶段 × 恐慌诉求（%）', OUTPUT_DIR / 'phase_panic_heatmap.png', figsize=(13, 5.6))

    # 时间序列断点对比：风声期、公告期、执行期
    print('\n【Step 4】事件窗口对比 ...')
    window_summary = phase_df.groupby('window').agg(
        n_posts=('content', 'size'),
        n_users=('user_id', 'nunique'),
        mean_panic=('panic_prob', 'mean'),
        high_panic_share=('panic_prob', lambda s: (s >= 0.8).mean()),
        p5_share=('panic_type_cn', lambda s: (s == '情绪宣泄').mean() if len(s) else np.nan),
        p6_share=('panic_type_cn', lambda s: (s == '信息寻求').mean() if len(s) else np.nan),
        p7_share=('panic_type_cn', lambda s: (s == '症状/确诊担忧').mean() if len(s) else np.nan),
        p4_share=('panic_type_cn', lambda s: (s == '隔离/封控影响').mean() if len(s) else np.nan),
        p2_share=('panic_type_cn', lambda s: (s == '生活物资焦虑').mean() if len(s) else np.nan),
    ).reindex(phase_order)
    window_summary.to_csv(OUTPUT_DIR / 'window_summary.csv', encoding='utf-8-sig')

    # 平衡用户面板
    print('\n【Step 5】构建平衡用户面板 ...')
    cfg = AnalysisConfig()
    user_day = phase_df.copy()
    user_day['day'] = user_day['date'].dt.floor('D')
    user_day = user_day.sort_values(['user_id', 'date'])
    # 每用户每天最多保留3条，减少高频用户偏置
    user_day = user_day.groupby(['user_id', 'day'], group_keys=False).head(cfg.max_posts_per_user_per_day).copy()

    user_phase = user_day.groupby(['user_id', 'window']).agg(
        n_posts=('content', 'size'),
        mean_panic=('panic_prob', 'mean'),
        high_panic_share=('panic_prob', lambda s: (s >= 0.8).mean()),
        p5_share=('panic_type_cn', lambda s: (s == '情绪宣泄').mean() if len(s) else np.nan),
        p6_share=('panic_type_cn', lambda s: (s == '信息寻求').mean() if len(s) else np.nan),
        p7_share=('panic_type_cn', lambda s: (s == '症状/确诊担忧').mean() if len(s) else np.nan),
        p4_share=('panic_type_cn', lambda s: (s == '隔离/封控影响').mean() if len(s) else np.nan),
        p2_share=('panic_type_cn', lambda s: (s == '生活物资焦虑').mean() if len(s) else np.nan),
    ).reset_index()

    # 只保留前后都出现过的用户：基线/风声 + 执行/稳定 至少各有若干条
    user_counts = user_phase.groupby('user_id').agg(
        n_total=('n_posts', 'sum'),
        n_windows=('window', 'nunique')
    )
    before_counts = user_phase[user_phase['window'].isin(['baseline', 'signal'])].groupby('user_id')['n_posts'].sum()
    after_counts = user_phase[user_phase['window'].isin(['announce', 'implement', 'stable'])].groupby('user_id')['n_posts'].sum()
    panel_users = user_counts[(user_counts['n_total'] >= cfg.min_user_posts) & (user_counts['n_windows'] >= 2)].index
    panel_users = [u for u in panel_users if before_counts.get(u, 0) >= cfg.min_before_posts and after_counts.get(u, 0) >= cfg.min_after_posts]
    panel = user_phase[user_phase['user_id'].isin(panel_users)].copy()
    panel.to_csv(OUTPUT_DIR / 'balanced_user_panel.csv', index=False, encoding='utf-8-sig')

    # 个体变化
    panel_before = panel[panel['window'].isin(['baseline', 'signal'])].groupby('user_id').agg(before_panic=('mean_panic', 'mean')).reset_index()
    panel_after = panel[panel['window'].isin(['announce', 'implement', 'stable'])].groupby('user_id').agg(after_panic=('mean_panic', 'mean')).reset_index()
    panel_change = panel_before.merge(panel_after, on='user_id', how='inner')
    panel_change['delta_panic'] = panel_change['after_panic'] - panel_change['before_panic']
    panel_change.to_csv(OUTPUT_DIR / 'balanced_user_change.csv', index=False, encoding='utf-8-sig')

    # 迁移矩阵：按用户前后主导恐慌层级
    def phase_label_map(mean_panic: float) -> str:
        if pd.isna(mean_panic):
            return '缺失'
        if mean_panic < 0.25:
            return '非恐慌'
        if mean_panic < 0.5:
            return '低恐慌'
        if mean_panic < 0.75:
            return '中恐慌'
        return '高恐慌'

    panel_change['before_level'] = panel_change['before_panic'].map(phase_label_map)
    panel_change['after_level'] = panel_change['after_panic'].map(phase_label_map)
    transition = pd.crosstab(panel_change['before_level'], panel_change['after_level']).reindex(index=GMM_ORDER, columns=GMM_ORDER, fill_value=0)
    transition.to_csv(OUTPUT_DIR / 'user_level_transition.csv', encoding='utf-8-sig')
    transition_pct = normalize_rows(transition) * 100
    save_heatmap(transition_pct, '平衡用户：公告前后恐慌层级迁移（%）', OUTPUT_DIR / 'user_level_transition_heatmap.png', figsize=(7.5, 6.0))

    # 个体变化分布
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(panel_change['delta_panic'], bins=30, kde=True, ax=ax)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title('平衡用户：公告前后恐慌变化分布', fontsize=14)
    ax.set_xlabel('后期均值 - 前期均值')
    ax.set_ylabel('人数')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'balanced_user_delta_hist.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    # 主题迁移：公告前后主导主题
    print('\n【Step 6】主题迁移分析 ...')
    # 使用主题级映射，方便前后比较
    def top_label(sub: pd.DataFrame, col: str) -> str:
        if col not in sub.columns or len(sub) == 0:
            return '缺失'
        vc = sub[col].value_counts()
        return vc.index[0] if len(vc) else '缺失'

    user_topic = phase_df.copy()
    before_sub = user_topic[user_topic['window'].isin(['baseline', 'signal'])].copy()
    after_sub = user_topic[user_topic['window'].isin(['announce', 'implement', 'stable'])].copy()

    topic_migration_rows = []
    for uid, g in before_sub.groupby('user_id'):
        if uid not in set(after_sub['user_id']):
            continue
        g_after = after_sub[after_sub['user_id'] == uid]
        before_main_nature = top_label(g, 'content_nature_cn') if 'content_nature_cn' in g.columns else '缺失'
        after_main_nature = top_label(g_after, 'content_nature_cn') if 'content_nature_cn' in g_after.columns else '缺失'
        before_main_panic = top_label(g, 'panic_type_cn') if 'panic_type_cn' in g.columns else '缺失'
        after_main_panic = top_label(g_after, 'panic_type_cn') if 'panic_type_cn' in g_after.columns else '缺失'
        topic_migration_rows.append({
            'user_id': uid,
            'before_main_nature': before_main_nature,
            'after_main_nature': after_main_nature,
            'before_main_panic': before_main_panic,
            'after_main_panic': after_main_panic,
            'before_mean_panic': g['panic_prob'].mean(),
            'after_mean_panic': g_after['panic_prob'].mean(),
        })
    topic_migration = pd.DataFrame(topic_migration_rows)
    topic_migration.to_csv(OUTPUT_DIR / 'topic_migration_users.csv', index=False, encoding='utf-8-sig')

    if len(topic_migration) > 0:
        mig_tab = pd.crosstab(topic_migration['before_main_panic'], topic_migration['after_main_panic'])
        mig_tab.to_csv(OUTPUT_DIR / 'panic_topic_migration_matrix.csv', encoding='utf-8-sig')
        mig_pct = normalize_rows(mig_tab.reindex(index=PANIC_ORDER, columns=PANIC_ORDER, fill_value=0)) * 100
        save_heatmap(mig_pct, '公告前后主导恐慌诉求迁移（%）', OUTPUT_DIR / 'panic_topic_migration_heatmap.png', figsize=(10, 7))

    # 传播环境：风声词/官方词/传言词
    print('\n【Step 7】传播环境分析 ...')
    rumor_kw = ['听说', '据说', '传言', '小道消息', '风声', '群里', '朋友圈', '真的假的', '消息说', '网传', '谣言']
    official_kw = ['通报', '发布会', '卫健委', '疾控中心', '专家', '官方', '公告', '通知', '人民日报', '新华社', '央视']
    recognition_kw = ['人传人', '肯定人传人', '会传染', '传染性', '可人传人', '明确人传人', '钟南山']

    phase_df['rumor_flag'] = phase_df['content'].astype(str).apply(lambda x: any(k in x for k in rumor_kw))
    phase_df['official_flag'] = phase_df['content'].astype(str).apply(lambda x: any(k in x for k in official_kw))
    phase_df['recognition_flag'] = phase_df['content'].astype(str).apply(lambda x: any(k in x for k in recognition_kw))
    env = phase_df.groupby('window').agg(
        rumor_share=('rumor_flag', 'mean'),
        official_share=('official_flag', 'mean'),
        recognition_share=('recognition_flag', 'mean'),
        retweet_ratio=('content', lambda s: s.astype(str).str.contains('转发理由|转发微博').mean()),
    ).reindex(phase_order)
    env.to_csv(OUTPUT_DIR / 'communication_environment.csv', encoding='utf-8-sig')

    # 输出一个总报告
    print('\n【Step 8】生成汇总报告 ...')
    report = []
    report.append('# 封城事件广播效应分析摘要\n')
    report.append('## 事件分段')
    report.append('- 基线期：2019-12-01 至 2020-01-19')
    report.append('- 风声期：2020-01-20 至 2020-01-22（“人传人”认知转折）')
    report.append('- 公告期：2020-01-23 00:00 至 09:59（封城广播）')
    report.append('- 执行期：2020-01-23 10:00 至 2020-01-26')
    report.append('- 稳定期：2020-01-27 至 2020-02-10\n')
    report.append('## 核心输出文件')
    report.append('- `overall_phase_summary.csv`')
    report.append('- `daily_panic_trend.csv`')
    report.append('- `window_summary.csv`')
    report.append('- `balanced_user_panel.csv`')
    report.append('- `balanced_user_change.csv`')
    report.append('- `user_level_transition.csv`')
    report.append('- `topic_migration_users.csv`')
    report.append('- `communication_environment.csv`\n')
    report.append('## 结果解释提示')
    report.append('- 若风声期 already 上升，则说明“人传人”认知是重要拐点。')
    report.append('- 若公告期短暂震荡、执行期主题转向封控/物资/求助，说明封城广播改变了恐慌结构。')
    report.append('- 若平衡用户面板也显示 delta_panic < 0，则说明存在真实个体下降。')
    report.append('- 若总体下降但平衡面板不显著，则主要是群体构成变化。')
    report.append('- 若主题迁移从 P7/P5 转向 P4/B3/A3，则说明恐慌由情绪化担忧转为政策适应。')
    with open(OUTPUT_DIR / 'event_broadcast_summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print('\n✅ 封城事件广播效应分析完成')
    print(f'输出目录: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
