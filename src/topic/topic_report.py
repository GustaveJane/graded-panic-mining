#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topic_report.py

面向论文输出的汇总脚本。

重点输出三套图表/表格：
1) GMM 层级 × 人工三层框架
2) BERTopic 主题 × GMM 层级
3) P 类恐慌诉求 × GMM 层级

这样可以直接对应你给的参考图风格：
- 表格更规整
- 主题命名更人工化
- 图表更适合正文展示
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(os.environ.get('TOPIC_BASE_DIR', Path(__file__).resolve().parent))
INPUT_DIR = Path(os.environ.get('TOPIC_OUTPUT_DIR', BASE_DIR / 'topic_outputs'))
OUTPUT_DIR = Path(os.environ.get('TOPIC_REPORT_DIR', BASE_DIR / 'topic_reports'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = ['非恐慌', '低恐慌', '中恐慌', '高恐慌']
GMM_ORDER = ['Low', 'Medium', 'High']
TOP_N = 8
TOP_N_TERMS_FOR_TABLE = 10

MANUAL_LEVEL1_ORDER = ['原创情绪表达', '新闻转发/通报', '互助/求助', '闲谈/生活吐槽']
MANUAL_LEVEL2_ORDER = ['国内疫情数据', '国际疫情动态', '政策干预信息', '医疗/科普信息']
PANIC_ORDER = ['医疗资源求助', '生活物资焦虑', '复工/复学焦虑', '隔离/封控影响', '情绪宣泄', '信息寻求', '症状/确诊担忧', '出行/交通焦虑']


def df_to_markdown(df):
    cols = list(df.columns)
    header = '| ' + ' | '.join(str(c) for c in cols) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    rows = []
    for _, row in df.iterrows():
        rows.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join([header, sep] + rows)


def load_required(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'未找到 {path}，请先运行 topic_model.py')
    return pd.read_csv(path)


def format_pct_frame(df):
    return (df * 100).round(1)


def save_heatmap(df, title, path, figsize=(12, 6), cmap='YlOrRd', annot=True):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=annot, fmt='.1f', cmap=cmap, ax=ax, linewidths=0.5, cbar_kws={'label': '占比 (%)'})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_grouped_bar(df, title, path, figsize=(12, 6)):
    n_groups, n_cats = df.shape
    x = np.arange(n_cats)
    width = 0.8 / max(n_groups, 1)
    fig, ax = plt.subplots(figsize=figsize)
    for i, group in enumerate(df.index):
        vals = df.loc[group].values
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=group)
        for xi, v in zip(x + i * width - 0.4 + width / 2, vals):
            if v >= 5:
                ax.text(xi, v + 0.3, f'{v:.0f}', ha='center', va='bottom', fontsize=6)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('占比 (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=30, ha='right')
    ax.legend(title='GMM 层级', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_stacked_bar(df, title, path, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(df.index))
    colors = plt.cm.tab20(np.linspace(0, 1, len(df.columns)))
    for i, col in enumerate(df.columns):
        vals = df[col].values * 100
        ax.bar(df.index, vals, bottom=bottom, label=col, color=colors[i])
        bottom += vals
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('占比 (%)')
    ax.set_xlabel('GMM 层级')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    print('=' * 70)
    print('topic_report.py — 论文可用主题分析汇总')
    print('=' * 70)

    doc_path = INPUT_DIR / 'doc_taxonomy_mapping.csv'
    topic_info_path = INPUT_DIR / 'topic_info.csv'
    taxonomy_path = INPUT_DIR / 'topic_taxonomy_mapping.csv'
    topic_dist_path = INPUT_DIR / 'topic_distribution_by_group.csv'
    l1_dist_path = INPUT_DIR / 'taxonomy_level1_distribution_by_group.csv'
    l2_dist_path = INPUT_DIR / 'taxonomy_level2_distribution_by_group.csv'
    panic_dist_path = INPUT_DIR / 'panic_type_distribution_by_group.csv'
    rep_docs_path = INPUT_DIR / 'topic_representative_docs.csv'

    print('\n【Step 1】读取数据 ...')
    df = load_required(doc_path)
    topic_info = load_required(topic_info_path)
    taxonomy = load_required(taxonomy_path)
    topic_dist = load_required(topic_dist_path)
    l1_dist = load_required(l1_dist_path)
    l2_dist = load_required(l2_dist_path)
    panic_dist = load_required(panic_dist_path)
    rep_docs = load_required(rep_docs_path)

    df = df[df['risk_group_name'].isin(GROUP_ORDER)].copy()
    df = df[df['topic'] != -1].copy()

    print(f'  有效文档数: {len(df):,}')
    print(f'  主题数: {df["topic"].nunique()}')

    # ---------- 1) 表：GMM 层级 × 人工三层框架 ----------
    print('\n【Step 2】生成 GMM 层级 × 人工三层框架表 ...')
    l1 = pd.crosstab(df['risk_group_name'], df['content_nature_cn'], normalize='index')
    l1 = l1.reindex(index=GROUP_ORDER, columns=MANUAL_LEVEL1_ORDER, fill_value=0)
    l1_pct = format_pct_frame(l1)
    l1_pct.to_csv(OUTPUT_DIR / 'figure_table_level1.csv', encoding='utf-8-sig')

    # B 类二级分类
    bdf = df[df['content_nature'] == 'B'].copy()
    l2 = pd.crosstab(bdf['risk_group_name'], bdf['nature_detail_cn'], normalize='index')
    l2 = l2.reindex(index=GROUP_ORDER, columns=MANUAL_LEVEL2_ORDER, fill_value=0)
    l2_pct = format_pct_frame(l2)
    l2_pct.to_csv(OUTPUT_DIR / 'figure_table_level2_B.csv', encoding='utf-8-sig')

    pdf = pd.crosstab(df[df['panic_type'] != '']['risk_group_name'], df[df['panic_type'] != '']['panic_type_cn'], normalize='index')
    pdf = pdf.reindex(index=GROUP_ORDER, columns=PANIC_ORDER, fill_value=0)
    pdf_pct = format_pct_frame(pdf)
    pdf_pct.to_csv(OUTPUT_DIR / 'figure_table_panic_types.csv', encoding='utf-8-sig')

    # 让表格更像参考图：再输出一个“总览表”
    overview_rows = []
    for g in GROUP_ORDER:
        top_l1 = l1_pct.loc[g].idxmax() if g in l1_pct.index else ''
        top_b = l2_pct.loc[g].idxmax() if g in l2_pct.index else ''
        top_p = pdf_pct.loc[g].idxmax() if g in pdf_pct.index else ''
        overview_rows.append({'GMM层级': g, '主导内容性质': top_l1, '主导新闻细分': top_b, '主导恐慌诉求': top_p})
    overview_df = pd.DataFrame(overview_rows)
    overview_df.to_csv(OUTPUT_DIR / 'figure_overview_table.csv', index=False, encoding='utf-8-sig')

    # ---------- 2) 图：GMM 层级 × 三层框架 ----------
    print('\n【Step 3】生成热力图与堆叠图 ...')
    save_heatmap(l1_pct, 'GMM 层级 × 内容性质（A/B/C/D）', OUTPUT_DIR / 'gmm_x_level1_heatmap.png', figsize=(11, 4.8))
    save_heatmap(l2_pct, 'GMM 层级 × B 类细分（仅新闻类）', OUTPUT_DIR / 'gmm_x_level2_heatmap.png', figsize=(11, 4.8))
    save_heatmap(pdf_pct, 'GMM 层级 × P 类恐慌诉求', OUTPUT_DIR / 'gmm_x_panic_heatmap.png', figsize=(13, 5.6))
    save_stacked_bar(l1, 'GMM 层级 × 内容性质堆叠图', OUTPUT_DIR / 'gmm_x_level1_stacked.png', figsize=(10, 5.8))
    save_grouped_bar(l1_pct, 'GMM 层级 × 内容性质分组图', OUTPUT_DIR / 'gmm_x_level1_grouped.png', figsize=(11, 5.2))
    save_grouped_bar(pdf_pct, 'GMM 层级 × P 类恐慌诉求分组图', OUTPUT_DIR / 'gmm_x_panic_grouped.png', figsize=(13, 5.2))

    # ---------- 3) 图：BERTopic 主题 × GMM 层级 ----------
    print('\n【Step 4】生成 BERTopic 主题 × GMM 图 ...')
    # 只保留更“美观”的主题：数量够大、且 topic_taxonomy 中有清晰标签
    topic_tax = taxonomy[taxonomy['topic_id'] != -1].copy()
    topic_tax = topic_tax.sort_values(['doc_count', 'share'], ascending=False)
    top_topics = topic_tax['topic_id'].head(TOP_N).tolist()
    topic_display = topic_tax.set_index('topic_id').loc[top_topics][['topic_name', 'content_nature_cn', 'nature_detail_cn', 'doc_count']].copy()
    topic_display['display'] = topic_display.apply(
        lambda r: f'T{r.name} {str(r["topic_name"])[:14]}', axis=1
    )
    topic_labels = topic_display['display'].tolist()

    topic_cross = pd.crosstab(df[df['topic'].isin(top_topics)]['topic'], df[df['topic'].isin(top_topics)]['risk_group_name'], normalize='columns')
    topic_cross = topic_cross.reindex(index=top_topics, columns=GROUP_ORDER, fill_value=0)
    topic_cross.index = topic_labels
    topic_cross = topic_cross[GROUP_ORDER] * 100
    topic_cross.to_csv(OUTPUT_DIR / 'figure_table_topic_x_gmm.csv', encoding='utf-8-sig')
    save_heatmap(topic_cross.T, 'BERTopic 主题 × GMM 层级（Top 8）', OUTPUT_DIR / 'topic_x_gmm_heatmap.png', figsize=(12, 5.5))
    save_grouped_bar(topic_cross.T, 'BERTopic 主题 × GMM 层级（Top 8）', OUTPUT_DIR / 'topic_x_gmm_grouped.png', figsize=(13, 5.5))

    # ---------- 4) 图：P 类恐慌诉求 × GMM 层级 ----------
    print('\n【Step 5】生成 P 类恐慌诉求 × GMM 图 ...')
    panic_counts = df[df['panic_type_cn'] != '']['panic_type_cn'].value_counts()
    top_p = [x for x in PANIC_ORDER if x in panic_counts.index]
    panic_cross = pd.crosstab(df[df['panic_type_cn'].isin(top_p)]['panic_type_cn'], df[df['panic_type_cn'].isin(top_p)]['risk_group_name'], normalize='columns')
    panic_cross = panic_cross.reindex(index=top_p, columns=GROUP_ORDER, fill_value=0)
    panic_cross = panic_cross[GROUP_ORDER] * 100
    panic_cross.to_csv(OUTPUT_DIR / 'figure_table_panic_x_gmm.csv', encoding='utf-8-sig')
    save_heatmap(panic_cross.T, 'P 类恐慌诉求 × GMM 层级', OUTPUT_DIR / 'panic_x_gmm_heatmap.png', figsize=(12, 5.5))
    save_grouped_bar(panic_cross.T, 'P 类恐慌诉求 × GMM 层级', OUTPUT_DIR / 'panic_x_gmm_grouped.png', figsize=(13, 5.5))

    # ---------- 5) 宏观摘要表 ----------
    print('\n【Step 6】生成论文摘要表 ...')
    summary_rows = []
    for g in GROUP_ORDER:
        sub = df[df['risk_group_name'] == g]
        l1_sub = sub['content_nature_cn'].value_counts(normalize=True)
        top_l1 = l1_sub.index[0] if len(l1_sub) > 0 else ''
        top_l1_pct = float(l1_sub.iloc[0]) if len(l1_sub) > 0 else 0
        panic_sub = sub[sub['panic_type_cn'] != '']['panic_type_cn'].value_counts(normalize=True)
        top_panic = panic_sub.index[0] if len(panic_sub) > 0 else ''
        top_panic_pct = float(panic_sub.iloc[0]) if len(panic_sub) > 0 else 0
        summary_rows.append({
            'GMM层级': g,
            '主导内容性质': top_l1,
            '占比': round(top_l1_pct * 100, 1),
            '主导恐慌诉求': top_panic,
            '占比2': round(top_panic_pct * 100, 1),
            '主题数': int(sub['topic'].nunique()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / 'gmm_summary_table.csv', index=False, encoding='utf-8-sig')

    # ---------- 6) 输出 Markdown 报告 ----------
    print('\n【Step 7】生成 Markdown 报告 ...')
    report_lines = []
    report_lines.append('# 主题分析结果汇总\n')
    report_lines.append('## 1. GMM 层级 × 人工三层框架\n')
    report_lines.append('### 1.1 内容性质（A/B/C/D）\n')
    report_lines.append(df_to_markdown(l1_pct.reset_index().rename(columns={'index': 'GMM层级'})))
    report_lines.append('\n### 1.2 B 类细分\n')
    report_lines.append(df_to_markdown(l2_pct.reset_index().rename(columns={'index': 'GMM层级'})))
    report_lines.append('\n### 1.3 P 类恐慌诉求\n')
    report_lines.append(df_to_markdown(pdf_pct.reset_index().rename(columns={'index': 'GMM层级'})))
    report_lines.append('\n## 2. GMM 概览表\n')
    report_lines.append(df_to_markdown(summary_df))

    report_lines.append('\n## 3. 论文写作建议\n')
    report_lines.append('- 正文优先使用宏观框架：非恐慌 vs 恐慌，以及 GMM 层级之间的递进关系。')
    report_lines.append('- 图表优先采用热力图和分组柱状图，标题尽量采用人工归并后的中文标签。')
    report_lines.append('- BERTopic 细主题可放在附录，正文不要展示过多碎主题。')
    report_lines.append('- 如需更美观，可直接在 PPT/论文中使用 `figure_overview_table.csv`、`gmm_x_level1_heatmap.png`、`gmm_x_panic_heatmap.png`、`topic_x_gmm_heatmap.png` 三类核心图。')

    report_path = OUTPUT_DIR / 'topic_analysis_summary.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print('  topic_analysis_summary.md 已保存')

    print('\n✅ topic_report.py 完成')
    print(f'  输出目录: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
