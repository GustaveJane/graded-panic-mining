#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_doc_taxonomy.py

复用远端已训练好的 BERTopic 主题结果（doc_topic_mapping.csv，含 doc_id/content/
risk_group/risk_group_name/panic_prob/topic），仅补上 event_broadcast_analysis.py
所需的人工分类字段与时间戳/user_id，生成完整的 doc_taxonomy_mapping.csv。

核心价值：不重训 BERTopic -> 主题编号与论文既有结果保持一致，且节省 GPU 时间。

输入：
- topic_outputs/doc_topic_mapping.csv   (已有主题)
- full_gmm_stratified.csv              (全量原始语料,用于回填 created_at/user_id)
输出：
- topic_outputs/doc_taxonomy_mapping.csv
  (含 content_nature/content_nature_cn/nature_detail/nature_detail_cn/
   panic_type/panic_type_cn/created_at/user_id)
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from topic_model import (
    clean_text,
    classify_content_nature,
    classify_nature_detail,
    classify_panic_type,
    NATURE_CN,
    NATURE_DETAIL_CN,
    PANIC_CN,
)

BASE_DIR = Path(os.environ.get('TOPIC_BASE_DIR', Path(__file__).resolve().parent))
INPUT_DIR = Path(os.environ.get('TOPIC_OUTPUT_DIR', BASE_DIR / 'topic_outputs'))
DTM_PATH = INPUT_DIR / 'doc_topic_mapping.csv'
OUT_PATH = INPUT_DIR / 'doc_taxonomy_mapping.csv'
FULL_PATH = Path(os.environ.get('FULL_TABLE_PATH', BASE_DIR / 'full_gmm_stratified.csv'))


def backfill_meta(df: pd.DataFrame) -> pd.DataFrame:
    """按 clean_text 后的 content 从全量原始语料反查 created_at / user_id。"""
    if not FULL_PATH.exists():
        print(f'  [meta] 未找到 {FULL_PATH}，跳过')
        df['created_at'] = pd.NaT
        df['user_id'] = ''
        return df

    needed_set = set(df['content'].map(clean_text).dropna())
    print(f'  [meta] 需回填唯一文本数: {len(needed_set):,}，来源: {FULL_PATH.name}')

    lookup = {}
    remaining = set(needed_set)
    n_total = len(needed_set)
    start = None
    for i, chunk in enumerate(pd.read_csv(
        FULL_PATH, chunksize=500000, low_memory=False,
        usecols=lambda c: c in ('content', 'created_at', 'user_id'),
    )):
        if start is None:
            start = __import__('time').time()
        cc = chunk['content'].map(clean_text)
        # 只处理仍未命中的文本
        cand_mask = cc.isin(remaining)
        if cand_mask.any():
            sub = chunk.loc[cand_mask, ['created_at', 'user_id']].copy()
            sub['_cc'] = cc[cand_mask].values
            sub = sub.drop_duplicates('_cc')
            for _, r in sub.iterrows():
                if r['_cc'] not in lookup:
                    lookup[r['_cc']] = (r['created_at'], r['user_id'])
            remaining -= set(sub['_cc'])
        elapsed = __import__('time').time() - start
        speed = (i + 1) * 500000 / elapsed if elapsed > 0 else 0
        print(f'  [meta] 块{i+1} 已索引 {len(lookup):,}/{n_total} 剩余{len(remaining):,} '
              f'({elapsed:.0f}s, {speed/1e6:.1f}M行/s)', end='\r')
        if not remaining:
            print('\n  [meta] 全部命中，提前退出')
            break
    print()

    meta = df['content'].map(clean_text).map(lambda cc: lookup.get(cc, (pd.NaT, '')))
    df['created_at'] = [m[0] for m in meta]
    df['user_id'] = [m[1] for m in meta]
    hit = df['user_id'].astype(str).str.len().gt(0).sum()
    print(f'  [meta] 回填完成: {hit:,}/{len(df):,} 行命中 user_id ({hit/len(df):.1%})')
    return df


def main():
    print('=' * 70)
    print('build_doc_taxonomy.py — 复用主题，补分类字段 + 时间戳/user_id')
    print('=' * 70)

    if not DTM_PATH.exists():
        raise FileNotFoundError(f'未找到 {DTM_PATH}')
    df = pd.read_csv(DTM_PATH)
    print(f'  读取 doc_topic_mapping.csv: {len(df):,} 行')

    print('\n【Step 1】生成人工分类字段 ...')
    df['content_nature'] = df['content'].map(clean_text).map(classify_content_nature)
    df['nature_detail'] = df.apply(
        lambda r: classify_nature_detail(r['content'], r['content_nature']), axis=1)
    df['panic_type'] = df['content'].map(clean_text).map(classify_panic_type)
    df['content_nature_cn'] = df['content_nature'].map(NATURE_CN)
    df['nature_detail_cn'] = df['nature_detail'].map(NATURE_DETAIL_CN)
    df['panic_type_cn'] = df['panic_type'].map(PANIC_CN).fillna('')
    print('  分类字段生成完成')
    print('  content_nature 分布:', df['content_nature_cn'].value_counts().to_dict())
    print('  panic_type_cn   分布(非空):', df[df['panic_type_cn'] != '']['panic_type_cn'].value_counts().to_dict())

    print('\n【Step 2】回填 created_at / user_id ...')
    df = backfill_meta(df)

    print('\n【Step 3】保存 doc_taxonomy_mapping.csv ...')
    out_cols = [
        'doc_id', 'content', 'risk_group', 'risk_group_name', 'panic_prob', 'topic',
        'content_nature', 'content_nature_cn', 'nature_detail', 'nature_detail_cn',
        'panic_type', 'panic_type_cn', 'created_at', 'user_id',
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f'  已保存: {OUT_PATH}')

    # 顺带生成 event_broadcast 需要的时间可解析性检查
    print('\n【Step 4】时间列抽样检查 ...')
    sample = df[['created_at', 'user_id']].dropna().head(5)
    print(sample.to_string() if len(sample) else '  (created_at 均为空，需检查回填)')
    print('\n✅ doc_taxonomy_mapping.csv 构建完成')


if __name__ == '__main__':
    main()
