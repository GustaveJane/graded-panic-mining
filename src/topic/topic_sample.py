#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topic_sample.py

从 full_gmm_stratified.csv 中分层抽样，生成主题分析用语料。

改进点：
1) 逐块流式读取，不把全量 36GB 数据一次性载入内存；
2) 非恐慌按“每块固定上限 + 最终再截断”的方式抽样，近似全局随机且更省内存；
3) 低/中/高恐慌尽量保留完整文本，必要时对中恐慌进行上限控制；
4) 清洗过短、乱码和纯噪声文本；
5) 输出可直接用于 BERTopic 的语料文件。
"""

import os
import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(os.environ.get('TOPIC_BASE_DIR', Path(__file__).resolve().parent))
INPUT_PATH = Path(os.environ.get('TOPIC_INPUT_PATH', BASE_DIR / 'full_gmm_stratified.csv'))
OUTPUT_PATH = Path(os.environ.get('TOPIC_OUTPUT_PATH', BASE_DIR / 'topic_corpus.csv'))

NONPANIC_SAMPLE = 30000
NONPANIC_PER_CHUNK = 120
PANIC_MID_CAP = 30000
MIN_TEXT_LEN = 8
MAX_DUP_PER_GROUP = 3
RANDOM_STATE = 42
CHUNK_SIZE = 200000

CLEAN_RE_LIST = [
    (r'http[s]?://\S+', ''),
    (r'www\.\S+', ''),
    (r'\s+', ' '),
]


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.strip()
    for pat, rep in CLEAN_RE_LIST:
        text = re.sub(pat, rep, text)
    text = re.sub(r'[\u200b\ufeff]', '', text)
    return text.strip()


def is_useful_text(text):
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < MIN_TEXT_LEN:
        return False
    # 过滤几乎没有有效字词的噪声
    if len(re.sub(r'[\W_]+', '', t)) < 4:
        return False
    return True


def dedup_by_content(df, max_per_group=MAX_DUP_PER_GROUP):
    """同一 risk_group 内，完全相同文本最多保留 max_per_group 条"""
    if len(df) == 0:
        return df
    return df.groupby(['risk_group', 'content'], group_keys=False).head(max_per_group).reset_index(drop=True)


def load_selected_columns(path):
    usecols = ['content', 'risk_group', 'risk_group_name', 'panic_prob', '_source_file']
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False, usecols=lambda c: c in usecols):
        yield chunk.copy()


def sample_group(df, group_id, n=None, random_state=RANDOM_STATE):
    sub = df[df['risk_group'] == group_id].copy()
    if n is None or len(sub) <= n:
        return sub
    return sub.sample(n=n, random_state=random_state)


def main():
    print('=' * 70)
    print('topic_sample.py — 分层抽样生成主题分析语料')
    print('=' * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f'未找到 {INPUT_PATH}，请先运行全量GMM分层脚本')

    print('\n【Step 1】逐块读取分层结果并做流式抽样 ...')
    nonpanic_parts = []
    low_parts = []
    mid_parts = []
    high_parts = []
    total_rows = 0

    for i, chunk in enumerate(load_selected_columns(INPUT_PATH), start=1):
        total_rows += len(chunk)
        chunk['content'] = chunk['content'].apply(clean_text)
        chunk = chunk[chunk['content'].apply(is_useful_text)].copy()
        if len(chunk) == 0:
            continue

        # 非恐慌：每块保留少量，最后再统一截断
        nonpanic_chunk = chunk[chunk['risk_group'] == 0]
        if len(nonpanic_chunk) > 0:
            take_n = min(NONPANIC_PER_CHUNK, len(nonpanic_chunk))
            nonpanic_parts.append(nonpanic_chunk.sample(n=take_n, random_state=RANDOM_STATE + i))

        # 低恐慌：尽量全取
        low_chunk = chunk[chunk['risk_group'] == 1]
        if len(low_chunk) > 0:
            low_parts.append(low_chunk)

        # 中恐慌：全取，但总量过大时在最后统一截断
        mid_chunk = chunk[chunk['risk_group'] == 2]
        if len(mid_chunk) > 0:
            mid_parts.append(mid_chunk)

        # 高恐慌：尽量全取
        high_chunk = chunk[chunk['risk_group'] == 3]
        if len(high_chunk) > 0:
            high_parts.append(high_chunk)

        if i % 25 == 0:
            print(f'  已处理 {i} 个 chunk，累计读取 {total_rows:,} 行')

    print(f'  累计读取总行数: {total_rows:,}')

    print('\n【Step 2】汇总各层样本 ...')
    nonpanic = pd.concat(nonpanic_parts, ignore_index=True) if nonpanic_parts else pd.DataFrame()
    low = pd.concat(low_parts, ignore_index=True) if low_parts else pd.DataFrame()
    mid = pd.concat(mid_parts, ignore_index=True) if mid_parts else pd.DataFrame()
    high = pd.concat(high_parts, ignore_index=True) if high_parts else pd.DataFrame()

    if len(nonpanic) > NONPANIC_SAMPLE:
        nonpanic = nonpanic.sample(n=NONPANIC_SAMPLE, random_state=RANDOM_STATE)
    print(f'  非恐慌: {len(nonpanic):,}')
    print(f'  低恐慌: {len(low):,}')
    print(f'  中恐慌: {len(mid):,}')
    print(f'  高恐慌: {len(high):,}')

    if len(mid) > PANIC_MID_CAP:
        mid = mid.sample(n=PANIC_MID_CAP, random_state=RANDOM_STATE)
        print(f'  中恐慌已截断至: {len(mid):,}')

    corpus_parts = [x for x in [nonpanic, low, mid, high] if len(x) > 0]
    if not corpus_parts:
        raise ValueError('未抽到任何可用文本，请检查输入数据或筛选规则')

    corpus = pd.concat(corpus_parts, ignore_index=True)

    print(f'\n【Step 3】跨组去重（同组重复文本最多保留 {MAX_DUP_PER_GROUP} 条） ...')
    before = len(corpus)
    corpus = dedup_by_content(corpus, MAX_DUP_PER_GROUP)
    print(f'  去重前 {before:,} → 去重后 {len(corpus):,}')

    corpus['doc_id'] = range(len(corpus))
    corpus = corpus[['doc_id', 'content', 'risk_group', 'risk_group_name', 'panic_prob', '_source_file']]

    print(f'\n【Step 4】保存语料到: {OUTPUT_PATH}')
    corpus.to_csv(str(OUTPUT_PATH), index=False, encoding='utf-8-sig')
    print(f'  语料总行数: {len(corpus):,}')

    print('\n各层最终占比:')
    for g in sorted(corpus['risk_group'].unique()):
        n = len(corpus[corpus['risk_group'] == g])
        name = corpus.loc[corpus['risk_group'] == g, 'risk_group_name'].iloc[0]
        print(f'  risk_group={g} ({name}): {n:,} ({n / len(corpus) * 100:.1f}%)')

    print('✅ topic_sample.py 完成')


if __name__ == '__main__':
    main()
