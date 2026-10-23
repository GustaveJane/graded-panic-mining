#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LCA1.py

用途
- 作为展示与解释脚本，围绕 panic=1 子样本内的 GMM 分层结果制作理论映射与诊断输出
- 不负责主分层标签生成
- 主要用于论文中的方法示意、理论对照和图表展示

说明
- 这里保留“LCA”命名仅作为历史兼容；实际方法主体仍然是 GMM 风险分层
- 重点是“解释层”而不是“主分析层”
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = r'/root/autodl-tmp/'
TEMP_FILES = [
    'temp19.csv', 'temp1.csv', 'temp2.csv', 'temp3.csv', 'temp4.csv',
    'temp5.csv', 'temp6.csv', 'temp7.csv', 'temp8.csv', 'temp9.csv',
    'temp10.csv', 'temp11.csv', 'temp12.csv'
]
PROB_COL = 'panic_prob'
PRED_COL = 'panic'
CHUNK_SIZE = 200000
SAMPLE_TARGET = 500000
K_RANGE = range(2, 6)
OUTPUT_DIR = os.path.join(DATA_DIR, 'lca1_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAMPLE_TARGET = 100000
MIN_CLASS_SHARE = 0.10

THEORY_FRAMEWORK = {
    'lazarus': {
        'name': 'Lazarus Cognitive-Appraisal Theory',
        'stages': ['non/benign', 'mild stress', 'appraised threat', 'harm/loss']
    },
    'dsm5': {
        'name': 'DSM-5 Inspired Severity Order',
        'stages': ['subclinical', 'mild', 'moderate', 'severe']
    },
    'barrett': {
        'name': 'Barrett Emotional Granularity',
        'stages': ['low granularity', 'mid granularity', 'high granularity']
    }
}

print('=' * 90)
print('LCA1.py - GMM 展示与理论映射')
print('=' * 90)


def load_panic1_probs():
    vals = []
    for fname in TEMP_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
            if PROB_COL not in chunk.columns or PRED_COL not in chunk.columns:
                continue
            prob = pd.to_numeric(chunk[PROB_COL], errors='coerce')
            pred = pd.to_numeric(chunk[PRED_COL], errors='coerce')
            mask = prob.notna() & pred.notna() & (pred == 1) & (prob >= 0) & (prob <= 1)
            if mask.sum() > 0:
                vals.extend(prob.loc[mask].astype(float).tolist())
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        raise ValueError('未找到 panic=1 样本')
    if len(vals) > SAMPLE_TARGET:
        vals = np.random.choice(vals, size=SAMPLE_TARGET, replace=False)
    return np.sort(vals)


def describe_distribution(probs):
    print('\n【Step 1】panic=1 子样本分布摘要')
    print(f'  n={len(probs):,}, min={probs.min():.4f}, p25={np.percentile(probs, 25):.4f}, median={np.median(probs):.4f}, p75={np.percentile(probs, 75):.4f}, max={probs.max():.4f}')
    print(f'  mean={probs.mean():.4f}, std={probs.std(ddof=1):.4f}, skew={stats.skew(probs, bias=False):.4f}')


def fit_gmm(probs):
    print('\n【Step 2】GMM 模型选择')
    X = probs.reshape(-1, 1)
    rows = []
    models = {}
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, random_state=42, max_iter=500, n_init=10, covariance_type='full')
        gmm.fit(X)
        labels = gmm.predict(X)
        sil = np.nan
        if k > 1 and len(np.unique(labels)) > 1:
            try:
                from sklearn.metrics import silhouette_score
                sil = silhouette_score(X, labels)
            except Exception:
                sil = np.nan
        rows.append({'K': k, 'BIC': gmm.bic(X), 'AIC': gmm.aic(X), 'Silhouette': sil, 'min_share': float(np.min(np.bincount(labels) / len(labels)))})
        models[k] = gmm
        sil_str = f'{sil:.4f}' if np.isfinite(sil) else 'N/A'
        print(f'  K={k}: BIC={rows[-1]["BIC"]:,.0f} | AIC={rows[-1]["AIC"]:,.0f} | Sil={sil_str}')
    ic = pd.DataFrame(rows)
    valid = ic[ic['min_share'] >= MIN_CLASS_SHARE].copy()
    if len(valid) > 0:
        best_k_bic = int(valid.sort_values('BIC').iloc[0]['K'])
        best_k_aic = int(valid.sort_values('AIC').iloc[0]['K'])
        valid_sil = valid['Silhouette'].fillna(-np.inf).to_numpy()
        best_k_sil = int(valid.iloc[int(np.argmax(valid_sil))]['K'])
    else:
        best_k_bic = int(ic.sort_values('BIC').iloc[0]['K'])
        best_k_aic = int(ic.sort_values('AIC').iloc[0]['K'])
        valid_sil = ic['Silhouette'].fillna(-np.inf).to_numpy()
        best_k_sil = int(ic.iloc[int(np.argmax(valid_sil))]['K'])
    print(f'  >>> BIC 最优 K={best_k_bic}')
    print(f'  >>> AIC 最优 K={best_k_aic}')
    print(f'  >>> Silhouette 最优 K={best_k_sil}')
    if len(valid) == 0:
        print(f'  >>> 未找到满足最小类占比 {MIN_CLASS_SHARE:.0%} 的模型，回退到纯统计最优')
    return models, ic, best_k_bic, best_k_aic, best_k_sil


def posterior_boundary(gmm, left_mu, right_mu):
    grid = np.linspace(0, 1, 20001).reshape(-1, 1)
    resp = gmm.predict_proba(grid)
    means = gmm.means_.flatten()
    li = int(np.argmin(np.abs(means - left_mu)))
    ri = int(np.argmin(np.abs(means - right_mu)))
    diff = resp[:, li] - resp[:, ri]
    lo, hi = min(left_mu, right_mu), max(left_mu, right_mu)
    mask = (grid.flatten() >= lo) & (grid.flatten() <= hi)
    x = grid.flatten()[mask]
    d = diff[mask]
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    if len(idx) > 0:
        i = idx[0]
        x1, x2 = x[i], x[i + 1]
        y1, y2 = d[i], d[i + 1]
        return float(x1 - y1 * (x2 - x1) / (y2 - y1)) if y2 != y1 else float(x1)
    return float(x[np.argmin(np.abs(d))])


def main():
    probs = load_panic1_probs()
    describe_distribution(probs)
    models, ic, best_k_bic, best_k_aic, best_k_sil = fit_gmm(probs)
    gmm = models[best_k_bic]
    means = np.sort(gmm.means_.flatten())

    print('\n【Step 3】组件均值与边界')
    print('  均值: ' + ', '.join([f'{m:.4f}' for m in means]))
    bounds = []
    for i in range(len(means) - 1):
        b = posterior_boundary(gmm, means[i], means[i + 1])
        bounds.append(b)
        print(f'  边界{i}: {b:.4f}')

    # 理论映射：只做解释，不强行一一对应
    mapping_rows = []
    for name, fw in THEORY_FRAMEWORK.items():
        mapping_rows.append({
            'framework': fw['name'],
            'stages': ' | '.join(fw['stages']),
            'notes': '用于论文解释层，不作为主分层依据'
        })
    pd.DataFrame(mapping_rows).to_csv(os.path.join(OUTPUT_DIR, 'theory_mapping.csv'), index=False, encoding='utf-8-sig')

    # 诊断摘要
    diag = pd.DataFrame([
        {'item': 'best_k_bic', 'value': best_k_bic},
        {'item': 'best_k_aic', 'value': best_k_aic},
        {'item': 'best_k_sil', 'value': best_k_sil},
        {'item': 'gmm_bounds', 'value': str([round(float(x), 6) for x in bounds])},
        {'item': 'means', 'value': str([round(float(x), 6) for x in means])},
        {'item': 'bic', 'value': float(ic.sort_values('BIC').iloc[0]['BIC'])},
        {'item': 'aic', 'value': float(ic.sort_values('AIC').iloc[0]['AIC'])},
        {'item': 'silhouette', 'value': float(ic.sort_values('BIC').iloc[0]['Silhouette']) if np.isfinite(ic.sort_values('BIC').iloc[0]['Silhouette']) else np.nan},
    ])
    diag.to_csv(os.path.join(OUTPUT_DIR, 'gmm_diagnostics_summary.csv'), index=False, encoding='utf-8-sig')

    print('\n【Step 4】理论映射输出')
    for _, row in pd.DataFrame(mapping_rows).iterrows():
        print(f"  - {row['framework']}: {row['stages']}")

    print('\n【Step 5】使用建议')
    print('  1) 主分析请使用四分类结果：非恐慌 / 低恐慌 / 中恐慌 / 高恐慌')
    print('  2) 本脚本仅用于展示 GMM 结构与理论叙述，不替代主分层脚本')
    print('  3) 若要写论文，可将理论映射放在方法补充或附录')

    print(f'\n输出目录: {OUTPUT_DIR}')
    print('✅ LCA1.py 完成')


if __name__ == '__main__':
    main()
