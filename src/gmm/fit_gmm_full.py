#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量 GMM 分层脚本（主分析版）

目标
1) 读取全量 temp*.csv 中的 panic 与 panic_prob
2) 将全样本划分为：非恐慌 / 低恐慌 / 中恐慌 / 高恐慌
3) 第一阶段：panic=0 作为“非恐慌”
4) 第二阶段：仅在 panic=1 子样本内，对 panic_prob 做 GMM 分层，得到低/中/高恐慌
5) 输出全量分层结果、阈值、摘要统计，供后续主题分析 / RDD / 事件研究 / 异质性分析使用

说明
- panic_prob 视为模型风险分数，而不是已校准真实概率
- GMM 只负责 panic=1 内部强度分层；panic=0 由二分类结果单独定义
- 主结果建议使用四分类：非恐慌、低恐慌、中恐慌、高恐慌
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==================== 配置 ====================
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
GRID_SIZE = 20001
OUTPUT_THRESHOLDS_PATH = os.path.join(DATA_DIR, 'full_gmm_thresholds.csv')
OUTPUT_STRATIFIED_PATH = os.path.join(DATA_DIR, 'full_gmm_stratified.csv')

# 统一四分类命名
LABEL_MAP = {
    0: '非恐慌',
    1: '低恐慌',
    2: '中恐慌',
    3: '高恐慌',
}

print('=' * 90)
print('全量 GMM 分层脚本（主分析版）')
print('=' * 90)

# ==================== 工具函数 ====================
def load_all_data():
    frames = []
    total_rows = 0
    found_rows = 0

    print('\n【Step 0】读取全量 temp*.csv ...')
    for fname in TEMP_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f'  跳过 {fname}（不存在）')
            continue

        file_frames = []
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
            total_rows += len(chunk)
            if PROB_COL not in chunk.columns or PRED_COL not in chunk.columns:
                continue

            prob = pd.to_numeric(chunk[PROB_COL], errors='coerce')
            pred = pd.to_numeric(chunk[PRED_COL], errors='coerce')
            mask = prob.notna() & pred.notna() & (prob >= 0) & (prob <= 1)
            if mask.sum() == 0:
                continue

            sub = chunk.loc[mask].copy()
            sub[PROB_COL] = prob.loc[mask].astype(float)
            sub[PRED_COL] = pred.loc[mask].astype(int)
            sub['_source_file'] = fname
            file_frames.append(sub)
            found_rows += len(sub)

        if file_frames:
            file_df = pd.concat(file_frames, ignore_index=True)
            frames.append(file_df)
            print(f'  {fname}: 读取 {len(file_df):,} 行有效样本')
        else:
            print(f'  {fname}: 未找到有效 panic/panic_prob')

    if not frames:
        raise ValueError('未读取到任何有效样本，请确认 temp*.csv 已完成标注输出')

    df = pd.concat(frames, ignore_index=True)
    print(f'\n  总扫描行数: {total_rows:,}')
    print(f'  有效样本行数: {len(df):,}')

    if len(df) > SAMPLE_TARGET:
        df = df.sample(n=SAMPLE_TARGET, random_state=42).copy()
        print(f'  随机采样: {SAMPLE_TARGET:,} 行用于全局探索与 GMM 拟合')
    else:
        print(f'  使用全部样本: {len(df):,} 行')

    return df.reset_index(drop=True)


def describe_distribution(probs, title='全量分布'):
    print(f'\n【Step 1】{title}')
    mean = float(np.mean(probs))
    median = float(np.median(probs))
    std = float(np.std(probs, ddof=1))
    skew = float(stats.skew(probs, bias=False))
    kurt = float(stats.kurtosis(probs, fisher=True, bias=False))
    jb_stat, jb_p = stats.jarque_bera(probs)

    hist, bin_edges = np.histogram(probs, bins=50, range=(0, 1))
    peak_idx = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
    peak_pos = ((bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2).tolist() if len(peak_idx) else []

    print(f'  n={len(probs):,}')
    print(f'  min={np.min(probs):.4f}, p25={np.percentile(probs, 25):.4f}, median={median:.4f}, p75={np.percentile(probs, 75):.4f}, max={np.max(probs):.4f}')
    print(f'  mean={mean:.4f}, std={std:.4f}')
    print(f'  skewness={skew:.4f}, kurtosis(excess)={kurt:.4f}')
    print(f'  Jarque-Bera: stat={jb_stat:.2f}, p={jb_p:.4g}')
    print(f'  方向判断: {"右偏" if skew > 0 else "左偏或对称"}')
    print(f'  直方图局部峰数（粗略）: {len(peak_idx)}')
    if peak_pos:
        print(f'  峰位置（粗略）: {[round(x, 3) for x in peak_pos[:10]]}')


def fit_gmm_models(probs):
    print('\n【Step 2】拟合 GMM(K=2~5) 并用 BIC/AIC 选型')
    X = probs.reshape(-1, 1)
    models = {}
    bic_list = []
    aic_list = []
    silhouette_list = []

    for k in K_RANGE:
        t0 = time.time()
        gmm = GaussianMixture(
            n_components=k,
            random_state=42,
            max_iter=500,
            n_init=10,
            covariance_type='full'
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        labels = gmm.predict(X)
        sil = np.nan
        if k > 1 and len(np.unique(labels)) > 1:
            try:
                from sklearn.metrics import silhouette_score
                sil = silhouette_score(X, labels)
            except Exception:
                sil = np.nan
        models[k] = gmm
        bic_list.append(bic)
        aic_list.append(aic)
        silhouette_list.append(sil)
        means = np.sort(gmm.means_.flatten())
        sil_str = f'{sil:.4f}' if np.isfinite(sil) else 'N/A'
        print(f'  K={k}: BIC={bic:,.0f} | AIC={aic:,.0f} | Sil={sil_str} | means={[f"{m:.3f}" for m in means]} | 耗时={time.time()-t0:.1f}s')

    best_k_bic = list(K_RANGE)[int(np.argmin(bic_list))]
    best_k_aic = list(K_RANGE)[int(np.argmin(aic_list))]
    valid_sil = np.array([s if np.isfinite(s) else -np.inf for s in silhouette_list])
    best_k_sil = list(K_RANGE)[int(np.argmax(valid_sil))]
    chosen_k = best_k_bic

    print(f'\n  >>> BIC 最优 K={best_k_bic}')
    print(f'  >>> AIC 最优 K={best_k_aic}')
    print(f'  >>> Silhouette 最优 K={best_k_sil}')
    if not (best_k_aic == best_k_bic == best_k_sil):
        print('  >>> 三指标不完全一致时，优先用 BIC，并结合样本量与理论可解释性')

    return models, chosen_k, best_k_bic, best_k_aic, best_k_sil, pd.DataFrame({'K': list(K_RANGE), 'BIC': bic_list, 'AIC': aic_list, 'Silhouette': silhouette_list})


def sort_components(gmm):
    means = gmm.means_.flatten()
    order = np.argsort(means)
    cov = gmm.covariances_.reshape(-1)
    return {
        'means': means[order],
        'stds': np.sqrt(cov)[order],
        'weights': gmm.weights_[order],
        'order': order,
    }


def posterior_boundary(gmm, left_mu, right_mu):
    x = np.linspace(0, 1, GRID_SIZE).reshape(-1, 1)
    resp = gmm.predict_proba(x)
    means = gmm.means_.flatten()
    idx_left = int(np.argmin(np.abs(means - left_mu)))
    idx_right = int(np.argmin(np.abs(means - right_mu)))

    diff = resp[:, idx_left] - resp[:, idx_right]
    lo, hi = min(left_mu, right_mu), max(left_mu, right_mu)
    mask = (x.flatten() >= lo) & (x.flatten() <= hi)
    xm = x.flatten()[mask]
    dm = diff[mask]

    if len(xm) < 2:
        return float((left_mu + right_mu) / 2)

    sign_changes = np.where(np.diff(np.sign(dm)) != 0)[0]
    if len(sign_changes) > 0:
        i = sign_changes[0]
        x1, x2 = xm[i], xm[i + 1]
        y1, y2 = dm[i], dm[i + 1]
        if y2 != y1:
            return float(x1 - y1 * (x2 - x1) / (y2 - y1))
        return float(x1)

    return float(xm[np.argmin(np.abs(dm))])


def evaluate_partition(probs, t1, t2):
    low = float(np.mean(probs < t1))
    mid = float(np.mean((probs >= t1) & (probs < t2)))
    high = float(np.mean(probs >= t2))
    return low, mid, high


def select_final_thresholds(panic_probs, gmm_boundaries, target_shares=(0.2, 0.6, 0.2), min_share=0.10):
    quantile_candidates = np.quantile(panic_probs, [0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75, 0.80])
    candidate_values = np.unique(np.round(np.concatenate([np.asarray(gmm_boundaries, dtype=float), quantile_candidates]), 6))
    if len(candidate_values) < 2:
        candidate_values = np.unique(np.round(quantile_candidates, 6))

    target = np.asarray(target_shares, dtype=float)
    scored = []
    for i, t1 in enumerate(candidate_values[:-1]):
        for t2 in candidate_values[i + 1:]:
            low, mid, high = evaluate_partition(panic_probs, t1, t2)
            penalty = 0.0
            if min(low, mid, high) < min_share:
                penalty += 100.0 * (min_share - min(low, mid, high) + 1e-6)
            if t2 - t1 < 0.03:
                penalty += 5.0
            shares = np.asarray([low, mid, high], dtype=float)
            loss = float(np.sum((shares - target) ** 2)) + penalty
            scored.append({
                't1': float(t1),
                't2': float(t2),
                'low_share': low,
                'mid_share': mid,
                'high_share': high,
                'score': loss,
                'from_gmm': int(t1 in np.asarray(gmm_boundaries, dtype=float)) + int(t2 in np.asarray(gmm_boundaries, dtype=float)),
            })

    if not scored:
        q1, q2 = np.quantile(panic_probs, [0.33, 0.67])
        low, mid, high = evaluate_partition(panic_probs, q1, q2)
        return float(q1), float(q2), pd.DataFrame([{'t1': q1, 't2': q2, 'low_share': low, 'mid_share': mid, 'high_share': high, 'score': np.nan, 'from_gmm': 0}])

    scored_df = pd.DataFrame(scored).sort_values(['score', 'from_gmm'], ascending=[True, False]).reset_index(drop=True)
    best = scored_df.iloc[0]
    return float(best['t1']), float(best['t2']), scored_df


def gmm_tier_for_panic1(prob, boundaries):
    tier = 1  # 1=低恐慌, 2=中恐慌, 3=高恐慌
    for b in boundaries:
        if prob >= b:
            tier += 1
    return min(tier, 3)


def label_full_sample(df, boundaries):
    df = df.copy()
    df['risk_group'] = np.where(df[PRED_COL] == 0, 0, np.nan)
    panic_mask = df[PRED_COL] == 1
    if panic_mask.any():
        df.loc[panic_mask, 'risk_group'] = df.loc[panic_mask, PROB_COL].apply(lambda x: gmm_tier_for_panic1(x, boundaries))
    df['risk_group'] = df['risk_group'].astype(int)
    df['risk_group_name'] = df['risk_group'].map(LABEL_MAP)
    return df


def summarize_groups(df):
    rows = []
    for g in sorted(df['risk_group'].unique()):
        sub = df[df['risk_group'] == g]
        rows.append({
            'group_id': int(g),
            'group_name': LABEL_MAP.get(int(g), f'Group_{g}'),
            'count': int(len(sub)),
            'share': float(len(sub) / len(df)),
            'prob_min': float(sub[PROB_COL].min()),
            'prob_mean': float(sub[PROB_COL].mean()),
            'prob_median': float(sub[PROB_COL].median()),
            'prob_max': float(sub[PROB_COL].max()),
        })
    return pd.DataFrame(rows)


def label_and_append_chunk(chunk, boundaries):
    chunk = chunk.copy()
    prob = pd.to_numeric(chunk[PROB_COL], errors='coerce')
    pred = pd.to_numeric(chunk[PRED_COL], errors='coerce')
    mask = prob.notna() & pred.notna() & (prob >= 0) & (prob <= 1)
    chunk = chunk.loc[mask].copy()
    if chunk.empty:
        return chunk

    chunk[PROB_COL] = prob.loc[mask].astype(float)
    chunk[PRED_COL] = pred.loc[mask].astype(int)
    chunk['risk_group'] = np.where(chunk[PRED_COL] == 0, 0, np.nan)
    panic_mask = chunk[PRED_COL] == 1
    if panic_mask.any():
        chunk.loc[panic_mask, 'risk_group'] = chunk.loc[panic_mask, PROB_COL].apply(lambda x: gmm_tier_for_panic1(x, boundaries))
    chunk['risk_group'] = chunk['risk_group'].astype(int)
    chunk['risk_group_name'] = chunk['risk_group'].map(LABEL_MAP)
    chunk['risk_group_4class'] = chunk['risk_group_name']
    return chunk


def save_full_stratified_data(boundaries):
    print('\n【Step 5】对全量数据逐块打标签并保存')
    total_rows = 0
    kept_rows = 0
    summary_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    wrote_header = False

    if os.path.exists(OUTPUT_STRATIFIED_PATH):
        os.remove(OUTPUT_STRATIFIED_PATH)

    for fname in TEMP_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
            total_rows += len(chunk)
            labeled = label_and_append_chunk(chunk, boundaries)
            if labeled.empty:
                continue
            kept_rows += len(labeled)
            counts = labeled['risk_group'].value_counts().to_dict()
            for k, v in counts.items():
                summary_counts[int(k)] = summary_counts.get(int(k), 0) + int(v)
            labeled.to_csv(OUTPUT_STRATIFIED_PATH, index=False, encoding='utf-8-sig', mode='a', header=not wrote_header)
            wrote_header = True

    summary_df = pd.DataFrame([
        {'group_id': gid, 'group_name': LABEL_MAP.get(gid, f'Group_{gid}'), 'count': cnt, 'share': cnt / kept_rows if kept_rows else 0.0}
        for gid, cnt in sorted(summary_counts.items())
    ])
    summary_df['group_name'] = summary_df['group_name'].astype(str)
    return total_rows, kept_rows, summary_df


# ==================== 主流程 ====================
def main():
    df = load_all_data()

    # 全局分布先看一眼
    describe_distribution(df[PROB_COL].to_numpy(dtype=np.float64), title='全样本 panic_prob 分布')

    # 第一阶段：非恐慌 / 恐慌
    nonpanic_n = int((df[PRED_COL] == 0).sum())
    panic_df = df[df[PRED_COL] == 1].copy()
    panic_probs = panic_df[PROB_COL].to_numpy(dtype=np.float64)

    print('\n【Step 2.5】两阶段定义')
    print(f'  非恐慌 (panic=0): {nonpanic_n:,} 行')
    print(f'  恐慌子样本 (panic=1): {len(panic_df):,} 行')

    if len(panic_df) == 0:
        raise ValueError('panic=1 子样本为空，无法进行二阶段分层')

    describe_distribution(panic_probs, title='panic=1 子样本 panic_prob 分布')

    # 只在 panic=1 子样本内拟合 GMM
    models, chosen_k, best_k_bic, best_k_aic, best_k_sil, model_ic = fit_gmm_models(panic_probs)
    gmm = models[chosen_k]
    comp = sort_components(gmm)
    means = comp['means']
    stds = comp['stds']
    weights = comp['weights']

    print('\n【Step 3】panic=1 子样本内按组件均值排序')
    print(f'  选定模型: K={chosen_k}')
    for i, (m, s, w) in enumerate(zip(means, stds, weights)):
        print(f'  Comp {i}: μ={m:.4f}, σ={s:.4f}, π={w:.3f}')

    print('\n【Step 4】用相邻组件后验概率相等点作为阈值')
    boundaries = []
    for i in range(len(means) - 1):
        b = posterior_boundary(gmm, means[i], means[i + 1])
        boundaries.append(b)
        print(f'  边界 {i}: {b:.4f}  ← μ={means[i]:.3f} vs μ={means[i+1]:.3f}')

    # 在 GMM 候选边界与分位数候选中，寻找更适合主题分析的最终阈值
    t1, t2, threshold_search_df = select_final_thresholds(panic_probs, boundaries)

    if not (0 <= t1 < t2 <= 1):
        q1, q2 = np.quantile(panic_probs, [0.33, 0.67])
        t1, t2 = float(q1), float(q2)

    low_share, mid_share, high_share = evaluate_partition(panic_probs, t1, t2)

    gmm_thresholds_df = pd.DataFrame({
        'chosen_k': [chosen_k],
        'best_k_bic': [best_k_bic],
        'best_k_aic': [best_k_aic],
        'best_k_sil': [best_k_sil],
        'panic1_boundaries': [str([round(float(x), 6) for x in boundaries])],
        'threshold_low_mid': [float(t1)],
        'threshold_mid_high': [float(t2)],
        'panic1_low_share': [float(low_share)],
        'panic1_mid_share': [float(mid_share)],
        'panic1_high_share': [float(high_share)],
        'means': [str([round(float(x), 6) for x in means])],
        'stds': [str([round(float(x), 6) for x in stds])],
        'weights': [str([round(float(x), 6) for x in weights])],
        'threshold_candidates_top5': [str(threshold_search_df.head(5).to_dict(orient='records'))],
    })
    gmm_thresholds_df = pd.concat([gmm_thresholds_df, model_ic], axis=1)
    gmm_thresholds_df.to_csv(OUTPUT_THRESHOLDS_PATH, index=False, encoding='utf-8-sig')

    print('\n【Step 4.5】阈值候选搜索结果（Top 5）')
    print(threshold_search_df.head(5).to_string(index=False))
    print(f'  选定阈值: low-mid={t1:.4f}, mid-high={t2:.4f}')
    print(f'  panic=1 三档占比: low={low_share:.4f}, mid={mid_share:.4f}, high={high_share:.4f}')

    # 全量样本四分类：逐块打标签，确保覆盖所有 temp*.csv 的有效行
    total_rows, kept_rows, group_summary = save_full_stratified_data([t1, t2])
    print('\n四分类汇总：')
    print(group_summary.to_string(index=False))
    print(f'\n  逐块处理总行数: {total_rows:,}')
    print(f'  实际保留有效行数: {kept_rows:,}')

    # 敏感性分析
    print('\n【Step 6】替代阈值敏感性分析')
    q_vals = np.quantile(panic_probs, [0.2, 0.4, 0.6, 0.8])
    print(f'  panic=1 GMM 边界: {[round(float(x), 4) for x in boundaries]}')
    print(f'  最终三档阈值: low-mid={t1:.4f}, mid-high={t2:.4f}')
    print(f'  分位数阈值(20/40/60/80%): {[round(float(x), 4) for x in q_vals]}')
    print('  理论阈值(参考): [0.50, 0.65, 0.88]')
    print(f'  中位数: {np.median(panic_probs):.4f}, p75: {np.percentile(panic_probs, 75):.4f}, p90: {np.percentile(panic_probs, 90):.4f}')
    print(f'  panic=1 三档占比: low={low_share:.4f}, mid={mid_share:.4f}, high={high_share:.4f}')

    # 论文可用表述
    print('\n【Step 7】论文可用方法表述')
    print('  本研究采用两阶段分层框架：首先由分类模型将样本划分为非恐慌与恐慌（panic=0/1），随后仅在恐慌子样本内基于连续风险分数 panic_prob 进行 GMM 分层，以识别低/中/高恐慌强度。')
    print(f'  GMM 模型在 K=2~5 中综合 BIC/AIC/Silhouette 选型，当前 panic=1 子样本的 BIC 最优 K 为 {best_k_bic}，AIC 最优 K 为 {best_k_aic}，Silhouette 最优 K 为 {best_k_sil}。')
    print('  GMM 组件按均值排序，并以相邻组件后验概率相等点作为候选阈值；若候选阈值不足以稳定形成三档，则回退至 panic=1 子样本内的分位数切分。')
    print('  最终全样本输出为四分类：非恐慌、低恐慌、中恐慌、高恐慌，可直接服务主题分析、RDD、事件研究和群体异质性分析。')

    print('\n【输出文件】')
    print(f'  阈值文件: {OUTPUT_THRESHOLDS_PATH}')
    print(f'  分层结果: {OUTPUT_STRATIFIED_PATH}')

    print('\n✅ 全量 GMM 分层完成')


if __name__ == '__main__':
    main()
