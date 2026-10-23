#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于人工复核标签的验证集抽样、模型校准与阈值比较脚本

用途：
1) 从人工复核后的标注数据中抽取一个小验证集（至少几百条）
2) 在验证集上比较原始 panic_prob、GMM 阈值、分位数阈值、理论阈值
3) 若存在独立训练集与验证集，则用训练集拟合校准器，在验证集上评估校准效果
4) 输出适合写入论文的方法有效性证据
"""

import os
import sys
import io
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
np.random.seed(42)

# ==================== 配置 ====================
INPUT_FILE = r"/root/autodl-tmp/covid_temp_check1.csv"
TEXT_COL = "text"
LABEL_COL = "check"          # 人工复核标签列
PROB_COL = "panic_prob"      # 模型输出概率列；若没有则脚本仍可完成抽样与阈值分析

TRAIN_SAVE_PATH = r"/root/autodl-tmp/temp_train1.csv"
VAL_SAVE_PATH = r"/root/autodl-tmp/temp_val1.csv"
VAL_SAMPLE_SAVE_PATH = r"/root/autodl-tmp/temp_val_sample_500.csv"
OUTPUT_SUMMARY_PATH = r"/root/autodl-tmp/threshold_validation_summary.csv"

VAL_SIZE = 0.2
MIN_VAL_SIZE = 500
RANDOM_STATE = 42
GMM_K_RANGE = range(2, 9)
THEORY_THRESHOLDS = [0.50, 0.65, 0.88]

# ==================== 工具函数 ====================
def ensure_binary_label(y: pd.Series) -> pd.Series:
    y = y.copy()
    if y.dtype == bool:
        return y.astype(int)
    if y.dtype.kind in "if":
        uniq = sorted(pd.unique(y.dropna()))
        if set(uniq).issubset({0, 1}):
            return y.astype(int)
    mapped = y.astype(str).str.strip().str.lower()
    yes_set = {"1", "true", "yes", "y", "是", "正类", "panic", "恐慌"}
    no_set = {"0", "false", "no", "n", "否", "负类", "non-panic", "非恐慌"}
    out = mapped.map(lambda v: 1 if v in yes_set else 0 if v in no_set else np.nan)
    if out.isna().any():
        raise ValueError("标签列无法自动映射为二分类，请先统一为 0/1 或明确的二元文本标签")
    return out.astype(int)


def calibration_metrics(y_true, p_pred, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=n_bins, strategy="quantile")
    ece = float(np.sum(np.abs(frac_pos - mean_pred) * np.histogram(p_pred, bins=n_bins, range=(0, 1))[0] / len(p_pred)))
    return frac_pos, mean_pred, ece


def best_threshold_by_f1(y_true, p_pred):
    thresholds = np.unique(np.round(np.linspace(0, 1, 1001), 4))
    best = {"threshold": 0.5, "f1": -1, "precision": None, "recall": None, "accuracy": None}
    for t in thresholds:
        pred = (p_pred >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        acc = accuracy_score(y_true, pred)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1), "precision": float(precision), "recall": float(recall), "accuracy": float(acc)}
    return best


def gmm_boundaries(probs, k=None):
    X = probs.reshape(-1, 1)
    models = {}
    bics = []
    aics = []
    for kk in GMM_K_RANGE:
        gmm = GaussianMixture(n_components=kk, random_state=RANDOM_STATE, max_iter=500, n_init=10, covariance_type='full')
        gmm.fit(X)
        models[kk] = gmm
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    best_k = list(GMM_K_RANGE)[int(np.argmin(bics))] if k is None else k
    gmm = models[best_k]
    means = gmm.means_.flatten()
    order = np.argsort(means)
    means = means[order]

    grid = np.linspace(0, 1, 20001).reshape(-1, 1)
    resp = gmm.predict_proba(grid)
    boundaries = []
    for i in range(len(means) - 1):
        left_mu, right_mu = means[i], means[i + 1]
        idx_left = int(np.argmin(np.abs(gmm.means_.flatten() - left_mu)))
        idx_right = int(np.argmin(np.abs(gmm.means_.flatten() - right_mu)))
        diff = resp[:, idx_left] - resp[:, idx_right]
        mask = (grid.flatten() >= min(left_mu, right_mu)) & (grid.flatten() <= max(left_mu, right_mu))
        xm = grid.flatten()[mask]
        dm = diff[mask]
        sign_changes = np.where(np.diff(np.sign(dm)) != 0)[0]
        if len(sign_changes) > 0:
            i0 = sign_changes[0]
            x1, x2 = xm[i0], xm[i0 + 1]
            y1, y2 = dm[i0], dm[i0 + 1]
            boundary = float(x1 - y1 * (x2 - x1) / (y2 - y1)) if y2 != y1 else float(x1)
        else:
            boundary = float(xm[np.argmin(np.abs(dm))])
        boundaries.append(boundary)
    return best_k, models[best_k], boundaries, bics, aics


def eval_threshold(y_true, p_pred, threshold, name):
    pred = (p_pred >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "method": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, p_pred)),
        "pr_auc": float(average_precision_score(y_true, p_pred)),
        "brier": float(brier_score_loss(y_true, p_pred)),
    }


# ==================== 主流程 ====================
def main():
    print("=" * 80)
    print("验证集校准与阈值比较脚本")
    print("=" * 80)

    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    required = {TEXT_COL, LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列: {missing}")

    keep_cols = [TEXT_COL, LABEL_COL] + ([PROB_COL] if PROB_COL in df.columns else [])
    df = df[keep_cols].dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df[LABEL_COL] = ensure_binary_label(df[LABEL_COL])

    print(f"数据总量: {len(df):,}")
    print(f"正类占比: {df[LABEL_COL].mean():.4f}")

    # 先按标签分层切分训练/验证
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL]
    )

    # 若验证集不足，额外从训练集再抽取到至少几百条
    if len(val_df) < MIN_VAL_SIZE:
        target_val_size = max(MIN_VAL_SIZE, int(len(df) * VAL_SIZE))
        val_df, train_df = train_test_split(
            df,
            test_size=max(1 - target_val_size / len(df), 0.01),
            random_state=RANDOM_STATE,
            stratify=df[LABEL_COL]
        )

    train_df.to_csv(TRAIN_SAVE_PATH, index=False, encoding='utf-8-sig')
    val_df.to_csv(VAL_SAVE_PATH, index=False, encoding='utf-8-sig')

    # 再从验证集中抽一个小标注集，便于人工复核或快速试验
    sample_n = min(max(MIN_VAL_SIZE, 500), len(val_df))
    if val_df[LABEL_COL].nunique() > 1:
        n_pos = max(1, int(round(sample_n * val_df[LABEL_COL].mean())))
        n_neg = max(1, sample_n - n_pos)
        pos_df = val_df[val_df[LABEL_COL] == 1].sample(n=min(n_pos, int((val_df[LABEL_COL] == 1).sum())), random_state=RANDOM_STATE)
        neg_df = val_df[val_df[LABEL_COL] == 0].sample(n=min(n_neg, int((val_df[LABEL_COL] == 0).sum())), random_state=RANDOM_STATE)
        val_sample = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
    else:
        val_sample = val_df.sample(n=sample_n, random_state=RANDOM_STATE)
    val_sample.to_csv(VAL_SAMPLE_SAVE_PATH, index=False, encoding='utf-8-sig')

    print(f"训练集: {len(train_df):,}")
    print(f"验证集: {len(val_df):,}")
    print(f"验证抽样集: {len(val_sample):,}")
    print(f"已保存: {TRAIN_SAVE_PATH}")
    print(f"已保存: {VAL_SAVE_PATH}")
    print(f"已保存: {VAL_SAMPLE_SAVE_PATH}")

    if PROB_COL not in df.columns:
        print("\n未检测到 panic_prob 列，脚本已完成抽样与划分；若要做校准和阈值比较，请补充模型输出列 panic_prob。")
        return

    # 仅使用有概率的样本做后续分析
    train_prob = train_df.dropna(subset=[PROB_COL]).copy()
    val_prob = val_df.dropna(subset=[PROB_COL]).copy()
    if len(train_prob) == 0 or len(val_prob) == 0:
        raise ValueError("训练集或验证集中没有可用的 panic_prob 数据")

    y_train = train_prob[LABEL_COL].to_numpy(dtype=int)
    p_train = pd.to_numeric(train_prob[PROB_COL], errors='coerce').clip(0, 1).to_numpy(dtype=float)
    y_val = val_prob[LABEL_COL].to_numpy(dtype=int)
    p_val = pd.to_numeric(val_prob[PROB_COL], errors='coerce').clip(0, 1).to_numpy(dtype=float)

    # ---------------- 校准 ----------------
    print("\n【Step 1】模型校准")
    cal_methods = {}

    lr = LogisticRegression(max_iter=1000)
    lr.fit(p_train.reshape(-1, 1), y_train)
    p_val_lr = lr.predict_proba(p_val.reshape(-1, 1))[:, 1]
    cal_methods["platt_logistic"] = p_val_lr

    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(p_train, y_train)
    p_val_iso = ir.transform(p_val)
    cal_methods["isotonic"] = p_val_iso

    # 原始分数
    cal_methods["raw"] = p_val

    for name, p in cal_methods.items():
        frac_pos, mean_pred, ece = calibration_metrics(y_val, p)
        print(f"  {name}: Brier={brier_score_loss(y_val, p):.4f}, ECE={ece:.4f}, AUC={roc_auc_score(y_val, p):.4f}")

    # 选一个默认更稳的校准结果：isotonic 优先，否则 logistic
    calibrated_name = "isotonic" if len(np.unique(p_train)) > 1 else "platt_logistic"
    p_val_cal = cal_methods[calibrated_name]
    print(f"  默认采用校准结果: {calibrated_name}")

    # ---------------- 阈值比较 ----------------
    print("\n【Step 2】阈值比较")

    # 2.1 在原始概率上拟合 GMM
    gmm_k, gmm_model, gmm_bounds, bics, aics = gmm_boundaries(p_train, k=None)
    print(f"  GMM 选型: K={gmm_k}")
    print(f"  GMM 阈值: {[round(x, 4) for x in gmm_bounds]}")

    # 取中间边界作为主阈值候选（若有多个阈值）
    gmm_main_threshold = float(gmm_bounds[len(gmm_bounds)//2]) if len(gmm_bounds) > 0 else 0.5

    quantile_threshold = float(np.quantile(p_train, 0.5))
    theory_threshold = float(THEORY_THRESHOLDS[1])

    best_raw = best_threshold_by_f1(y_val, p_val)
    best_cal = best_threshold_by_f1(y_val, p_val_cal)

    results = []
    results.append(eval_threshold(y_val, p_val, gmm_main_threshold, "gmm_raw"))
    results.append(eval_threshold(y_val, p_val, quantile_threshold, "quantile_raw_median"))
    results.append(eval_threshold(y_val, p_val, theory_threshold, "theory_065"))
    results.append(eval_threshold(y_val, p_val_cal, gmm_main_threshold, "gmm_on_calibrated_prob"))
    results.append(eval_threshold(y_val, p_val_cal, quantile_threshold, "quantile_on_calibrated_prob"))
    results.append(eval_threshold(y_val, p_val_cal, theory_threshold, "theory_on_calibrated_prob"))
    results.append(eval_threshold(y_val, p_val, best_raw["threshold"], "best_f1_raw"))
    results.append(eval_threshold(y_val, p_val_cal, best_cal["threshold"], f"best_f1_{calibrated_name}"))

    res_df = pd.DataFrame(results).sort_values(["f1", "roc_auc"], ascending=False)
    res_df.to_csv(OUTPUT_SUMMARY_PATH, index=False, encoding='utf-8-sig')

    print("\n阈值比较结果（按 F1 排序）:")
    print(res_df[["method", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]].to_string(index=False))
    print(f"\n已保存汇总: {OUTPUT_SUMMARY_PATH}")

    # ---------------- 方法有效性证据 ----------------
    print("\n【Step 3】可直接写入论文的方法证据")
    print(f"  1) 验证集规模: {len(val_df):,}（其中有概率值样本 {len(val_prob):,}）")
    print(f"  2) 人工标签正类占比: {y_val.mean():.4f}")
    print(f"  3) 原始概率 AUC: {roc_auc_score(y_val, p_val):.4f}, Brier: {brier_score_loss(y_val, p_val):.4f}")
    print(f"  4) 校准后 AUC({calibrated_name}): {roc_auc_score(y_val, p_val_cal):.4f}, Brier: {brier_score_loss(y_val, p_val_cal):.4f}")
    print(f"  5) GMM 主阈值: {gmm_main_threshold:.4f}")
    print(f"  6) 验证集最优 F1 阈值（原始概率）: {best_raw['threshold']:.4f}，F1={best_raw['f1']:.4f}")
    print(f"  7) 验证集最优 F1 阈值（校准后概率）: {best_cal['threshold']:.4f}，F1={best_cal['f1']:.4f}")
    print("  8) 若 GMM / 分位数 / 理论阈值在验证集上的结果接近，则可作为方法稳健性证据")

    print("\n【论文可用表述】")
    print("我们首先从人工复核后的数据中按标签分层抽取独立验证集，并保留训练/验证拆分互不重叠。")
    print("在训练集上对模型输出分数进行校准，并在验证集上比较 GMM 阈值、分位数阈值与理论阈值的分类表现。")
    print("若不同阈值下的主要结论一致，则说明阈值设定对主结果不敏感，从而提高方法的可信度。")

    print("\n✅ 全部完成")


if __name__ == '__main__':
    main()
