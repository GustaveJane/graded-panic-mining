# -*- coding: utf-8 -*-
"""构念效度检验分析：panic_prob 到底是"恐慌强度"还是"分类器置信度"。

输入：
  construct_validity_truth.csv        真值（panic_prob / gmm_tier / 是否注意力检查）
  construct_validity_annotation*.csv  各标注员填好的表（可多个）

输出：控制台报告 + construct_validity_report.json

用法：
  python construct_validity_analysis.py truth.csv 甲.csv 乙.csv 丙.csv
"""
import sys
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats

TRUTH = sys.argv[1] if len(sys.argv) > 1 else "construct_validity_truth.csv"
ANN_FILES = sys.argv[2:] if len(sys.argv) > 2 else sorted(
    [f for f in glob.glob("construct_validity_annotation*.csv")])

truth = pd.read_csv(TRUTH, encoding="utf-8-sig")
truth["item_id"] = truth["item_id"].astype(str)

raters = {}
for f in ANN_FILES:
    a = pd.read_csv(f, encoding="utf-8-sig")
    a["item_id"] = a["item_id"].astype(str)
    col = [c for c in a.columns if "强度" in c][0]
    name = f.replace("construct_validity_annotation", "").replace(".csv", "") or "rater"
    a = a[["item_id", col]].rename(columns={col: name})
    a[name] = pd.to_numeric(a[name], errors="coerce")
    raters[name] = a

if not raters:
    print("未找到标注文件。"); sys.exit(1)

m = truth.copy()
for name, a in raters.items():
    m = m.merge(a, on="item_id", how="left")
rate_cols = list(raters.keys())
print(f"标注员：{rate_cols}；条目：{len(m)}")

report = {"n_items": int(len(m)), "raters": rate_cols}


def krippendorff_alpha_ordinal(data):
    """有序变量的 Krippendorff alpha（仅用完全标注的条目）。"""
    X = data.dropna().values
    if X.shape[0] < 3 or X.shape[1] < 2:
        return None
    n, m_ = X.shape
    vals = np.unique(X)
    # 观测不一致
    Do = 0.0
    for row in X:
        for i in range(m_):
            for j in range(i + 1, m_):
                Do += (row[i] - row[j]) ** 2
    Do /= (n * m_ * (m_ - 1) / 2)
    # 期望不一致
    counts = np.array([(X == v).sum() for v in vals], dtype=float)
    n_tot = counts.sum()
    De = 0.0
    for i, vi in enumerate(vals):
        for j, vj in enumerate(vals):
            if i == j:
                continue
            De += counts[i] * counts[j] * (vi - vj) ** 2
    De /= (n_tot * (n_tot - 1))
    return 1 - Do / De if De > 0 else None


# ---------- 1. 评分者间一致性 ----------
complete = m[rate_cols].dropna()
if len(complete) >= 3 and len(rate_cols) >= 2:
    alpha = krippendorff_alpha_ordinal(m[rate_cols])
    report["krippendorff_alpha_ordinal"] = round(alpha, 4) if alpha else None
    print(f"\n[1] Krippendorff alpha (ordinal) = {alpha:.4f}  (n={len(complete)})"
          if alpha else "\n[1] 标注不全，跳过一致性")
    if len(rate_cols) >= 3:
        try:
            import itertools
            cats = [complete[c].astype(int).values for c in rate_cols]
            fk = stats.friedmanchisquare(*cats)
            report["friedman_p"] = round(float(fk.pvalue), 5)
            print(f"    Friedman 检验（评分者系统偏差）p = {fk.pvalue:.4f}")
        except Exception as e:
            print("    Friedman 跳过:", e)

# ---------- 2. 人工强度均分 vs panic_prob（主指标）----------
m["human_mean"] = m[rate_cols].mean(axis=1)
mm = m.dropna(subset=["human_mean", "panic_prob"])
rho, p_rho = stats.spearmanr(mm["panic_prob"], mm["human_mean"])
report["spearman_rho"] = round(float(rho), 4)
report["spearman_p"] = float(p_rho)
report["n_for_spearman"] = int(len(mm))
print(f"\n[2] Spearman(panic_prob, 人工强度均分) = {rho:.4f}, p = {p_rho:.3g}, n = {len(mm)}")

# ---------- 3. 各层人工强度差异 ----------
groups = [g["human_mean"].dropna().values for _, g in mm.groupby("gmm_tier")]
groups = [g for g in groups if len(g) >= 3]
if len(groups) >= 2:
    H, p_kw = stats.kruskal(*groups)
    report["kruskal_H"] = round(float(H), 4)
    report["kruskal_p"] = float(p_kw)
    print(f"\n[3] Kruskal-Wallis（各层人工强度）H = {H:.4f}, p = {p_kw:.3g}")
    print("    各层人工强度均值：")
    tbl = mm.groupby(["gmm_tier", "layer_name"])["human_mean"].agg(
        ["size", "mean", "std"]).round(3)
    print(tbl.to_string())
    report["tier_means"] = {str(k): float(v) for k, v in
                            mm.groupby("gmm_tier")["human_mean"].mean().items()}
    # 单调性
    tm = mm.groupby("gmm_tier")["human_mean"].mean().values
    report["monotonic_increasing"] = bool(np.all(np.diff(tm) > 0))
    print(f"    均值随层单调递增: {report['monotonic_increasing']}")

# ---------- 4. 注意力检查（评分者内一致性）----------
# 机制：标注包中若干组文本完全相同的条目对，其中一条标记为 is_attention_check=1。
# 认真标注的标注员应对同一文本给出相同强度评分；若同一条目对打分差异 >=2，视为未通过。
att_ids = set(truth.loc[truth["is_attention_check"] == 1, "item_id"])
ann_full = pd.concat([a for a in raters.values()], axis=0)  # 合并所有标注员长表
# 用 ann 源文本建 item_id->text 映射（从 annotation*.csv 的 text 列取）
id2text = {}
for f in ANN_FILES:
    t = pd.read_csv(f, encoding="utf-8-sig", dtype=str).fillna("")
    for _, row in t.iterrows():
        id2text[str(row["item_id"])] = row["text"]
# 每个注意力条目找同文本的普通条目
att_results = {}
for att_id in att_ids:
    att_text = id2text.get(att_id, "")
    if not att_text:
        continue
    peer_ids = [i for i, t in id2text.items() if t == att_text and i != att_id]
    for rater in rate_cols:
        s_att = m.loc[m["item_id"] == att_id, rater]
        for peer in peer_ids:
            s_peer = m.loc[m["item_id"] == peer, rater]
            if len(s_att) and len(s_peer):
                d = abs(float(s_att.iloc[0]) - float(s_peer.iloc[0]))
                att_results.setdefault(rater, []).append(d)
for rater, ds in att_results.items():
    ds = [d for d in ds if d == d]  # 去 NaN
    if ds:
        n_ok = sum(1 for d in ds if d <= 1)
        report.setdefault("attention_check", {})[rater] = {
            "pairs": len(ds),
            "pass_pairs": n_ok,
            "max_abs_diff": max(ds),
        }
        print(f"\n[4] {rater}: 注意力对 {len(ds)} 组，评分内差异<=1 通过 {n_ok} 组，最大差异 {max(ds)}")

# ---------- 5. 结论判定 ----------
print("\n" + "=" * 60)
if rho >= 0.6 and report.get("monotonic_increasing"):
    verdict = "支持强度分层：panic_prob 与人工强度高度相关且各层单调递增"
elif rho >= 0.4:
    verdict = "部分支持：相关性中等，建议用'部分支持强度分层'的谨慎表述"
else:
    verdict = "不支持强度分层：应改称'置信度分层'，弱化 RQ3 机制论述"
report["verdict"] = verdict
print("结论判定：", verdict)
print("=" * 60)

with open("construct_validity_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\nsaved: construct_validity_report.json")
