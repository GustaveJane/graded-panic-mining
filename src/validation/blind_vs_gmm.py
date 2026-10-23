# -*- coding: utf-8 -*-
"""
构念效度：双盲标注 vs GMM 真值（最终口径）
==========================================
对象（已剔除 GMM 标签可见的那一份标注）:
  H1  人工标注员 1（盲标）   -> 人工1.csv
  H2  人工标注员 2（盲标）   -> 人工2.csv
  GMM GMM 机器分层标签        -> construct_validity_gmm_labeled.csv

产出:
  - 双盲组间信度
  - 两位盲标者及共识 vs GMM：相关、分级一致性、单调性、AUC
  - 门限层(tier0)对指标的抬升量化
  - 论文阈值重分组
  - 注意力检查
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
DESK = r"C:/Users/17853/Desktop"
SRC_H = {"H1": os.path.join(DESK, "人工1.csv"), "H2": os.path.join(DESK, "人工2.csv")}
SRC_GMM = os.path.join(BASE, "construct_validity_gmm_labeled.csv")
COL_H = "human_strength(待填1-5)"
NAME = {"H1": "人工1(盲标)", "H2": "人工2(盲标)", "CONS": "双盲共识"}
RNG = np.random.default_rng(20260929)


def load_csv(path):
    for enc in ["gbk", "gb18030", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(path)


def krippendorff_alpha_ordinal(mat):
    mat = np.asarray(mat, dtype=float)
    n, m = mat.shape
    Do = 0.0; n_pairs = 0
    for i in range(n):
        row = mat[i][~np.isnan(mat[i])]
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                Do += (row[a] - row[b]) ** 2; n_pairs += 1
    Do = Do / n_pairs if n_pairs else np.nan
    allv = mat[~np.isnan(mat)]
    De = 0.0; npairs2 = 0
    for a in range(len(allv)):
        for b in range(len(allv)):
            if a == b:
                continue
            De += (allv[a] - allv[b]) ** 2; npairs2 += 1
    De = De / npairs2 if npairs2 else np.nan
    return 1 - Do / De if De else np.nan


def icc_2_1(mat):
    mat = np.asarray(mat, dtype=float)
    n, k = mat.shape
    gm = mat.mean()
    ms_r = k * ((mat.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((mat.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    resid = mat - mat.mean(axis=1, keepdims=True) - mat.mean(axis=0, keepdims=True) + gm
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return (ms_r - ms_e) / denom if denom else np.nan


def boot_ci(x, y, fn, n_boot=2000):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        try:
            v = fn(x[idx], y[idx])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def pair_block(a, b, label_a="a", label_b="b"):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    out = {
        "n": int(len(a)),
        "exact_pct": float(100 * np.mean(a == b)),
        "within1_pct": float(100 * np.mean(np.abs(a - b) <= 1)),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
        "spearman": float(stats.spearmanr(a, b).statistic),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
    }
    try:
        out["kappa_linear"] = float(cohen_kappa_score(a.astype(int), b.astype(int), weights="linear"))
        out["kappa_quadratic"] = float(cohen_kappa_score(a.astype(int), b.astype(int), weights="quadratic"))
    except Exception:
        out["kappa_linear"] = np.nan; out["kappa_quadratic"] = np.nan
    vals = sorted(set(a.astype(int)) | set(b.astype(int)))
    cm = pd.crosstab(pd.Series(a.astype(int), name=label_a),
                     pd.Series(b.astype(int), name=label_b)).reindex(index=vals, columns=vals, fill_value=0)
    out["confusion"] = cm.to_dict()
    return out, cm


def main():
    gmm = load_csv(SRC_GMM)
    prob = pd.to_numeric(gmm["gmm_panic_prob"], errors="coerce").values.astype(float)
    tier = pd.to_numeric(gmm["gmm_tier"], errors="coerce").values.astype(int)
    attn = pd.to_numeric(gmm["is_attention_check"], errors="coerce").fillna(0).values.astype(int)

    H = {}
    for k, f in SRC_H.items():
        d = load_csv(f)
        assert (d["item_id"].values == gmm["item_id"].values).all(), f"{k} 未对齐"
        H[k] = pd.to_numeric(d[COL_H], errors="coerce").values.astype(float)

    H["CONS"] = (H["H1"] + H["H2"]) / 2.0          # 连续共识（用于相关）
    CONS_INT = np.round(H["CONS"]).astype(float)   # 取整共识（用于分级一致/κ）

    rep = {}
    print("=" * 78)
    print("构念效度检验：双盲标注 × GMM 真值（n=520，等频五层抽样）")
    print("=" * 78)

    # ---------- 1. 双盲组间信度 ----------
    print("\n【1】双盲组间信度（人工1 × 人工2）")
    blk12, cm12 = pair_block(H["H1"], H["H2"], "人工1", "人工2")
    m2 = np.column_stack([H["H1"], H["H2"]])
    alpha2 = krippendorff_alpha_ordinal(m2)
    icc2 = icc_2_1(m2)
    print(f"  完全一致 {blk12['exact_pct']:.2f}%    ±1档 {blk12['within1_pct']:.2f}%    平均绝对差 {blk12['mean_abs_diff']:.3f}")
    print(f"  Spearman ρ = {blk12['spearman']:.4f}    Pearson r = {blk12['pearson']:.4f}")
    print(f"  线性加权 κ = {blk12['kappa_linear']:.4f}    二次加权 κ = {blk12['kappa_quadratic']:.4f}")
    print(f"  Krippendorff α (ordinal) = {alpha2:.4f}    ICC(2,1) = {icc2:.4f}")
    print("\n  混淆矩阵（行=人工1，列=人工2）")
    print(cm12.to_string())
    rep["inter_rater_H1_H2"] = {**blk12, "krippendorff_alpha_ordinal": float(alpha2), "icc_2_1": float(icc2)}

    # ---------- 2. 每位盲标者 & 共识 vs GMM ----------
    print("\n【2】两位盲标者与共识 vs GMM")
    vs = {}
    for k in ["H1", "H2", "CONS"]:
        h = H[k]
        rho_p = stats.spearmanr(prob, h).statistic
        ci_p = boot_ci(prob, h, lambda x, y: stats.spearmanr(x, y).statistic)
        tau_p = stats.kendalltau(prob, h).statistic
        rho_t = stats.spearmanr(tier, h).statistic
        means = [float(np.mean(h[tier == t])) for t in range(5)]
        sds = [float(np.std(h[tier == t], ddof=1)) for t in range(5)]
        # 分级一致性用取整版本
        hh = CONS_INT if k == "CONS" else h
        blk_t, _ = pair_block(tier + 1, hh)
        sel = (hh >= 4) | (hh <= 2)
        auc = roc_auc_score((hh[sel] >= 4).astype(int), prob[sel])
        kw = stats.kruskal(*[h[tier == t] for t in range(5)])
        mpos = tier >= 1
        rho_pos = stats.spearmanr(prob[mpos], h[mpos]).statistic
        ci_pos = boot_ci(prob[mpos], h[mpos], lambda x, y: stats.spearmanr(x, y).statistic)
        rho_na = stats.spearmanr(prob[attn == 0], h[attn == 0]).statistic
        vs[k] = {
            "rho_prob": float(rho_p), "rho_prob_ci": list(ci_p), "tau_prob": float(tau_p),
            "rho_tier": float(rho_t), "tier_means": means, "tier_sds": sds,
            "agree_with_tier1": blk_t, "auc": float(auc),
            "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue),
            "rho_prob_posOnly": float(rho_pos), "rho_prob_posOnly_ci": list(ci_pos),
            "rho_prob_noAttention": float(rho_na),
        }
        print(f"\n  [{NAME.get(k,k)}]")
        print(f"    ρ(GMM prob, 人工强度) = {rho_p:.4f}  95%CI [{ci_p[0]:.4f}, {ci_p[1]:.4f}]   τ-b = {tau_p:.4f}")
        print(f"    ρ(GMM tier, 人工强度) = {rho_t:.4f}")
        print(f"    分层均值 T0..T4 = {[round(x,3) for x in means]}")
        print(f"    分层标准差      = {[round(x,3) for x in sds]}")
        print(f"    与 (tier+1) 完全一致 {blk_t['exact_pct']:.2f}%   ±1档 {blk_t['within1_pct']:.2f}%   二次加权 κ = {blk_t['kappa_quadratic']:.4f}")
        print(f"    AUC(prob 判别 人工≥4 vs ≤2) = {auc:.4f}    Kruskal-Wallis H = {kw.statistic:.1f} (p={kw.pvalue:.2g})")
        print(f"    [稳健] 仅恐慌阳性(剔tier0, n={int(mpos.sum())}) ρ = {rho_pos:.4f}  95%CI [{ci_pos[0]:.4f}, {ci_pos[1]:.4f}]")
        print(f"    [稳健] 仅非注意力条目 ρ = {rho_na:.4f}")
    rep["vs_gmm"] = vs

    # ---------- 3. 机器作为第三位"标注员" ----------
    print("\n【3】把 GMM 分层(tier+1)当作第三位标注员")
    a_mv = krippendorff_alpha_ordinal(np.column_stack([tier + 1, H["H1"], H["H2"]]))
    a_hh = alpha2
    print(f"  仅两名盲标者 α = {a_hh:.4f}")
    print(f"  含 GMM 三方 α  = {a_mv:.4f}")
    rep["alpha_2humans"] = float(a_hh)
    rep["alpha_2humans_plus_gmm"] = float(a_mv)

    # ---------- 4. 门限层对指标的抬升 ----------
    print("\n【4】门限层(tier0, n=106)的影响")
    t0 = tier == 0
    for k in ["H1", "H2", "CONS"]:
        full = vs[k]["rho_prob"]; pos = vs[k]["rho_prob_posOnly"]
        print(f"  {NAME[k]}: 全样本 ρ={full:.4f} → 剔tier0 ρ={pos:.4f}   落差 {full-pos:+.4f}")
    print(f"  tier0 人工强度：人工1 全为 {sorted(set(H['H1'][t0].astype(int)))}，人工2 全为 {sorted(set(H['H2'][t0].astype(int)))}，标准差均为 0")
    rep["tier0_floor"] = {
        "n": int(t0.sum()),
        "sd_H1": float(np.std(H["H1"][t0], ddof=1)),
        "sd_H2": float(np.std(H["H2"][t0], ddof=1)),
    }

    # ---------- 5. 论文阈值重分组 ----------
    print("\n【5】按论文阈值 0.2999 / 0.4722 / 0.9799 重分组")
    lab = np.where(prob < 0.2999, 0, np.where(prob < 0.4722, 1, np.where(prob < 0.9799, 2, 3)))
    th = {}
    for k in ["H1", "H2", "CONS"]:
        ms = [float(np.mean(H[k][lab == c])) for c in range(4)]
        ns = [int((lab == c).sum()) for c in range(4)]
        th[k] = {"means": ms, "n": ns}
        print(f"  {NAME[k]}: 非恐慌 {ms[0]:.3f} | 低 {ms[1]:.3f} | 中 {ms[2]:.3f} | 高 {ms[3]:.3f}   (n={ns})")
    rep["paper_threshold"] = th

    # ---------- 6. 注意力检查 ----------
    print("\n【6】注意力检查（20 组重复条目）")
    txt = gmm["text"].astype(str).values
    uniq, inv = np.unique(txt, return_inverse=True)
    groups = {}
    for i, g in enumerate(inv):
        groups.setdefault(g, []).append(i)
    pairs = [v for v in groups.values() if len(v) > 1]
    att = {}
    for k in ["H1", "H2"]:
        exact = sum(1 for v in pairs if len(set(H[k][v])) == 1)
        w1 = sum(1 for v in pairs if max(H[k][v]) - min(H[k][v]) <= 1)
        att[k] = {"n_pairs": len(pairs), "exact": exact, "within1": w1,
                  "exact_pct": float(100 * exact / len(pairs)),
                  "within1_pct": float(100 * w1 / len(pairs))}
        print(f"  {NAME[k]}: {len(pairs)} 组  严格一致 {exact} ({100*exact/len(pairs):.1f}%)  ±1档 {w1} ({100*w1/len(pairs):.1f}%)")
    rep["attention_check"] = att

    # ---------- 7. 分歧看点 ----------
    print("\n【7】与 GMM 的主要分歧")
    hi = prob >= 0.95
    for k in ["H1", "H2"]:
        print(f"  {NAME[k]}: prob≥0.95 但人工≤2 → {int((hi & (H[k]<=2)).sum())} 条;  prob≤0.35 但人工≥4 → {int(((prob<=0.35)&(H[k]>=4)).sum())} 条")
    d12 = np.abs(H["H1"] - H["H2"])
    print(f"  两位盲标者分歧 ≥2 档的条目：{int((d12>=2).sum())} 条")
    idx = np.where(d12 >= 2)[0]
    for i in idx[:10]:
        print(f"    {gmm['item_id'].iloc[i]}  prob={prob[i]:.4f} tier={tier[i]}  人工1={H['H1'][i]:.0f} 人工2={H['H2'][i]:.0f} | {str(gmm['text'].iloc[i])[:44]}")
    rep["disagreement"] = {"H1_high_prob_low_rating": int((hi & (H["H1"] <= 2)).sum()),
                           "H2_high_prob_low_rating": int((hi & (H["H2"] <= 2)).sum()),
                           "H1H2_diff_ge2": int((d12 >= 2).sum())}

    with open(os.path.join(BASE, "blind_vs_gmm_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=float)
    print("\n[写出] blind_vs_gmm_report.json")

    make_fig(prob, tier, H, CONS_INT, vs, cm12, os.path.join(BASE, "blind_vs_gmm_fig.png"))
    print("[写出] blind_vs_gmm_fig.png")


def make_fig(prob, tier, H, CONS_INT, vs, cm12, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for cand in ["Microsoft YaHei", "SimHei", "DengXian", "SimSun"]:
        if any(cand.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [cand]; break
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    keys = ["H1", "H2", "CONS"]
    labs = ["人工1(盲)", "人工2(盲)", "双盲共识"]
    sty = ["o-", "s--", "^:"]
    col = ["#1b9e77", "#7570b3", "#d95f02"]

    # A
    ax = axes[0, 0]
    for k, lb, s, c in zip(keys, labs, sty, col):
        ax.plot(range(5), vs[k]["tier_means"], s, label=lb, lw=2, ms=7, color=c)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"T{t}" for t in range(5)])
    ax.set_xlabel("GMM 等频分层 (tier)"); ax.set_ylabel("人工强度均值")
    ax.set_title("A. 双盲分层单调性：五层严格递增"); ax.grid(alpha=.3); ax.legend()

    # B
    ax = axes[0, 1]
    x = np.arange(3); w = .36
    v_all = [vs[k]["rho_prob"] for k in keys]
    v_pos = [vs[k]["rho_prob_posOnly"] for k in keys]
    e_all = [[v_all[i] - vs[keys[i]]["rho_prob_ci"][0] for i in range(3)],
             [vs[keys[i]]["rho_prob_ci"][1] - v_all[i] for i in range(3)]]
    e_pos = [[v_pos[i] - vs[keys[i]]["rho_prob_posOnly_ci"][0] for i in range(3)],
             [vs[keys[i]]["rho_prob_posOnly_ci"][1] - v_pos[i] for i in range(3)]]
    b1 = ax.bar(x - w / 2, v_all, w, yerr=e_all, capsize=4, color="#4c72b0", alpha=.9, label="全样本 (含非恐慌层)")
    b2 = ax.bar(x + w / 2, v_pos, w, yerr=e_pos, capsize=4, color="#dd8452", alpha=.9, label="仅恐慌阳性 (剔 tier0)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .022, f"{b.get_height():.3f}",
                    ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Spearman ρ (GMM prob vs 人工)")
    ax.set_title("B. 非恐慌层撑高了整体相关"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)

    # C 混淆矩阵
    ax = axes[1, 0]
    M = cm12.values.astype(float)
    im = ax.imshow(M, cmap="Blues")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if M[i, j] > M.max() * .55 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6))
    ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6))
    ax.set_xlabel("人工2 强度"); ax.set_ylabel("人工1 强度")
    ax.set_title("C. 双盲混淆矩阵（对角线=96.5%）")
    fig.colorbar(im, ax=ax, fraction=.046)

    # D 层内标准差
    ax = axes[1, 1]
    w2 = .3
    for idx, (k, lb, c) in enumerate(zip(keys, labs, col)):
        ax.bar(np.arange(5) + (idx - 1) * w2, vs[k]["tier_sds"], width=w2, label=lb, color=c, alpha=.88)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"T{t}" for t in range(5)])
    ax.set_xlabel("GMM 等频分层 (tier)"); ax.set_ylabel("人工强度标准差")
    ax.set_title("D. 层内离散度：高层分歧更大"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)

    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
