# -*- coding: utf-8 -*-
"""HAC 带宽(lag)敏感性与残差自相关诊断。

背景（老师批注 [P06][42]）：
  "聚合时间序列 RDD 存在强自相关，常规标准误严重低估；
   p=0.029 在校正自相关后大概率彻底消失。"

实测发现相反：HAC 下 SE 反而变小、p 更小。本脚本查明机制：
  1) 拟合局部线性 RDD（三角核 WLS），取残差
  2) 报告残差 ACF(1..5)、Ljung-Box
  3) 报告 HC1 与 HAC(maxlags=0..10) 的 SE 序列，看结论对 lag 选择的敏感性

数据：/root/autodl-tmp/daily_stats_detail.csv（正确口径 panic/covid）
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

CUTOFF = pd.Timestamp("2020-01-23")
DONUT = (pd.Timestamp("2020-01-20"), pd.Timestamp("2020-01-23"))

d = pd.read_csv("/root/autodl-tmp/daily_stats_detail.csv", encoding="utf-8-sig")
d["date"] = pd.to_datetime(d["date"])
d = d.sort_values("date").reset_index(drop=True)
d["outcome"] = np.where(d["covid_posts"] > 0, d["panic_posts"] / d["covid_posts"], 0.0)
d["day_rel"] = (d["date"] - CUTOFF).dt.days


def fit(df, bw, exclude=None):
    s = df[df["day_rel"].abs() <= bw].copy()
    if exclude is not None:
        s = s[~((s["date"] >= exclude[0]) & (s["date"] < exclude[1]))]
    if len(s) < 12:
        return None
    X = s["day_rel"].values.astype(float)
    D = (X >= 0).astype(float)
    A = np.column_stack([np.ones(len(s)), D, X, D * X])
    y = s["outcome"].values.astype(float)
    w = 1.0 - np.abs(X) / bw
    out = {"n": len(s), "A": A, "y": y, "w": w, "X": X}
    return out


print("=" * 96)
print("HAC lag 敏感性：因变量 = panic_posts / covid_posts，断点 2020-01-23，三角核 WLS")
print("=" * 96)

for spec, excl in [("Standard", None), ("Donut 1/20-22", DONUT)]:
    print("\n" + "-" * 96)
    print("[%s]" % spec)
    print("-" * 96)
    print("%4s %3s %10s %10s %9s  %s" % ("bw", "n", "tau", "HC1_se", "HC1_p", "HAC_se by maxlags (p)"))
    for bw in [7, 14, 21, 30]:
        r = fit(d, bw, excl)
        if r is None:
            continue
        m = sm.WLS(r["y"], r["A"], weights=r["w"])
        hc1 = m.fit(cov_type="HC1")
        cells = []
        for L in [0, 1, 2, 3, 5, 7, 10]:
            if L == 0:
                f = m.fit(cov_type="HAC", cov_kwds={"maxlags": 0})
            else:
                f = m.fit(cov_type="HAC", cov_kwds={"maxlags": L})
            cells.append("L%-2d:%.5f(%.3f)" % (L, f.bse[1], f.pvalues[1]))
        print("%4d %3d %+10.5f %10.5f %9.5f  %s"
              % (bw, r["n"], hc1.params[1], hc1.bse[1], hc1.pvalues[1], " ".join(cells)))

print("\n" + "=" * 96)
print("残差自相关诊断（bw=14, Standard）")
print("=" * 96)
r = fit(d, 14, None)
m = sm.WLS(r["y"], r["A"], weights=r["w"])
res = m.fit(cov_type="HC1").resid
n = len(res)
print("n = %d" % n)
for k in range(1, 6):
    ac = float(np.corrcoef(res[:-k], res[k:])[0, 1])
    print("  ACF(lag %d) = %+.4f" % (k, ac))
lb = acorr_ljungbox(res, lags=[1, 2, 3, 5], return_df=True)
print("\n  Ljung-Box:")
print(lb.to_string())

print("\n" + "=" * 96)
print("解读要点")
print("=" * 96)
r14 = fit(d, 14, None)
m14 = sm.WLS(r14["y"], r14["A"], weights=r14["w"])
h1 = m14.fit(cov_type="HC1")
h3 = m14.fit(cov_type="HAC", cov_kwds={"maxlags": 3})
print("  bw=14 Standard  HC1 : tau=%.5f SE=%.5f p=%.5f" % (h1.params[1], h1.bse[1], h1.pvalues[1]))
print("  bw=14 Standard  HAC3: tau=%.5f SE=%.5f p=%.5f" % (h3.params[1], h3.bse[1], h3.pvalues[1]))
print("  -> SE %s (HC1 -> HAC3)" % ("变小" if h3.bse[1] < h1.bse[1] else "变大"))
