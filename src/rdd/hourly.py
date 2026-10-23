# -*- coding: utf-8 -*-
"""
小时级 RDD：以 2020-01-23 10:00 封城生效为 cutoff。

动机（老师批注 [P06][42]、[P15][47]）：
  1. 日粒度下"宣布(凌晨)"与"执行(10时)"同日，构不成两个独立断点 → 小时级可分。
  2. 春节(1/25)与封城(1/23)只隔2天 → 小时级同日内部对比天然规避跨日假期混淆。
  3. 1/20 人传人确认与 1/23 只隔3天，日粒度任何带宽都会吞掉前者 → 小时级可收窄带宽。

数据：/root/autodl-tmp/bh_full_hourly.csv
  列：hour,total_posts,covid_posts,panic_posts,panic_rate,overall_panic_score
  覆盖 2019-12-02 00:00 ~ 2020-02-10 23:00，共 1704 小时。

方法：局部线性 RDD（三角核 WLS）
  y = a + b*D + c*X + d*(D*X) + e,  D = 1(X>=0), tau = b
  权重 = 三角核 * covid_posts（对聚合数据做精度加权，避免凌晨小样本小时主导）
  标准误 = HAC(Newey-West)，小时级数据取 maxlags=24（覆盖日周期）
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "/root/autodl-tmp/bh_full_hourly.csv"
OUT = "/root/autodl-tmp/hourly_rdd_results.csv"

ENFORCE = pd.Timestamp("2020-01-23 10:00:00")   # 封城生效
ANNOUNCE = pd.Timestamp("2020-01-23 00:00:00")  # 公告（凌晨）

h = pd.read_csv(DATA, encoding="utf-8-sig")
h.columns = [c.lstrip("\ufeff").strip() for c in h.columns]
h["hour"] = pd.to_datetime(h["hour"])
h = h.sort_values("hour").reset_index(drop=True)
print("hourly range:", h.hour.min(), "->", h.hour.max(), "n =", len(h))


def fit_rdd(df, cutoff, bw_hours, outcome, cov="HAC", maxlags=24, min_posts=1):
    """局部线性 RDD，三角核 WLS。返回 (tau, se, p, n, n_left, n_right)."""
    d = df[df["covid_posts"] >= min_posts].copy()
    d["X"] = (d["hour"] - cutoff).dt.total_seconds() / 3600.0
    d = d[d["X"].abs() <= bw_hours].copy()
    if len(d) < 20:
        return None
    d["D"] = (d["X"] >= 0).astype(float)

    # 三角核
    d["kern"] = 1.0 - (d["X"].abs() / bw_hours)
    d["w"] = d["kern"] * d["covid_posts"].clip(lower=1)

    X = np.column_stack([
        np.ones(len(d)),
        d["D"].values,
        d["X"].values,
        (d["D"] * d["X"]).values,
    ])
    y = d[outcome].astype(float).values

    # 已按核加权，再喂 WLS 时用 w 的平方根
    model = sm.WLS(y, X, weights=d["w"].values)
    if cov == "HAC":
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    else:
        res = model.fit(cov_type="HC1")

    tau = res.params[1]
    se = res.bse[1]
    p = res.pvalues[1]
    return dict(tau=tau, se=se, p=p, n=len(d),
                n_left=int((d["X"] < 0).sum()), n_right=int((d["X"] >= 0).sum()))


rows = []
for outcome in ["panic_rate", "overall_panic_score"]:
    for cutoff_name, cutoff in [("enforce_10:00", ENFORCE), ("announce_00:00", ANNOUNCE)]:
        for bw in [24, 48, 72, 120, 168]:
            # Standard
            for spec, kw in [("Standard", {}),
                             ("Donut_announce", {"exclude": (ANNOUNCE, ENFORCE)})]:
                df = h.copy()
                if "exclude" in kw:
                    lo, hi = kw["exclude"]
                    df = df[~((df["hour"] >= lo) & (df["hour"] < hi))]
                for min_posts in [1, 30]:
                    r = fit_rdd(df, cutoff, bw, outcome, "HAC", 24, min_posts)
                    if r is None:
                        continue
                    rows.append(dict(outcome=outcome, cutoff=cutoff_name, bw_h=bw,
                                     spec=spec, min_posts=min_posts,
                                     tau=round(r["tau"], 5),
                                     se=round(r["se"], 5),
                                     t=round(r["tau"] / r["se"], 3),
                                     p=round(r["p"], 5),
                                     n=r["n"], n_left=r["n_left"], n_right=r["n_right"]))

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200)
print("\n" + "=" * 100)
print("小时级 RDD（HAC maxlags=24，三角核，精度加权）")
print("=" * 100)
print(res[res["min_posts"] == 1].to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
print("\n--- min_posts=30（剔除凌晨小样本小时）---")
print(res[res["min_posts"] == 30].to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
print("\nsaved:", OUT)
