# -*- coding: utf-8 -*-
"""小时级 RDD 的安慰剂断点检验 + 事件时间剖面导出。

目的：检验 1/23 10:00 的零结果是不是"特异性"的。
若 1/21、1/22、1/24、1/25、1/27 等安慰剂断点同样无显著断点，
说明小时级数据在整个衰减相位上都测不到离散转折，
支持把 RDD 降格为描述性检验，并回应魔鬼代言人的零假设。
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "/root/autodl-tmp/bh_full_hourly.csv"
h = pd.read_csv(DATA, encoding="utf-8-sig")
h.columns = [c.lstrip("\ufeff").strip() for c in h.columns]
h["hour"] = pd.to_datetime(h["hour"])
h = h.sort_values("hour").reset_index(drop=True)


def fit_rdd(df, cutoff, bw_hours, outcome, maxlags=24):
    d = df.copy()
    d["X"] = (d["hour"] - cutoff).dt.total_seconds() / 3600.0
    d = d[d["X"].abs() <= bw_hours].copy()
    if len(d) < 20:
        return None
    d["D"] = (d["X"] >= 0).astype(float)
    d["w"] = (1.0 - d["X"].abs() / bw_hours) * d["covid_posts"].clip(lower=1)
    X = np.column_stack([np.ones(len(d)), d["D"].values, d["X"].values, (d["D"] * d["X"]).values])
    y = d[outcome].astype(float).values
    res = sm.WLS(y, X, weights=d["w"].values).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return dict(tau=res.params[1], se=res.bse[1], p=res.pvalues[1], n=len(d))


# 安慰剂断点：封城前后各取若干日（均为 10:00）
placebo_days = ["2020-01-18", "2020-01-19", "2020-01-21", "2020-01-22",
                "2020-01-23", "2020-01-24", "2020-01-25", "2020-01-27",
                "2020-01-29", "2020-01-31"]
rows = []
for day in placebo_days:
    cutoff = pd.Timestamp(day + " 10:00:00")
    for outcome in ["panic_rate", "overall_panic_score"]:
        for bw in [48, 72, 120]:
            r = fit_rdd(h, cutoff, bw, outcome)
            if r is None:
                continue
            rows.append(dict(cutoff_date=day, is_real=(day == "2020-01-23"),
                             outcome=outcome, bw_h=bw,
                             tau=round(r["tau"], 5), se=round(r["se"], 5),
                             t=round(r["tau"] / r["se"], 3), p=round(r["p"], 5), n=r["n"]))

pl = pd.DataFrame(rows)
pl.to_csv("/root/autodl-tmp/hourly_rdd_placebo.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print("=" * 90)
print("安慰剂断点检验（cutoff 均为 10:00，HAC maxlags=24）")
print("=" * 90)
for oc in ["panic_rate", "overall_panic_score"]:
    print(f"\n--- {oc} ---")
    print(pl[pl.outcome == oc].to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

# 事件时间剖面：以 1/23 10:00 为 0 点
h["evt_h"] = (h["hour"] - pd.Timestamp("2020-01-23 10:00:00")).dt.total_seconds() / 3600.0
prof = h[(h.evt_h >= -168) & (h.evt_h <= 168)][
    ["hour", "evt_h", "total_posts", "covid_posts", "panic_posts",
     "panic_rate", "overall_panic_score"]]
prof.to_csv("/root/autodl-tmp/hourly_event_profile.csv", index=False, encoding="utf-8-sig")
print("\nevent profile saved: /root/autodl-tmp/hourly_event_profile.csv", prof.shape)
