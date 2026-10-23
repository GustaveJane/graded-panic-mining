#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantify how the denominator definition changes the daily panic series
and the RDD estimate at the Wuhan lockdown cutoff (2020-01-23).

Compares:
  OLD  daily_stats_detail_2019_2020.csv : panic_rate = panic_posts / total_posts * 100
  NEW  daily_stats_detail.csv           : panic_coverage = panic_posts / covid_posts

Runs standard RDD and Donut RDD (exclude 2020-01-20..22) at BW = 7/14/21/30,
with both HC1 and HAC(3) standard errors.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

CUTOFF = pd.Timestamp("2020-01-23")
DONUT_START = pd.Timestamp("2020-01-20")
DONUT_END = pd.Timestamp("2020-01-22")
START, END = pd.Timestamp("2019-12-01"), pd.Timestamp("2020-06-30")

NEW = r"C:\Users\17853\WorkBuddy\2026-08-20-09-20-57\_inbox_2026-09-15\daily_stats_detail.csv"
OLD = r"C:\Users\17853\Downloads\daily_stats_detail_2019_2020.csv"


def rdd(sub, outcome, bw, cov):
    if len(sub) < 4:
        return dict(tau=np.nan, se=np.nan, p=np.nan, n=len(sub))
    d = sub.copy()
    d["D"] = (d["day_rel"] >= 0).astype(float)
    d["X_c"] = d["day_rel"].astype(float)
    d["interact"] = d["D"] * d["X_c"]
    X = sm.add_constant(d[["D", "X_c", "interact"]])
    w = 1 - np.abs(d["day_rel"]) / bw
    w = np.clip(w, 0, None)
    if cov == "HAC":
        m = sm.WLS(d[outcome], X, weights=w).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    else:
        m = sm.WLS(d[outcome], X, weights=w).fit(cov_type="HC1")
    return dict(tau=float(m.params["D"]), se=float(m.bse["D"]),
                p=float(m.pvalues["D"]), n=len(d))


new = pd.read_csv(NEW, encoding="utf-8-sig")
old = pd.read_csv(OLD, encoding="utf-8-sig")
new["date"] = pd.to_datetime(new["date"])
old["date"] = pd.to_datetime(old["date"])

new["panic_coverage"] = np.where(new["covid_posts"] > 0,
                                 new["panic_posts"] / new["covid_posts"], 0.0)
new["old_style_rate"] = np.where(new["total_posts"] > 0,
                                 new["panic_posts"] / new["total_posts"] * 100, 0.0)

cmp = new.merge(old[["date", "panic_rate"]], on="date", how="inner")
cmp["diff"] = cmp["old_style_rate"] - cmp["panic_rate"]
print("=" * 78)
print("1) TWO DENOMINATOR DEFINITIONS ON THE SAME DAYS")
print("=" * 78)
print(f"days matched              : {len(cmp)}")
print(f"max |old_panic_rate - recomputed| : {cmp['diff'].abs().max():.6e}  (confirms old = panic/total*100)")
print(f"COVID-related share of all posts  : mean {100*(new['covid_posts']/new['total_posts']).replace([np.inf,-np.inf],np.nan).mean():.2f}%"
      f"  min {100*(new['covid_posts']/new['total_posts']).replace([np.inf,-np.inf],np.nan).min():.2f}%")

for label, df, col in [("OLD  panic_posts/total_posts*100", old, "panic_rate"),
                       ("NEW  panic_posts/covid_posts    ", new, "panic_coverage")]:
    sub = df[(df["date"] >= START) & (df["date"] <= END)]
    print(f"{label}: min {sub[col].min():.4f}  max {sub[col].max():.4f}  mean {sub[col].mean():.4f}")

print()
print("=" * 78)
print("2) RDD ESTIMATES: OLD DENOMINATOR (panic/total*100)")
print("=" * 78)
o = old[(old["date"] >= START) & (old["date"] <= END)].copy()
o["day_rel"] = (o["date"] - CUTOFF).dt.days
rows = []
for cov in ["HC1", "HAC"]:
    for bw in [7, 14, 21, 30]:
        std = rdd(o[(o["day_rel"] >= -bw) & (o["day_rel"] <= bw)], "panic_rate", bw, cov)
        dn = o[(o["day_rel"] >= -bw) & (o["day_rel"] <= bw)]
        dn = dn[~((dn["date"] >= DONUT_START) & (dn["date"] <= DONUT_END))]
        don = rdd(dn, "panic_rate", bw, cov)
        rows.append(dict(cov=cov, bw=bw, spec="Standard", **std))
        rows.append(dict(cov=cov, bw=bw, spec="Donut", **don))
old_res = pd.DataFrame(rows)
print(old_res.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

print()
print("=" * 78)
print("3) RDD ESTIMATES: NEW DENOMINATOR (panic/covid)")
print("=" * 78)
n = new[(new["date"] >= START) & (new["date"] <= END)].copy()
n["day_rel"] = (n["date"] - CUTOFF).dt.days
rows = []
for cov in ["HC1", "HAC"]:
    for bw in [7, 14, 21, 30]:
        std = rdd(n[(n["day_rel"] >= -bw) & (n["day_rel"] <= bw)], "panic_coverage", bw, cov)
        dn = n[(n["day_rel"] >= -bw) & (n["day_rel"] <= bw)]
        dn = dn[~((dn["date"] >= DONUT_START) & (dn["date"] <= DONUT_END))]
        don = rdd(dn, "panic_coverage", bw, cov)
        rows.append(dict(cov=cov, bw=bw, spec="Standard", **std))
        rows.append(dict(cov=cov, bw=bw, spec="Donut", **don))
new_res = pd.DataFrame(rows)
print(new_res.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

print()
print("=" * 78)
print("4) SIDE-BY-SIDE TAU (HAC, matching the old paper pipeline)")
print("=" * 78)
m = old_res[old_res["cov"] == "HAC"].merge(
    new_res[new_res["cov"] == "HAC"], on=["bw", "spec"], suffixes=("_old", "_new"))
m["ratio"] = (m["tau_new"] / m["tau_old"]).round(3)
print(m[["spec", "bw", "tau_old", "tau_new", "ratio",
         "se_old", "se_new", "p_old", "p_new", "n_old", "n_new"]].to_string(
    index=False, float_format=lambda x: f"{x:+.5f}"))

print()
print("=" * 78)
print("4b) SIDE-BY-SIDE TAU (HC1, what donut v2 / p1_2 use)")
print("=" * 78)
m2 = old_res[old_res["cov"] == "HC1"].merge(
    new_res[new_res["cov"] == "HC1"], on=["bw", "spec"], suffixes=("_old", "_new"))
m2["ratio"] = (m2["tau_new"] / m2["tau_old"]).round(3)
print(m2[["spec", "bw", "tau_old", "tau_new", "ratio",
          "p_old", "p_new"]].to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

print()
print("=" * 78)
print("5) KEY DAYS AROUND LOCKDOWN / SPRING FESTIVAL")
print("=" * 78)
key = new[(new["date"] >= "2020-01-17") & (new["date"] <= "2020-02-05")].copy()
key["old_pct"] = np.where(key["total_posts"] > 0, key["panic_posts"] / key["total_posts"] * 100, 0)
print(key[["date", "total_posts", "covid_posts", "panic_posts",
           "panic_coverage", "old_pct"]].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
