# -*- coding: utf-8 -*-
"""
RDD 春节/假日安慰剂 —— 先复现论文既有数值做校验，再跑新检验。
数据源：D:\\xinguanbiaozhu\\paper_figures_server\\daily_unified_metrics.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

SRC = r"D:\xinguanbiaozhu\paper_figures_server\daily_unified_metrics.csv"


def load():
    df = pd.read_csv(SRC, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    # days relative to lockdown 2020-01-23
    df['d'] = (df['date'] - pd.Timestamp('2020-01-23')).dt.days
    # panic coverage = panic posts / covid posts  (COVID denominator)
    df['cov'] = np.where(df['covid_posts'] > 0,
                         df['panic_posts'] / df['covid_posts'], np.nan)
    return df


def rdd(df, ycol, cutoff, h, kernel='triangular', exclude=None,
        hac_lags=3, se='HAC'):
    """Local-linear WLS discontinuity at `cutoff` in days-from-2020-01-23 units."""
    sub = df.copy()
    sub = sub[(sub['d'] >= cutoff - h) & (sub['d'] <= cutoff + h)]
    if exclude:
        lo, hi = exclude
        sub = sub[(sub['d'] < lo) | (sub['d'] > hi)]
    sub = sub.dropna(subset=[ycol]).sort_values('d').reset_index(drop=True)

    x = sub['d'].to_numpy(float) - cutoff
    y = sub[ycol].to_numpy(float)
    D = (x >= 0).astype(float)
    if kernel == 'triangular':
        w = np.maximum(0.0, 1.0 - np.abs(x) / h)
    else:
        w = np.ones_like(x)
    X = np.column_stack([np.ones_like(x), D, x, x * D])
    n, k = X.shape

    # WLS point estimate
    XWX = X.T @ (X * w[:, None])
    XWy = X.T @ (w * y)
    XWX_inv = np.linalg.pinv(XWX)
    beta = XWX_inv @ XWy
    e = y - X @ beta

    if se == 'HC1':
        u = X * (w * e)[:, None]
        S = (u.T @ u) * (n / (n - k))
    else:
        # Bartlett-kernel HAC on weighted scores u_t = w_t e_t x_t
        u = X * (w * e)[:, None]
        S = u.T @ u
        for l in range(1, hac_lags + 1):
            kk = 1.0 - l / (hac_lags + 1.0)
            G = u[l:].T @ u[:-l]
            S += kk * (G + G.T)
    V = XWX_inv @ S @ XWX_inv
    se_tau = float(np.sqrt(max(V[1, 1], 0.0)))
    tau = float(beta[1])
    tstat = tau / se_tau if se_tau > 0 else np.nan
    p = float(2 * stats.t.sf(abs(tstat), df=n - k)) if np.isfinite(tstat) else np.nan
    return dict(tau=tau, se=se_tau, p=p, n=n)


def fmt(r):
    return f"tau={r['tau']:+.4f}  SE={r['se']:.4f}  p={r['p']:.3f}  n={r['n']}"


def main():
    df = load()
    print("=" * 78)
    print("A. 数据口径校验：与论文 Table (daily/hourly) 对照")
    print("=" * 78)
    for ds in ['2020-01-19', '2020-01-20', '2020-01-21', '2020-01-22',
               '2020-01-23', '2020-01-24', '2020-01-25', '2020-01-26']:
        r = df[df['date'] == ds]
        if len(r):
            r = r.iloc[0]
            print(f"  {ds}  covid={int(r['covid_posts']):>7}  panic={int(r['panic_posts']):>6}  "
                  f"coverage={r['cov']*100:6.2f}%  overall_score={r['overall_panic_score']:.4f}")
    print()

    print("=" * 78)
    print("B. 复现校验：主估计（标准规格）—— 论文应为")
    print("   ±7  -0.0294/-0.0255 | ±14 -0.0289/-0.0238 | ±21 -0.0249/-0.0178 | ±30 +0.0042/-0.0129")
    print("=" * 78)
    for h in [7, 14, 21, 30]:
        a = rdd(df, 'cov', 0, h, se='HAC')
        b = rdd(df, 'overall_panic_score', 0, h, se='HAC')
        print(f"  ±{h:<2}  coverage : {fmt(a)}")
        print(f"       overall  : {fmt(b)}")
    print()
    print("  -- 主带宽 ±14 的 HC1 对照（论文：HAC(3) SE=0.0064 vs HC1 SE=0.0159）--")
    print(f"     HC1 : {fmt(rdd(df, 'cov', 0, 14, se='HC1'))}")
    print()

    print("=" * 78)
    print("C. 复现校验：Donut（排除 1/20-22）—— 论文应为")
    print("   ±7 +0.0014(p=.862) | ±14 -0.0209(p=.002) | ±21 -0.0164(p=.001) | ±30 +0.0302(p=.441)")
    print("=" * 78)
    for h in [7, 14, 21, 30]:
        a = rdd(df, 'cov', 0, h, exclude=(-3, -1), se='HAC')
        print(f"  ±{h:<2}  donut coverage : {fmt(a)}")


if __name__ == '__main__':
    main()
