# -*- coding: utf-8 -*-
"""
补 1：RDD 春节 / 假日安慰剂

数据源（权威，与论文主表同源）：
  daily_stats_detail.csv          2019-12-01 ~ 2020-12-31（397 天）
  daily_unified_metrics.csv       2019-12-01 ~ 2020-06-30（含 overall_panic_score）

实现与论文一致：WLS + 三角核 + 两侧异斜率局部线性，HAC(3) 为主估计。
选窗规则：以断点日为中心 ±h 天（对断点=2020-01-23 时与论文等价）。
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"D:\xinguanbiaozhu\rdd_holiday"
LOCKDOWN = pd.Timestamp('2020-01-23')


def load_daily():
    """主数据源 = daily_unified_metrics.csv（2026-09-15 重建，与论文主表同版）；
    其覆盖 2019-12-01~2020-06-30，2020-07 之后补 daily_stats_detail.csv（仅供国庆安慰剂使用）。

    注：两份文件在 2019-12-30 与 12-31 上不一致。12-30 是退化日
    （covid_posts=1, panic_posts=2 -> panic_rate=2.0），单日即可让 ±30 规格的
    前侧均值抬升 6.7pp。故一律以 unified 版为准。
    """
    u = pd.read_csv(f"{BASE}/daily_unified_metrics.csv", encoding='utf-8-sig')
    u.columns = [c.strip() for c in u.columns]
    u['date'] = pd.to_datetime(u['date'])
    u = u[['date', 'total_posts', 'covid_posts', 'panic_posts',
           'panic_rate', 'overall_panic_score']].copy()
    u = u.sort_values('date').reset_index(drop=True)

    s = pd.read_csv(f"{BASE}/daily_stats_detail.csv", encoding='utf-8-sig')
    s.columns = [c.strip() for c in s.columns]
    s['date'] = pd.to_datetime(s['date'])
    s['panic_rate'] = np.where(s['covid_posts'] > 0,
                              s['panic_posts'] / s['covid_posts'], 0.0)
    s['overall_panic_score'] = np.nan
    s = s[['date', 'total_posts', 'covid_posts', 'panic_posts',
           'panic_rate', 'overall_panic_score']]
    tail = s[s['date'] > u['date'].max()].copy()

    df = pd.concat([u, tail], ignore_index=True).sort_values('date').reset_index(drop=True)
    return df


def load_unified():
    df = pd.read_csv(f"{BASE}/daily_unified_metrics.csv", encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def local_linear_core(running, y, w, use_hac=True, hac_lags=3):
    running = np.asarray(running, float)
    treated = (running >= 0).astype(float)
    X = pd.DataFrame({
        "const": 1.0, "treated": treated,
        "running": running, "running_x_treated": running * treated,
    })
    m = sm.WLS(np.asarray(y, float), X, weights=np.asarray(w, float))
    if use_hac:
        return m.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return m.fit(cov_type="HC1")


def rdd(daily, outcome, cutoff, bandwidth, exclude_window=None,
        min_covid_posts=0, use_hac=True, hac_lags=3):
    cutoff = pd.Timestamp(cutoff)
    sub = daily.loc[(daily['date'] - cutoff).dt.days.abs() <= bandwidth].copy()
    sub['running'] = (sub['date'] - cutoff).dt.days
    if exclude_window is not None:
        lo, hi = exclude_window
        sub = sub.loc[~((sub['date'] >= pd.Timestamp(lo)) & (sub['date'] < pd.Timestamp(hi)))].copy()
    if min_covid_posts > 0:
        sub = sub.loc[sub['covid_posts'] >= min_covid_posts].copy()
    sub = sub.dropna(subset=[outcome])
    n = len(sub)
    if n < 6:
        return None
    w = np.maximum(1 - np.abs(sub['running'].values) / bandwidth, 0.0)
    fit = local_linear_core(sub['running'].values, sub[outcome].values, w, use_hac, hac_lags)
    return dict(
        cutoff=str(cutoff.date()), bandwidth=bandwidth, n=n,
        tau=float(fit.params['treated']), se=float(fit.bse['treated']),
        p=float(fit.pvalues['treated']),
        pre=float(sub.loc[sub['running'] < 0, outcome].mean()),
        post=float(sub.loc[sub['running'] >= 0, outcome].mean()),
    )


def show(tag, r):
    if r is None:
        print(f"  {tag:<34} 样本不足")
        return
    star = '***' if r['p'] < 0.01 else ('**' if r['p'] < 0.05 else ('*' if r['p'] < 0.10 else ''))
    print(f"  {tag:<34} tau={r['tau']:+.4f}  SE={r['se']:.4f}  p={r['p']:.4f}{star:<3} "
          f"n={r['n']:<3} pre={r['pre']*100:5.2f}% post={r['post']*100:5.2f}%")


def main():
    d = load_daily()
    print("=" * 96)
    print("0. 数据校验：daily_stats_detail.csv 覆盖与断点窗口")
    print("=" * 96)
    print(f"  日期范围 {d['date'].min().date()} ~ {d['date'].max().date()}  ({len(d)} 天)")
    for ds in ['2020-01-19', '2020-01-20', '2020-01-21', '2020-01-22',
               '2020-01-23', '2020-01-24', '2020-01-25', '2020-01-26']:
        r = d[d['date'] == ds]
        if len(r):
            r = r.iloc[0]
            print(f"    {ds} covid={int(r['covid_posts']):>7} panic={int(r['panic_posts']):>6} "
                  f"coverage={r['panic_rate']*100:6.2f}%")
    print()

    print("=" * 96)
    print("1. 复现校验（必须与论文一致）")
    print("   论文：±7 -0.0294(SE .0109) | ±14 -0.0289(SE .0064) | ±21 -0.0249(SE .0054) | ±30 +0.0042")
    print("   donut(排1/20-22)：±7 +0.0014 | ±14 -0.0209(p .0023) | ±21 -0.0164(p .0008) | ±30 +0.0302(p .4410)")
    print("=" * 96)
    for h in [7, 14, 21, 30]:
        show(f"standard ±{h}", rdd(d, 'panic_rate', LOCKDOWN, h))
    print()
    print("  -- ±30 剔除退化日 2019-12-30（covid=1, panic=2 -> rate=2.0）--")
    show("standard ±30 (去 12/30)",
         rdd(d, 'panic_rate', LOCKDOWN, 30,
             exclude_window=('2019-12-30', '2019-12-31')))
    print()
    for h in [7, 14, 21, 30]:
        show(f"donut(1/20-22) ±{h}",
             rdd(d, 'panic_rate', LOCKDOWN, h,
                 exclude_window=('2020-01-20', '2020-01-23')))
    print()

    print("=" * 96)
    print("2. 【B 路】春节安慰剂：把断点移到 2020-01-25（除夕 1/24）")
    print("=" * 96)
    for h in [7, 14, 21]:
        show(f"春节 placebo ±{h}", rdd(d, 'panic_rate', '2020-01-25', h))
    print()
    print("  -- 对照：同带宽下 1/23 真断点 --")
    for h in [7, 14, 21]:
        show(f"真实 1/23 ±{h}", rdd(d, 'panic_rate', LOCKDOWN, h))
    print()

    print("=" * 96)
    print("3. 【B 路】延长 donut：排除 2020-01-20 ~ 01-27（含春节峰值）")
    print("=" * 96)
    for h in [14, 21, 30]:
        show(f"donut(1/20-27) ±{h}",
             rdd(d, 'panic_rate', LOCKDOWN, h,
                 exclude_window=('2020-01-20', '2020-01-28')))
    print()
    print("  -- 逐日排除窗口敏感性（带宽 ±14）--")
    for lo, hi, lab in [('2020-01-20', '2020-01-23', '排 1/20-22'),
                        ('2020-01-20', '2020-01-25', '排 1/20-24'),
                        ('2020-01-20', '2020-01-26', '排 1/20-25'),
                        ('2020-01-20', '2020-01-27', '排 1/20-26'),
                        ('2020-01-20', '2020-01-28', '排 1/20-27'),
                        ('2020-01-24', '2020-01-26', '排 1/24-25(仅春节)')]:
        show(f"{lab} ±14", rdd(d, 'panic_rate', LOCKDOWN, 14, exclude_window=(lo, hi)))
    print()

    print("=" * 96)
    print("4. 【A 路】假日构成效应安慰剂：同一设计搬到 2020 年内其他节日")
    print("=" * 96)
    holidays = [('2020-04-04', '清明'), ('2020-05-01', '劳动节'),
                ('2020-06-25', '端午'), ('2020-10-01', '国庆')]
    for ds, name in holidays:
        for h in [7, 14, 21]:
            show(f"{name} {ds} ±{h}", rdd(d, 'panic_rate', ds, h, min_covid_posts=1))
        print()
    print("  -- 对照：1/23 真断点（加 min_covid_posts=1 过滤后）--")
    for h in [7, 14, 21]:
        show(f"真实 1/23 ±{h} (min_covid=1)", rdd(d, 'panic_rate', LOCKDOWN, h, min_covid_posts=1))
    print()

    print("=" * 96)
    print("5. 安慰剂断点扫描（null 分布）：2020-02 至 2020-11 每 7 天一个伪断点，带宽 ±14")
    print("=" * 96)
    scan = []
    for dt in pd.date_range('2020-02-07', '2020-11-30', freq='7D'):
        r = rdd(d, 'panic_rate', dt, 14, min_covid_posts=1)
        if r:
            scan.append(r)
    if scan:
        taus = np.array([x['tau'] for x in scan])
        ps = np.array([x['p'] for x in scan])
        true_tau = rdd(d, 'panic_rate', LOCKDOWN, 14)['tau']
        print(f"  伪断点个数 {len(taus)}")
        print(f"  tau 分布: mean={taus.mean():+.4f}  sd={taus.std():.4f}  "
              f"min={taus.min():+.4f}  max={taus.max():+.4f}")
        print(f"  真断点 tau = {true_tau:+.4f}")
        print(f"  |tau| >= |真值| 的伪断点占比: {(np.abs(taus) >= abs(true_tau)).mean()*100:.1f}%")
        print(f"  p<0.05 的伪断点占比: {(ps < 0.05).mean()*100:.1f}%")
        print(f"  负向且 p<0.10 的伪断点: {int(((taus < 0) & (ps < 0.10)).sum())} / {len(taus)}")
        pd.DataFrame(scan).to_csv(f"{BASE}/rdd_placebo_scan.csv", index=False,
                                  encoding='utf-8-sig')
    print()

    print("=" * 96)
    print("6. overall_panic_score 侧复核（数据仅到 2020-06-30）")
    print("=" * 96)
    u = load_unified()
    for ds, name in [('2020-04-04', '清明'), ('2020-05-01', '劳动节'), ('2020-06-25', '端午')]:
        show(f"{name} {ds} ±14 (overall)", rdd(u, 'overall_panic_score', ds, 14))
    show("春节 1/25 ±14 (overall)", rdd(u, 'overall_panic_score', '2020-01-25', 14))
    show("真实 1/23 ±14 (overall)", rdd(u, 'overall_panic_score', LOCKDOWN, 14))


if __name__ == '__main__':
    main()
