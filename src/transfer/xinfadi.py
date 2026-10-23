# -*- coding: utf-8 -*-
"""新发地迁移复现（本地分析版）

输入：D:/xinguanbiaozhu/xinfadi_replication/sub_{jan,may,jun,jul}_{daily,pos}.csv
  - *_daily.csv: date, n_covid, n_panic
  - *_pos.csv  : date, panic_prob（仅 panic_prob>=0.3）

三套阈值口径：
  (i)  论文主口径（全时段拟合）t1=0.4722, t2=0.9799
  (ii) 仅一月拟合（2020-01 恐慌阳性）→ 冻结套用   ← 真 out-of-sample
  (iii)六月窗口重估（20/80 分位校准）
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import norm

BASE = r'D:/xinguanbiaozhu/xinfadi_replication'
BIN = 0.3
PAPER_T1, PAPER_T2 = 0.4722, 0.9799
PAPER_MEANS = [0.366, 0.517, 0.719, 0.906, 0.985]
PAPER_BOUNDS = [0.4317, 0.6145, 0.8220, 0.9569]

BASE_LO, BASE_HI = '2020-05-09', '2020-06-08'
RESP_LO, RESP_HI = '2020-06-11', '2020-06-13'
REC_LO, REC_HI = '2020-06-19', '2020-06-30'


def read_daily(prefix):
    rows = {}
    with open(f'{BASE}/sub_{prefix}_daily.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows[r['date']] = (int(r['n_covid']), int(r['n_panic']))
    return rows


def read_pos(prefix):
    out = []
    with open(f'{BASE}/sub_{prefix}_pos.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out.append(float(r['panic_prob']))
    return np.asarray(out)


daily = {p: read_daily(p) for p in ('jan', 'may', 'jun', 'jul')}
pos = {p: read_pos(p) for p in ('jan', 'jun')}

# 合并 5-7 月的日度序列
merged = {}
for p in ('may', 'jun', 'jul'):
    merged.update(daily[p])
series = [dict(date=d, n_covid=c, n_panic=n,
               coverage=round(100 * n / c, 4) if c else None)
          for d, (c, n) in sorted(merged.items())]

print(f'日度序列 {len(series)} 天  ({series[0]["date"]} ~ {series[-1]["date"]})')
print(f'一月恐慌阳性 {len(pos["jan"]):,} 条 | 六月恐慌阳性 {len(pos["jun"]):,} 条')


def seg(lo, hi, src=series):
    return [r['coverage'] for r in src if lo <= r['date'] <= hi and r['coverage'] is not None]


base, resp, rec = seg(BASE_LO, BASE_HI), seg(RESP_LO, RESP_HI), seg(REC_LO, REC_HI)
b_mean, b_sd = float(np.mean(base)), float(np.std(base, ddof=1))
r_mean = float(np.mean(resp))
peak = max([r for r in series if r['coverage'] is not None], key=lambda r: r['coverage'])

summary = dict(
    baseline_window=f'{BASE_LO}~{BASE_HI}', baseline_days=len(base),
    baseline_mean_pct=round(b_mean, 4), baseline_sd_pct=round(b_sd, 4),
    response_window=f'{RESP_LO}~{RESP_HI}', response_mean_pct=round(r_mean, 4),
    delta_pp=round(r_mean - b_mean, 4),
    z_score=round((r_mean - b_mean) / b_sd, 2) if b_sd > 0 else None,
    peak_date=peak['date'], peak_coverage_pct=peak['coverage'],
    fold_over_baseline=round(peak['coverage'] / b_mean, 2),
    recovery_window=f'{REC_LO}~{REC_HI}',
    recovery_mean_pct=round(float(np.mean(rec)), 4),
    recovery_vs_baseline_ratio=round(float(np.mean(rec)) / b_mean, 3),
    n_days_above_baseline_until=f'首个 <=baseline 的日期：' +
        str(next((r['date'] for r in series if r['date'] > '2020-06-13'
                  and r['coverage'] is not None and r['coverage'] <= b_mean), None)),
)

print('\n--- 事件指标 ---')
print(json.dumps(summary, ensure_ascii=False, indent=2))


def natural_boundaries(mus, sds, ws):
    order = np.argsort(mus)
    mus, sds, ws = mus[order], sds[order], ws[order]
    xs = np.linspace(mus[0] - 6 * sds[0], mus[-1] + 6 * sds[-1], 400001)
    bounds = []
    for k in range(len(mus) - 1):
        pa = ws[k] * norm.pdf(xs, mus[k], sds[k])
        pb = ws[k + 1] * norm.pdf(xs, mus[k + 1], sds[k + 1])
        cand = np.where(np.diff(np.sign(pa - pb)) != 0)[0]
        cand = cand[(xs[cand] > mus[k]) & (xs[cand] < mus[k + 1])]
        bounds.append(float(xs[cand[0]]) if len(cand) else float('nan'))
    return bounds


def fit_gmm(probs, tag):
    X = probs.reshape(-1, 1)
    rows = []
    for k in range(1, 9):
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=0, n_init=3).fit(X)
        sil = float('nan') if k == 1 else float(
            silhouette_score(X, gm.predict(X), sample_size=20000, random_state=0))
        rows.append(dict(K=k, BIC=round(float(gm.bic(X)), 1), AIC=round(float(gm.aic(X)), 1),
                         silhouette=None if np.isnan(sil) else round(sil, 4),
                         means=[round(float(m), 4) for m in np.sort(gm.means_.ravel())]))
    best = max([r for r in rows if r['silhouette'] is not None], key=lambda r: r['silhouette'])
    K = best['K']
    gm = GaussianMixture(n_components=K, covariance_type='full', random_state=0, n_init=3).fit(X)
    idx = np.argsort(gm.means_.ravel())
    mus = gm.means_.ravel()[idx]
    sds = np.sqrt(gm.covariances_.ravel())[idx]
    ws = gm.weights_[idx]
    bounds = natural_boundaries(mus, sds, ws)
    print(f'\n[{tag}] n={len(X):,}  silhouette 最优 K={K}')
    print(f'   成分均值 {[round(float(m), 4) for m in mus]}')
    print(f'   自然边界 {[None if np.isnan(b) else round(b, 4) for b in bounds]}')
    return rows, K, [float(m) for m in mus], bounds


sel_jan, K_jan, means_jan, bounds_jan = fit_gmm(pos['jan'], '2020-01')
sel_jun, K_jun, means_jun, bounds_jun = fit_gmm(pos['jun'], '2020-06')


def fit_k5(probs, tag):
    """固定 K=5（与论文一致），用于成分均值/边界对比。"""
    X = probs.reshape(-1, 1)
    gm = GaussianMixture(n_components=5, covariance_type='full',
                         random_state=0, n_init=3).fit(X)
    idx = np.argsort(gm.means_.ravel())
    mus = gm.means_.ravel()[idx]
    sds = np.sqrt(gm.covariances_.ravel())[idx]
    ws = gm.weights_[idx]
    b = natural_boundaries(mus, sds, ws)
    print(f'[{tag}] K=5 成分均值 {[round(float(m), 4) for m in mus]}')
    print(f'        自然边界 {[None if np.isnan(x) else round(x, 4) for x in b]}')
    return [float(m) for m in mus], b


means_jan5, bounds_jan5 = fit_k5(pos['jan'], '2020-01')
means_jun5, bounds_jun5 = fit_k5(pos['jun'], '2020-06')

q_jan = np.quantile(pos['jan'], [0.20, 0.80])
q_jun = np.quantile(pos['jun'], [0.20, 0.80])


def shares(p, t1, t2):
    n = len(p)
    return dict(low=round(100 * float((p < t1).sum()) / n, 2),
                medium=round(100 * float(((p >= t1) & (p < t2)).sum()) / n, 2),
                high=round(100 * float((p >= t2).sum()) / n, 2), n_positive=n)


rows = []
for name, t1, t2 in [
    ('(i) Paper thresholds (full-window fit)', PAPER_T1, PAPER_T2),
    ('(ii) January-refit frozen (20/80)', float(q_jan[0]), float(q_jan[1])),
    ('(iii) June refit (20/80)', float(q_jun[0]), float(q_jun[1])),
]:
    rows.append(dict(rule=name, t1=round(t1, 4), t2=round(t2, 4), **shares(pos['jun'], t1, t2)))

print('\n--- 六月窗口在三套阈值下的层级构成（分母=六月恐慌阳性）---')
for r in rows:
    print(f"  {r['rule']}\n     t1={r['t1']} t2={r['t2']}  low={r['low']}%  med={r['medium']}%  high={r['high']}%")

mean_drift = [dict(idx=i + 1, paper=pm, june_refit=round(jm, 4), delta=round(jm - pm, 4))
              for i, (pm, jm) in enumerate(zip(PAPER_MEANS, means_jun5))]
bounds_drift = [dict(idx=i + 1, paper=pb,
                     june_refit=None if np.isnan(jb) else round(jb, 4),
                     delta=None if np.isnan(jb) else round(jb - pb, 4))
                for i, (pb, jb) in enumerate(zip(PAPER_BOUNDS, bounds_jun5))]
max_dev = max(abs(d['delta']) for d in mean_drift)

print('\n--- 成分均值对比（论文 vs 六月重估）---')
for d in mean_drift:
    print(f"  #{d['idx']}  paper={d['paper']}  june={d['june_refit']}  Δ={d['delta']:+.4f}")
print(f'  最大偏差 = {max_dev:.4f}')

# ---------------- 绘图 ----------------
sel = [r for r in series if '2020-05-01' <= r['date'] <= '2020-07-15' and r['coverage'] is not None]
xs = [datetime.strptime(r['date'], '%Y-%m-%d') for r in sel]
ys = np.asarray([r['coverage'] for r in sel])
ma = np.convolve(ys, np.ones(3) / 3, mode='same')

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(xs, ys, marker='o', ms=3, lw=1.0, color='#4C72B0', alpha=0.6, label='Daily panic coverage (%)')
ax.plot(xs, ma, lw=2.0, color='#C44E52', label='3-day moving average')
for d, lab in [('2020-06-11', 'Xinfadi report'), ('2020-06-13', 'Wartime mechanism'),
               ('2020-06-16', 'Level-2 response')]:
    ax.axvline(datetime.strptime(d, '%Y-%m-%d'), color='gray', ls='--', lw=0.9, alpha=0.75)
    ax.text(datetime.strptime(d, '%Y-%m-%d'), ys.max() * 0.97, lab, rotation=90,
            fontsize=7, va='top', ha='right', color='#5F5E5A')
ax.axvspan(datetime(2020, 5, 9), datetime(2020, 6, 8), color='#85B7EB', alpha=0.12)
ax.text(datetime(2020, 5, 13), 0.15, 'baseline 30d', fontsize=8, color='#185FA5')
ax.set_ylabel('Panic coverage (%)', fontsize=10)
ax.set_title('Graded panic around the Beijing Xinfadi outbreak (zero retraining, full corpus)', fontsize=10.5)
ax.legend(fontsize=8, frameon=False, loc='upper right')
ax.grid(alpha=0.2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
fig.tight_layout()
fig.savefig(f'{BASE}/fig_xinfadi_daily.png', dpi=300)
plt.close(fig)

jan_series = [dict(date=d, coverage=round(100 * n / c, 4) if c else None)
              for d, (c, n) in sorted(daily['jan'].items())]
j1 = [r for r in jan_series if '2020-01-08' <= r['date'] <= '2020-01-31' and r['coverage'] is not None]
j6 = [r for r in series if '2020-06-01' <= r['date'] <= '2020-06-30' and r['coverage'] is not None]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, s_, title, marks in [
    (axes[0], j1, 'January — Wuhan lockdown (calibration)', ['2020-01-20', '2020-01-23']),
    (axes[1], j6, 'June — Beijing Xinfadi (replication)', ['2020-06-11', '2020-06-13', '2020-06-16'])]:
    ax.plot([datetime.strptime(r['date'], '%Y-%m-%d') for r in s_],
            [r['coverage'] for r in s_], marker='o', ms=3, lw=1.2, color='#4C72B0')
    for m in marks:
        ax.axvline(datetime.strptime(m, '%Y-%m-%d'), color='#C44E52', ls='--', lw=0.9, alpha=0.8)
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
axes[0].set_ylabel('Panic coverage (%)', fontsize=10)
fig.suptitle('Signature comparison (shared y-axis — note the level difference)', fontsize=10)
fig.tight_layout()
fig.savefig(f'{BASE}/fig_signature_compare.png', dpi=300)
plt.close(fig)

# ---------------- 落盘 ----------------
import pandas as pd
pd.DataFrame(series).to_csv(f'{BASE}/tab_daily_series.csv', index=False)
pd.DataFrame(rows).to_csv(f'{BASE}/tab_thresholds.csv', index=False)
pd.DataFrame(sel_jan).to_csv(f'{BASE}/tab_gmm_selection_january.csv', index=False)
pd.DataFrame(sel_jun).to_csv(f'{BASE}/tab_gmm_selection_june.csv', index=False)
pd.DataFrame(bounds_drift).to_csv(f'{BASE}/tab_natural_boundaries.csv', index=False)
pd.DataFrame(mean_drift).to_csv(f'{BASE}/tab_component_means.csv', index=False)

summary.update(dict(
    gmm_K_january=K_jan, gmm_K_june=K_jun,
    january_component_means=[round(m, 4) for m in means_jan],
    june_component_means=[round(m, 4) for m in means_jun],
    january_natural_boundaries=[round(b, 4) for b in bounds_jan],
    june_natural_boundaries=[None if np.isnan(b) else round(b, 4) for b in bounds_jun],
    max_component_mean_deviation=round(max_dev, 4),
    tier_shares_under_three_rules=rows,
    natural_boundary_drift=bounds_drift,
    n_positive_january=int(len(pos['jan'])), n_positive_june=int(len(pos['jun'])),
))
with open(f'{BASE}/results_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f'\n完成 → {BASE}')
