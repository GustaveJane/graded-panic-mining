# -*- coding: utf-8 -*-
"""标签噪声敏感性实验（回应审稿意见：topic reassignment 64.37% 对样本内比较的影响）

思路：
  1. 用论文 Table 8（tab:12）的层内类别占比 + 各层样本量，构造与论文一致的列联表
  2. 以概率 p 随机重分配 L1 标签（标签与真实类别独立），重算 Cramér's V 与 A 类梯度
  3. 扫描 p，定位「梯度失效」的临界噪声水平
  4. 与论文报告的 35.6% 标签不一致率对照

说明：IPW 权重在层内为常数，故「层内占比」加权前后不变，可直接用 tab:12 的百分比。
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

OUT = r'D:/xinguanbiaozhu/label_noise_sensitivity'
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261022)

# ---------- Table 8 (tab:12) 的层内占比（%） ----------
CATS = ['A', 'B', 'C', 'D', 'N']
SHARES = np.array([
    [24.62, 51.43, 1.65, 19.56, 2.73],   # Non-panic
    [67.04, 16.98, 3.21, 12.37, 0.40],   # Low
    [71.30, 15.43, 3.68, 9.27, 0.32],    # Medium
    [75.81, 16.03, 1.56, 6.46, 0.14],    # High
]) / 100.0
# 各层样本量（由 doc_taxonomy_mapping.csv 按 t1/t2 分层实测）
N_TIER = np.array([79132, 69818, 64242, 65779])

P_OBS = 0.3563   # = 1 - 0.6437，论文报告的 topic 重分配不一致率
TARGET_V = 0.268  # 论文报告的 Cramér's V


def cramers_v(table):
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def build_obs():
    obs = np.round(SHARES * N_TIER[:, None]).astype(int)
    # 修正取整误差，使每层合计等于 n
    for i in range(4):
        diff = N_TIER[i] - obs[i].sum()
        obs[i, np.argmax(obs[i])] += diff
    return obs


OBS = build_obs()
V0 = cramers_v(OBS)
A0 = 100 * OBS[:, 0] / OBS.sum(1)

print('=== 口径校验 ===')
print(f'构造列联表合计 {OBS.sum():,}（论文 n=278,971）')
print(f'复算 Cramér\'s V = {V0:.4f}  （论文报告 0.268）')
print(f'A 类占比 Non={A0[0]:.2f}%  High={A0[3]:.2f}%  梯度={A0[3]-A0[0]:.2f}pp  （论文 24.62 → 75.81）')
print()

# ---------- 文档级展开 ----------
tier_flat = np.repeat(np.arange(4), N_TIER)
cat_flat = np.concatenate([
    RNG.choice(5, size=N_TIER[i], p=SHARES[i] / SHARES[i].sum()) for i in range(4)
])
MARG = np.bincount(cat_flat, minlength=5) / len(cat_flat)
print('全局边际分布:', dict(zip(CATS, np.round(MARG * 100, 2))))
print()


def simulate_noise(cat_base, p, rng):
    """以概率 p 将标签替换为边际分布抽样（与真实类别独立）。"""
    out = cat_base.copy()
    mask = rng.random(len(out)) < p
    k = int(mask.sum())
    if k:
        out[mask] = rng.choice(5, size=k, p=MARG)
    return out


def metrics(cat_arr):
    table = np.zeros((4, 5), dtype=int)
    for t in range(4):
        sel = tier_flat == t
        table[t] = np.bincount(cat_arr[sel], minlength=5)
    v = cramers_v(table)
    a = 100 * table[:, 0] / table.sum(1)
    return v, a, table


P_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.3563, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
REPS = 100

rows = []
print('=== 噪声注入模拟（每档 100 次重复）===')
print('  p       V(mean±sd)        A_Non    A_High   梯度(pp)')
for p in P_GRID:
    vs, grads, a_nons, a_highs, a_lows, a_meds = [], [], [], [], [], []
    for _ in range(REPS):
        c = cat_flat if p == 0 else simulate_noise(cat_flat, p, RNG)
        v, a, _ = metrics(c)
        vs.append(v); grads.append(a[3] - a[0]); a_nons.append(a[0]); a_highs.append(a[3])
        a_lows.append(a[1]); a_meds.append(a[2])
    rows.append(dict(p=p, V=float(np.mean(vs)), V_sd=float(np.std(vs)),
                     A_non=float(np.mean(a_nons)), A_low=float(np.mean(a_lows)),
                     A_med=float(np.mean(a_meds)), A_high=float(np.mean(a_highs)),
                     gradient_pp=float(np.mean(grads)), gradient_sd=float(np.std(grads)),
                     monotone=bool(np.mean(a_nons) < np.mean(a_lows) < np.mean(a_meds) < np.mean(a_highs))))
    print(f'  {p:.4f}  {np.mean(vs):.4f}±{np.std(vs):.4f}   {np.mean(a_nons):6.2f}  {np.mean(a_lows):6.2f}  {np.mean(a_meds):6.2f}  {np.mean(a_highs):6.2f}   {np.mean(grads):6.2f}   {"单调" if np.mean(a_nons) < np.mean(a_lows) < np.mean(a_meds) < np.mean(a_highs) else "非单调 ✗"}')

tab = pd.DataFrame(rows)
tab.to_csv(f'{OUT}/tab_label_noise_sensitivity.csv', index=False, encoding='utf-8-sig')

# ---------- 临界点 ----------
def interp(target, ys):
    xs = tab['p'].to_numpy(); ys = np.asarray(ys)
    for i in range(len(xs) - 1):
        if (ys[i] - target) * (ys[i + 1] - target) <= 0 and ys[i] != ys[i + 1]:
            return float(xs[i] + (xs[i + 1] - xs[i]) * (ys[i] - target) / (ys[i] - ys[i + 1]))
    return None


p_v_neg = interp(0.10, tab['V'].to_numpy())      # V 降到 0.10（可忽略关联）
p_g_half = interp(tab['gradient_pp'].iloc[0] / 2, tab['gradient_pp'].to_numpy())  # 梯度减半

row_obs = tab[np.isclose(tab['p'], P_OBS)].iloc[0]
V_at_obs = float(row_obs['V'])
G_at_obs = float(row_obs['gradient_pp'])
V_true_est = V0 / (1 - P_OBS)

summary = dict(
    n_total=int(OBS.sum()), cramers_v_recomputed=round(V0, 4),
    A_non_pct=round(float(A0[0]), 2), A_high_pct=round(float(A0[3]), 2),
    gradient_pp=round(float(A0[3] - A0[0]), 2),
    p_observed_inconsistency=P_OBS,
    V_at_p_observed=round(V_at_obs, 4),
    gradient_at_p_observed_pp=round(G_at_obs, 2),
    gradient_retention_pct=round(100 * G_at_obs / (A0[3] - A0[0]), 1),
    V_attenuation_estimate=round(float(V_true_est), 4),
    p_critical_V_below_0_10=p_v_neg,
    p_critical_gradient_halved=p_g_half,
)
with open(f'{OUT}/results_label_noise.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print()
print('=== 关键结论 ===')
print(json.dumps(summary, ensure_ascii=False, indent=2))

# ---------- 出图 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

ax = axes[0]
ax.plot(tab['p'], tab['V'], marker='o', ms=3, lw=1.6, color='#4C72B0')
ax.fill_between(tab['p'], tab['V'] - tab['V_sd'], tab['V'] + tab['V_sd'], color='#4C72B0', alpha=0.18)
ax.axvline(P_OBS, color='#C44E52', ls='--', lw=1.0)
ax.text(P_OBS + 0.02, tab['V'].max() * 0.9, '64.37% consistency\n(35.6% noise)', fontsize=8, color='#A32D2D')
ax.axhline(0.10, color='gray', ls=':', lw=0.9)
ax.text(0.42, 0.105, 'negligible association (V=0.10)', fontsize=7.5, color='#5F5E5A')
ax.set_xlabel('Proportion of L1 labels randomized', fontsize=10)
ax.set_ylabel("Cramér's V", fontsize=10)
ax.set_title('Association strength vs. label noise', fontsize=10)
ax.grid(alpha=0.2)

ax = axes[1]
ax.plot(tab['p'], tab['A_non'], marker='o', ms=3, lw=1.5, label='Non-panic tier')
ax.plot(tab['p'], tab['A_high'], marker='s', ms=3, lw=1.5, label='High-panic tier')
ax.axvline(P_OBS, color='#C44E52', ls='--', lw=1.0)
ax.set_xlabel('Proportion of L1 labels randomized', fontsize=10)
ax.set_ylabel('Share of class A (%)', fontsize=10)
ax.set_title('Class-A share by tier vs. label noise', fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(f'{OUT}/fig_label_noise_sensitivity.png', dpi=300)
plt.close(fig)
print(f'\n产物 → {OUT}')
