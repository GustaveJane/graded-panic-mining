# -*- coding: utf-8 -*-
"""重画 Figure 3：RDD 主图（MSE-optimal 带宽口径）

与论文 tex 的 tab:rddbw / tab:inf 完全对齐：
  (a) Panic coverage      bw = ±15 days  (h* = 15.45, mserd), n_days = 31
  (b) Overall panic score bw = ±11 days  (h* = 11.41, mserd), n_days = 23
规格：局部线性 + 三角核 + HAC(3)，断点 2020-01-23
"""
import os
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import WLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = r"D:/xinguanbiaozhu/experiments/results/rdd/daily_unified_metrics.csv"
OUT_DIR = r"D:/xinguanbiaozhu/RDD口径统一/重画产物"
LOCKDOWN = pd.Timestamp("2020-01-23")

COLOR_BLUE = "#4C72B0"
COLOR_ORANGE = "#DD8452"
COLOR_LIGHT = "#BFBFBF"


def local_linear_core(running, y, w, hac_lags=3):
    """WLS + 三角核 + HAC，与 统一论文出图分析_v2.py 完全一致。"""
    running = np.asarray(running, dtype=float)
    treated = (running >= 0).astype(float)
    X = pd.DataFrame({
        "const": 1.0,
        "treated": treated,
        "running": running,
        "running_x_treated": running * treated,
    })
    model = WLS(np.asarray(y, dtype=float), X, weights=np.asarray(w, dtype=float))
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def fit_rdd(daily, outcome, bw):
    sub = daily.loc[daily["days_from_lockdown"].abs() <= bw].copy()
    sub["running"] = (sub["date"] - LOCKDOWN).dt.days
    sub["kernel_weight"] = np.maximum(1 - np.abs(sub["running"]) / bw, 0.0)
    f = local_linear_core(sub["running"].values, sub[outcome].values,
                          sub["kernel_weight"].values)
    eff, se = float(f.params["treated"]), float(f.bse["treated"])
    res = dict(effect=eff, se=se, p=float(f.pvalues["treated"]),
               lo=eff - 1.96 * se, hi=eff + 1.96 * se, n=int(len(sub)),
               pre=float(sub.loc[sub["running"] < 0, outcome].mean()),
               post=float(sub.loc[sub["running"] >= 0, outcome].mean()))
    return res, sub, f


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    daily = pd.read_csv(DATA, encoding="utf-8-sig")
    daily["date"] = pd.to_datetime(daily["date"])

    specs = [
        ("panic_rate", "Panic coverage", 15),
        ("overall_panic_score", "Overall panic score", 11),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=False)
    out_rows = []

    for ax, (outcome, title, bw) in zip(axes, specs):
        res, sub, f = fit_rdd(daily, outcome, bw)
        out_rows.append(dict(outcome=outcome, bw=bw, **res))
        print(f"[{title}] bw=±{bw}  tau={res['effect']:+.4f}  SE={res['se']:.4f}  "
              f"CI[{res['lo']:+.4f},{res['hi']:+.4f}]  N={res['n']}  p={res['p']:.2e}")

        pre = sub[sub["running"] < 0]
        post = sub[sub["running"] >= 0]

        ax.scatter(pre["running"], pre[outcome], s=35, color=COLOR_BLUE,
                   alpha=0.55, label="Pre-lockdown", zorder=3)
        ax.scatter(post["running"], post[outcome], s=35, color=COLOR_ORANGE,
                   alpha=0.55, label="Post-lockdown", zorder=3)

        x_left = np.linspace(-bw, 0, 120)
        x_right = np.linspace(0, bw, 120)
        left_pred = f.params["const"] + f.params["running"] * x_left
        right_pred = (f.params["const"] + f.params["treated"]
                      + (f.params["running"] + f.params["running_x_treated"]) * x_right)
        ax.plot(x_left, left_pred, color=COLOR_BLUE, linewidth=2.2)
        ax.plot(x_right, right_pred, color=COLOR_ORANGE, linewidth=2.2)

        ax.axvline(0, color="black", linestyle="--", linewidth=1.8,
                   label="Lockdown (Jan 23)")
        ax.axhline(res["pre"], color=COLOR_BLUE, linestyle=":", alpha=0.25)

        ax.set_xlim(-bw, bw)
        ax.set_xticks(np.arange(-(bw // 5) * 5, (bw // 5) * 5 + 1, 5))
        ax.set_xlabel("Days relative to Jan 23", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title} | bw=±{bw} days", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", frameon=False, fontsize=9)

        txt = (rf"$\hat{{\tau}}$={res['effect']:+.4f}" + "\n"
               f"SE={res['se']:.4f}\n"
               f"95% CI [{res['lo']:+.4f}, {res['hi']:+.4f}]\n"
               f"N_days={res['n']}")
        ax.text(0.03, 0.96, txt, transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=COLOR_LIGHT, alpha=0.95))

    plt.suptitle("Figure 3. Local Linear RDD at the Wuhan Lockdown Cutoff\n"
                 "(MSE-optimal bandwidths: ±15 days for panic coverage, ±11 days for overall panic score)",
                 fontsize=13.5, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "image6.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")

    pd.DataFrame(out_rows).to_csv(os.path.join(OUT_DIR, "rerun_fig3_mse_optimal.csv"),
                                  index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
