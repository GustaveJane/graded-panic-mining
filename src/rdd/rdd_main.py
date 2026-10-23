#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weibo COVID-19 unified analysis for paper figures.

This script unifies the statistics and plotting conventions used in:
- analysis.py: long-run trends, event-window comparisons, single-outcome RDD
- dual_rdd.py: dual-outcome RDD, bandwidth sensitivity, cutoff-date sensitivity

Key unifications:
1) Daily outcomes are constructed from the same data pipeline.
2) Panic coverage and panic intensity are defined consistently.
3) Additional overall panic score is added to reduce selection bias.
4) Event-window plots focus on the late-January policy-information window.
5) RDD results are estimated with the same local-linear triangular-kernel design.
6) Standard errors are reported with HAC robustness for daily time series.
7) Output figures are publication-oriented and use a consistent visual style.
"""

from __future__ import annotations

import os
import csv
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats import diagnostic as sm_diagnostic

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
# Prefer local project path; fall back to the original autodl path if present.
_LOCAL_BASE = r"D:\xinguanbiaozhu"
_SERVER_BASE = "/root/autodl-tmp"
BASE_DIR = _LOCAL_BASE if os.path.isdir(_LOCAL_BASE) else _SERVER_BASE
OUTPUT_DIR = os.path.join(BASE_DIR, "paper_figures_unified")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Local machine currently has only partial labeled data:
# - result/temp19.csv: 2019-12 with covid_related + panic_pred (no panic_prob)
# - labeled_results/covid_1-6_*: 2020-01~06 random sample with panic_related
# Original server files (temp19_1 / temp1..temp6 with panic_prob) are used when available.
DATA_FILES = [
    # Server-style full inference outputs (used when present).
    # NOTE: temp19_1.csv (panic_col=panic_pred) is the legacy December file and is
    # no longer on the server; temp19.csv below carries the same month with the
    # current column names. Keep both so either layout resolves.
    {"path": os.path.join(BASE_DIR, "temp19_1.csv"), "panic_col": "panic_pred"},
    {"path": os.path.join(BASE_DIR, "temp19.csv"), "panic_col": "panic"},
    {"path": os.path.join(BASE_DIR, "temp1.csv"), "panic_col": "panic"},
    {"path": os.path.join(BASE_DIR, "temp2.csv"), "panic_col": "panic"},
    {"path": os.path.join(BASE_DIR, "temp3.csv"), "panic_col": "panic"},
    {"path": os.path.join(BASE_DIR, "temp4.csv"), "panic_col": "panic"},
    {"path": os.path.join(BASE_DIR, "temp5.csv"), "panic_col": "panic"},
    {"path": os.path.join(BASE_DIR, "temp6.csv"), "panic_col": "panic"},
    # Local fallbacks
    {
        "path": os.path.join(BASE_DIR, "result", "temp19.csv"),
        "panic_col": "panic_pred",
        "encoding": "gb18030",
    },
    {
        "path": os.path.join(
            BASE_DIR,
            "labeled_results",
            "covid_1-6_random5000_labeled_20260416_184448.csv",
        ),
        "panic_col": "panic_related",
        "encoding": "utf-8",
        "assume_covid": True,
    },
]

START_DATE = pd.Timestamp("2019-12-01")
END_DATE = pd.Timestamp("2020-06-30")
LOCKDOWN = pd.Timestamp("2020-01-23")
HUMAN_TO_HUMAN = pd.Timestamp("2020-01-20")

# Main events for descriptive plots
EVENTS = {
    "Human-to-human transmission confirmation": HUMAN_TO_HUMAN,
    "Wuhan lockdown": LOCKDOWN,
    "National mourning day": pd.Timestamp("2020-04-04"),
    "Wuhan reopening": pd.Timestamp("2020-04-08"),
    "Hubei reopening notice": pd.Timestamp("2020-03-24"),
}

# Style
COLOR_BLUE = "#3A506B"
COLOR_ORANGE = "#C97B63"
COLOR_RED = "#C0392B"
COLOR_GRAY = "#7F8C8D"
COLOR_GREEN = "#2E8B57"
COLOR_LIGHT = "#D7DBDD"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# Helpers
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_read_csv(
    path: str,
    usecols: List[str] | None = None,
    encoding: str | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read CSV with encoding fallback for local Windows files."""
    encodings = []
    if encoding:
        encodings.append(encoding)
    encodings.extend(["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"])

    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                usecols=usecols,
                encoding=enc,
                low_memory=False,
                nrows=nrows,
            )
        except Exception as exc:  # noqa: BLE001 - try next encoding / schema
            last_err = exc
            continue
    raise RuntimeError(f"Failed to read {path}: {last_err}")


def load_and_prepare() -> pd.DataFrame:
    frames = []
    for cfg in DATA_FILES:
        path = cfg["path"]
        panic_col = cfg["panic_col"]
        if not os.path.exists(path):
            print(f"[skip] missing: {path}")
            continue

        # Discover usable columns first (local files vary).
        header_only = safe_read_csv(path, encoding=cfg.get("encoding"), nrows=0)
        available = set(header_only.columns)
        if "created_at" not in available:
            print(f"[skip] no created_at: {path}")
            continue
        if panic_col not in available:
            print(f"[skip] no {panic_col}: {path}")
            continue

        want = ["created_at", panic_col]
        if "covid_related" in available:
            want.append("covid_related")
        if "panic_prob" in available:
            want.append("panic_prob")
        # Keep the post id when present: it is the only safe de-duplication key.
        id_col = next((c for c in ("_id", "id", "mid", "weibo_id") if c in available), None)
        if id_col is not None:
            want.append(id_col)

        df = safe_read_csv(path, usecols=want, encoding=cfg.get("encoding"))
        df = df.rename(columns={panic_col: "panic"})

        if "covid_related" not in df.columns:
            if cfg.get("assume_covid", False):
                # Sample is drawn from COVID-related posts.
                df["covid_related"] = 1
                print(f"[note] assume covid_related=1 for {os.path.basename(path)}")
            else:
                print(f"[skip] no covid_related: {path}")
                continue

        if "panic_prob" not in df.columns:
            # Local labeled data only has hard 0/1 labels.
            # Use binary labels as a probability proxy so the unified pipeline still runs.
            df["panic_prob"] = pd.to_numeric(df["panic"], errors="coerce").fillna(0).astype(float)
            print(f"[note] synthesize panic_prob from {panic_col} for {os.path.basename(path)}")

        frames.append(df)
        print(f"[load] {os.path.basename(path)} rows={len(df):,}")

    if not frames:
        raise FileNotFoundError("No input files were found.")

    df = pd.concat(frames, ignore_index=True)

    # Normalize timestamps and keep the target period.
    dt = pd.to_datetime(df["created_at"], format="mixed", errors="coerce")
    df = df.loc[dt.notna()].copy()
    df["created_at"] = dt.loc[dt.notna()]
    df["date"] = df["created_at"].dt.floor("D")
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()

    # Basic cleaning
    df["covid_related"] = pd.to_numeric(df["covid_related"], errors="coerce").fillna(0).astype(int)
    df["panic"] = pd.to_numeric(df["panic"], errors="coerce").fillna(0).astype(int)
    df["panic_prob"] = pd.to_numeric(df["panic_prob"], errors="coerce")
    # Final fallback if some rows still lack probabilities.
    missing_prob = df["panic_prob"].isna()
    if missing_prob.any():
        df.loc[missing_prob, "panic_prob"] = df.loc[missing_prob, "panic"].astype(float)
        print(f"[note] filled {int(missing_prob.sum()):,} missing panic_prob from binary panic")

    # De-duplicate across the split input files.
    #
    # IMPORTANT: do NOT de-duplicate on (created_at, covid_related, panic, panic_prob).
    # Those four columns are shared by thousands of distinct posts, so keying on them
    # collapses the corpus to a fraction of its size (in this project it removed ~75% of
    # the rows and shifted the RDD estimate from tau=-0.0289 to tau=-0.0369).
    # Use the post id when the input carries one; otherwise only drop fully identical rows.
    id_col = next((c for c in ("_id", "id", "mid", "weibo_id") if c in df.columns), None)
    if id_col is not None:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col]).copy()
        print(f"[dedup] by {id_col}: {before:,} -> {len(df):,} rows")
    else:
        before = len(df)
        df = df.drop_duplicates().copy()
        print(f"[dedup] no id column, dropped fully identical rows: {before:,} -> {len(df):,}")

    print(f"[data] total rows={len(df):,} | date range={df['date'].min().date()} ~ {df['date'].max().date()}")
    print(
        "[data] covid_related=1: "
        f"{int((df['covid_related']==1).sum()):,} | panic=1: {int((df['panic']==1).sum()):,}"
    )
    return df


def build_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")

    # Daily base counts from all posts.
    daily = df.groupby("date").agg(
        total_posts=("covid_related", "count"),
        covid_posts=("covid_related", "sum"),
        panic_posts=("panic", "sum"),
        overall_panic_score=("panic_prob", "mean"),
    ).reindex(all_dates, fill_value=0)

    # Coverage among covid-related posts.
    daily["panic_rate"] = np.where(daily["covid_posts"] > 0, daily["panic_posts"] / daily["covid_posts"], 0.0)

    # Conditional intensity among panic-labeled posts.
    cond = (
        df.loc[df["panic"] == 1]
        .groupby("date")["panic_prob"]
        .mean()
        .reindex(all_dates)
    )
    daily["conditional_panic_score"] = cond
    daily["conditional_panic_score"] = daily["conditional_panic_score"].fillna(0.0)

    daily = daily.reset_index().rename(columns={"index": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["days_from_lockdown"] = (daily["date"] - LOCKDOWN).dt.days
    daily["days_from_human_to_human"] = (daily["date"] - HUMAN_TO_HUMAN).dt.days

    # Smoothed series for display.
    for col in ["panic_rate", "overall_panic_score", "conditional_panic_score", "covid_posts", "panic_posts"]:
        daily[f"{col}_ma7"] = daily[col].rolling(window=7, center=True, min_periods=1).mean()

    return daily


def format_pct(x: float) -> str:
    return f"{x*100:.1f}%"


# =========================
# Plot 1: Long-run trends
# =========================
def plot_long_run_trends(daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={"hspace": 0.08})

    # Panel A: volume
    ax = axes[0]
    ax.plot(daily["date"], np.log10(daily["covid_posts"].clip(lower=1)), color=COLOR_BLUE, linewidth=2.0)
    ax.set_ylabel("log10(COVID posts + 1)", fontsize=11)
    ax.set_title("A. Daily volume of COVID-related discussion", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Panel B: panic coverage
    ax = axes[1]
    ax.plot(daily["date"], daily["panic_rate"], color=COLOR_RED, linewidth=1.0, alpha=0.35)
    ax.plot(daily["date"], daily["panic_rate_ma7"], color=COLOR_RED, linewidth=2.5, label="7-day moving average")
    ax.set_ylabel("Panic coverage", fontsize=11)
    ax.set_title("B. Panic coverage among COVID-related posts", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.2)

    # Panel C: panic intensity
    ax = axes[2]
    ax.plot(daily["date"], daily["conditional_panic_score"], color=COLOR_ORANGE, linewidth=1.0, alpha=0.35)
    ax.plot(daily["date"], daily["conditional_panic_score_ma7"], color=COLOR_ORANGE, linewidth=2.5, label="7-day moving average")
    ax.set_ylabel("Conditional panic score", fontsize=11)
    ax.set_title("C. Mean panic probability among panic-labeled posts", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.2)

    for ax in axes:
        for name, dt in EVENTS.items():
            if START_DATE <= dt <= END_DATE:
                ax.axvline(dt, color=COLOR_GRAY, linestyle="--", linewidth=1, alpha=0.35)
        ax.set_xlim(START_DATE, END_DATE)

    month_ticks = pd.date_range("2019-12-01", "2020-06-01", freq="MS")
    axes[-1].set_xticks(month_ticks)
    axes[-1].set_xticklabels([d.strftime("%b") for d in month_ticks])
    axes[-1].set_xlabel("Date", fontsize=11)

    plt.suptitle("Figure 1. Temporal Evolution of COVID-Related Discussion and Panic", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig1_long_run_trends.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


# =========================
# Plot 2: Policy-information window
# =========================
def plot_policy_window(daily: pd.DataFrame) -> None:
    window_start = pd.Timestamp("2020-01-10")
    window_end = pd.Timestamp("2020-02-06")
    sub = daily[(daily["date"] >= window_start) & (daily["date"] <= window_end)].copy()

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"hspace": 0.1})

    for ax, y, ylab, title, color in [
        (axes[0], "panic_rate", "Panic coverage", "Panic coverage around the policy-information window", COLOR_RED),
        (axes[1], "conditional_panic_score", "Conditional panic score", "Panic intensity around the policy-information window", COLOR_ORANGE),
    ]:
        ax.plot(sub["date"], sub[y], color=color, linewidth=1.0, alpha=0.35)
        ax.plot(sub["date"], sub[f"{y}_ma7"], color=color, linewidth=2.5)
        ax.axvline(HUMAN_TO_HUMAN, color="black", linestyle="--", linewidth=1.5)
        ax.axvline(LOCKDOWN, color="black", linestyle="--", linewidth=1.5)
        ax.axvspan(HUMAN_TO_HUMAN, LOCKDOWN, color=COLOR_LIGHT, alpha=0.45)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(window_start, window_end)

    axes[1].set_xlabel("Date", fontsize=11)
    plt.suptitle("Figure 2. Panic Dynamics Around the Human-to-Human Transmission Announcement and Wuhan Lockdown", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig2_policy_window.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


# =========================
# RDD utilities
# =========================
@dataclass
class RDDResult:
    outcome: str
    bandwidth: int
    n_days: int
    effect: float
    se: float
    pvalue: float
    ci_low: float
    ci_high: float
    pre_mean: float
    post_mean: float


def local_linear_core(
    running: np.ndarray,
    y: np.ndarray,
    kernel_weight: np.ndarray,
    use_hac: bool = True,
    hac_lags: int = 3,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Shared WLS + (HAC | HC1) core for every RDD specification.

    Identical for the standard, donut and hourly variants: same local-linear
    parameterisation with side-specific slopes, same triangular kernel weights,
    same covariance estimator. Keeping this in one place is what makes the
    donut and hourly results comparable with the daily main table.
    """
    running = np.asarray(running, dtype=float)
    treated = (running >= 0).astype(float)
    X = pd.DataFrame({
        "const": 1.0,
        "treated": treated,
        "running": running,
        "running_x_treated": running * treated,
    })
    model = sm.WLS(np.asarray(y, dtype=float), X,
                   weights=np.asarray(kernel_weight, dtype=float))
    if use_hac:
        return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model.fit(cov_type="HC1")


def fit_local_linear_rdd(
    daily: pd.DataFrame,
    outcome: str,
    bandwidth: int,
    cutoff: pd.Timestamp = LOCKDOWN,
    use_hac: bool = True,
    hac_lags: int = 3,
    exclude_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    min_covid_posts: int = 0,
) -> Tuple[RDDResult, pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    """Local-linear triangular-kernel RDD on the daily series.

    Parameters
    ----------
    daily : DataFrame with a ``date`` column and a ``days_from_lockdown`` column.
    outcome : column to use as the dependent variable.
    bandwidth : symmetric bandwidth in days.
    cutoff : discontinuity date.
    use_hac : True -> HAC (Newey-West); False -> HC1.
    hac_lags : maxlags for the HAC estimator.
    exclude_window : optional (start, end) half-open interval to drop from the
        support, i.e. the "donut hole". Used for the 2020-01-20~01-22 spike.
    min_covid_posts : drop days with fewer than this many COVID-related posts.
        Days with ``covid_posts == 0`` receive ``panic_rate = 0`` by definition
        rather than by measurement, which is a structural artefact; 29 such days
        exist, all in December 2019, and six of them fall inside a +-30-day
        window. Set to 1 to drop them.
    """
    sub = daily.loc[daily["days_from_lockdown"].abs() <= bandwidth].copy()
    sub["running"] = (sub["date"] - cutoff).dt.days
    if exclude_window is not None:
        lo, hi = exclude_window
        sub = sub.loc[~((sub["date"] >= lo) & (sub["date"] < hi))].copy()
    if min_covid_posts > 0:
        sub = sub.loc[sub["covid_posts"] >= min_covid_posts].copy()
    sub["treated"] = (sub["running"] >= 0).astype(int)
    sub["kernel_weight"] = np.maximum(1 - np.abs(sub["running"]) / bandwidth, 0.0)

    fit = local_linear_core(sub["running"].values, sub[outcome].values,
                           sub["kernel_weight"].values, use_hac, hac_lags)

    eff = float(fit.params["treated"])
    se = float(fit.bse["treated"])
    pvalue = float(fit.pvalues["treated"])
    ci_low = eff - 1.96 * se
    ci_high = eff + 1.96 * se

    res = RDDResult(
        outcome=outcome,
        bandwidth=bandwidth,
        n_days=len(sub),
        effect=eff,
        se=se,
        pvalue=pvalue,
        ci_low=ci_low,
        ci_high=ci_high,
        pre_mean=float(sub.loc[sub["running"] < 0, outcome].mean()),
        post_mean=float(sub.loc[sub["running"] >= 0, outcome].mean()),
    )
    return res, sub, fit


def rdd_result_to_dict(r: RDDResult) -> Dict[str, float]:
    return {
        "outcome": r.outcome,
        "bandwidth": r.bandwidth,
        "n_days": r.n_days,
        "effect": r.effect,
        "se": r.se,
        "pvalue": r.pvalue,
        "ci_low": r.ci_low,
        "ci_high": r.ci_high,
        "pre_mean": r.pre_mean,
        "post_mean": r.post_mean,
    }


# =========================
# Plot 3: RDD main figure
# =========================
def plot_rdd_main(daily: pd.DataFrame, bw: int = 14) -> Tuple[Dict[str, float], Dict[str, float]]:
    outcomes = [
        ("panic_rate", "Panic coverage", COLOR_RED),
        ("overall_panic_score", "Overall panic score", COLOR_BLUE),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=False)
    results = {}

    for ax, (outcome, title, color) in zip(axes, outcomes):
        res, sub, fit = fit_local_linear_rdd(daily, outcome, bandwidth=bw, use_hac=True, hac_lags=3)
        results[outcome] = rdd_result_to_dict(res)

        pre = sub[sub["running"] < 0]
        post = sub[sub["running"] >= 0]

        ax.scatter(pre["running"], pre[outcome], s=35, color=COLOR_BLUE, alpha=0.55, label="Pre-lockdown", zorder=3)
        ax.scatter(post["running"], post[outcome], s=35, color=COLOR_ORANGE, alpha=0.55, label="Post-lockdown", zorder=3)

        x_left = np.linspace(-bw, 0, 120)
        x_right = np.linspace(0, bw, 120)

        left_pred = fit.params["const"] + fit.params["running"] * x_left
        right_pred = (
            fit.params["const"]
            + fit.params["treated"]
            + (fit.params["running"] + fit.params["running_x_treated"]) * x_right
        )
        ax.plot(x_left, left_pred, color=COLOR_BLUE, linewidth=2.2)
        ax.plot(x_right, right_pred, color=COLOR_ORANGE, linewidth=2.2)

        # Simple pointwise bands from coefficient uncertainty are omitted here to keep the figure clean.
        ax.axvline(0, color="black", linestyle="--", linewidth=1.8, label="Lockdown (Jan 23)")
        ax.axhline(res.pre_mean, color=COLOR_BLUE, linestyle=":", alpha=0.25)

        ax.set_xlim(-bw, bw)
        ax.set_xlabel("Days relative to Jan 23", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title} | bw=±{bw} days", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", frameon=False, fontsize=9)

        txt = (
            rf"$\hat{{\tau}}$={res.effect:+.4f}" + "\n"
            f"SE={res.se:.4f}\n"
            f"95% CI [{res.ci_low:+.4f}, {res.ci_high:+.4f}]\n"
            f"N_days={res.n_days}"
        )
        ax.text(
            0.03,
            0.96,
            txt,
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_LIGHT, alpha=0.95),
        )

    plt.suptitle("Figure 3. Local Linear RDD at the Wuhan Lockdown Cutoff", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"fig3_rdd_main_bw{bw}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")
    return results["panic_rate"], results["overall_panic_score"]


# =========================
# Plot 4: Bandwidth sensitivity
# =========================
def plot_bandwidth_sensitivity(daily: pd.DataFrame, bandwidths: List[int] = [7, 14, 21, 30]) -> pd.DataFrame:
    rows = []
    for outcome in ["panic_rate", "overall_panic_score"]:
        for bw in bandwidths:
            res, _, _ = fit_local_linear_rdd(daily, outcome, bandwidth=bw, use_hac=True, hac_lags=3)
            rows.append(rdd_result_to_dict(res))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(OUTPUT_DIR, "rdd_bandwidth_results.csv"), index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=False)
    titles = {
        "panic_rate": "Panic coverage",
        "overall_panic_score": "Overall panic score",
    }
    colors = {
        "panic_rate": COLOR_RED,
        "overall_panic_score": COLOR_BLUE,
    }

    for ax, outcome in zip(axes, ["panic_rate", "overall_panic_score"]):
        sub = out_df[out_df["outcome"] == outcome].sort_values("bandwidth")
        x = np.arange(len(sub))
        ax.errorbar(
            x,
            sub["effect"],
            yerr=1.96 * sub["se"],
            fmt="o",
            color=colors[outcome],
            ecolor=COLOR_GRAY,
            elinewidth=1.5,
            capsize=5,
            markersize=8,
        )
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"±{b}d" for b in sub["bandwidth"]])
        ax.set_xlabel("Bandwidth", fontsize=11)
        ax.set_ylabel("RDD effect", fontsize=11)
        ax.set_title(titles[outcome], fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)
        for i, row in sub.reset_index(drop=True).iterrows():
            ax.text(i, row["effect"], f"{row['effect']:+.4f}", ha="center", va="bottom" if row["effect"] >= 0 else "top", fontsize=9)

    plt.suptitle("Figure 4. Sensitivity of RDD Estimates to Bandwidth Choice", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig4_bandwidth_sensitivity.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")
    return out_df


# =========================
# Plot 5: Cutoff-date sensitivity
# =========================
def plot_cutoff_date_sensitivity(daily: pd.DataFrame, bandwidth: int = 21) -> pd.DataFrame:
    # Dates around the lockdown. This is a cutoff-date sensitivity profile, not a classic density placebo.
    candidate_dates = pd.to_datetime([
        "2020-01-20",
        "2020-01-21",
        "2020-01-22",
        "2020-01-23",
        "2020-01-24",
        "2020-01-25",
        "2020-01-26",
    ])

    rows = []
    for outcome in ["panic_rate", "overall_panic_score"]:
        for dt in candidate_dates:
            # Re-center the running variable at each candidate cutoff.
            tmp = daily.copy()
            tmp["running"] = (tmp["date"] - dt).dt.days
            sub = tmp[tmp["running"].abs() <= bandwidth].copy()
            sub["treated"] = (sub["running"] >= 0).astype(int)
            sub["kernel_weight"] = np.maximum(1 - np.abs(sub["running"]) / bandwidth, 0.0)
            X = pd.DataFrame({
                "const": 1.0,
                "treated": sub["treated"].astype(float),
                "running": sub["running"].astype(float),
                "running_x_treated": sub["running"].astype(float) * sub["treated"].astype(float),
            })
            model = sm.WLS(sub[outcome], X, weights=sub["kernel_weight"]).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            rows.append(
                {
                    "outcome": outcome,
                    "cutoff": dt,
                    "effect": float(model.params["treated"]),
                    "se": float(model.bse["treated"]),
                    "pvalue": float(model.pvalues["treated"]),
                    "ci_low": float(model.params["treated"] - 1.96 * model.bse["treated"]),
                    "ci_high": float(model.params["treated"] + 1.96 * model.bse["treated"]),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(OUTPUT_DIR, "cutoff_date_sensitivity.csv"), index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True, gridspec_kw={"hspace": 0.08})
    titles = {
        "panic_rate": "Panic coverage",
        "overall_panic_score": "Overall panic score",
    }
    colors = {
        "panic_rate": COLOR_RED,
        "overall_panic_score": COLOR_BLUE,
    }

    for ax, outcome in zip(axes, ["panic_rate", "overall_panic_score"]):
        sub = out_df[out_df["outcome"] == outcome].sort_values("cutoff")
        x = np.arange(len(sub))
        point_colors = [COLOR_RED if d == LOCKDOWN else COLOR_GRAY for d in sub["cutoff"]]
        ax.errorbar(
            x,
            sub["effect"],
            yerr=1.96 * sub["se"],
            fmt="none",
            ecolor=COLOR_GRAY,
            elinewidth=1.2,
            capsize=4,
        )
        ax.scatter(x, sub["effect"], s=70, color=point_colors, zorder=3)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([d.strftime("%m/%d") for d in sub["cutoff"]])
        ax.set_ylabel("RDD effect", fontsize=11)
        ax.set_title(f"{titles[outcome]} | bandwidth=±{bandwidth} days", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)

        for i, row in sub.reset_index(drop=True).iterrows():
            ax.text(i, row["effect"], f"{row['effect']:+.4f}\np={row['pvalue']:.3f}", ha="center", va="bottom" if row["effect"] >= 0 else "top", fontsize=8)

    plt.suptitle("Figure 5. Cutoff-Date Sensitivity Around the Wuhan Lockdown Window", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig5_cutoff_date_sensitivity.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")
    return out_df


# =========================
# Summary tables and diagnostics
# =========================
def write_event_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, dt in [("Human-to-human transmission confirmation", HUMAN_TO_HUMAN), ("Wuhan lockdown", LOCKDOWN)]:
        pre = daily[(daily["date"] >= dt - pd.Timedelta(days=7)) & (daily["date"] < dt)]
        post = daily[(daily["date"] >= dt) & (daily["date"] <= dt + pd.Timedelta(days=7))]
        rows.append(
            {
                "event": name,
                "date": dt.strftime("%Y-%m-%d"),
                "pre_panic_rate": pre["panic_rate"].mean(),
                "post_panic_rate": post["panic_rate"].mean(),
                "change_panic_rate": post["panic_rate"].mean() - pre["panic_rate"].mean(),
                "pre_overall_panic_score": pre["overall_panic_score"].mean(),
                "post_overall_panic_score": post["overall_panic_score"].mean(),
                "change_overall_panic_score": post["overall_panic_score"].mean() - pre["overall_panic_score"].mean(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "event_window_summary.csv"), index=False, encoding="utf-8-sig")
    return out


def adf_check(series: pd.Series) -> Dict[str, float]:
    # A light-touch stationarity diagnostic for the daily outcome series.
    try:
        adf = sm.tsa.stattools.adfuller(series.dropna(), autolag="AIC")
        return {"adf_stat": float(adf[0]), "adf_pvalue": float(adf[1])}
    except Exception:
        return {"adf_stat": np.nan, "adf_pvalue": np.nan}


# =========================
# Main
# =========================
# =========================
# Block 2: donut / HAC / hourly / denominator extensions
# =========================
# The 2020-01-20~01-22 spike (human-to-human confirmation) sits three days to the
# left of the lockdown cutoff and inflates the pre-side local-linear fit. The
# donut specification drops it.
DONUT_WINDOW: Tuple[pd.Timestamp, pd.Timestamp] = (
    pd.Timestamp("2020-01-20"),
    pd.Timestamp("2020-01-23"),
)

# Hourly series used for the sub-daily discontinuity check. The lockdown
# announcement (00:00) and its enforcement (10:00) fall on the same calendar day,
# so they can only be separated at hourly resolution.
HOURLY_FILE_CANDIDATES = [
    os.path.join(BASE_DIR, "bh_full_hourly.csv"),
    "/root/autodl-tmp/bh_full_hourly.csv",
]
HOURLY_ENFORCE = pd.Timestamp("2020-01-23 10:00:00")
HOURLY_ANNOUNCE = pd.Timestamp("2020-01-23 00:00:00")
HOURLY_BANDWIDTHS = [24, 48, 72, 120, 168]
HOURLY_PLACEBO_DATES = [
    "2020-01-18", "2020-01-19", "2020-01-21", "2020-01-22", "2020-01-23",
    "2020-01-24", "2020-01-25", "2020-01-27", "2020-01-29", "2020-01-31",
]
PLACEBO_BANDWIDTHS = [48, 72, 120]


def run_donut_comparison(
    daily: pd.DataFrame,
    outcomes: List[str] = ("panic_rate", "overall_panic_score"),
    bandwidths: List[int] = (7, 14, 21, 30),
) -> pd.DataFrame:
    """Standard vs donut RDD, reported under both HAC(3) and HC1."""
    rows = []
    for outcome in outcomes:
        for bw in bandwidths:
            for spec, excl in (("Standard", None), ("Donut", DONUT_WINDOW)):
                for cov in ("HAC", "HC1"):
                    r, _, _ = fit_local_linear_rdd(
                        daily, outcome, bw, LOCKDOWN,
                        use_hac=(cov == "HAC"), hac_lags=3, exclude_window=excl,
                    )
                    rows.append({
                        "outcome": outcome, "bandwidth": bw, "spec": spec, "cov": cov,
                        "tau": round(r.effect, 6), "se": round(r.se, 6),
                        "p": round(r.pvalue, 6), "n_days": r.n_days,
                        "pre_mean": round(r.pre_mean, 6), "post_mean": round(r.post_mean, 6),
                    })
    return pd.DataFrame(rows)


def run_hac_sensitivity(
    daily: pd.DataFrame,
    outcome: str = "panic_rate",
    bandwidths: List[int] = (7, 14, 21, 30),
    lags: List[int] = (0, 1, 2, 3, 5, 7, 10),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """HAC maxlag sensitivity plus residual autocorrelation diagnostics.

    Motivation: the supervisor expected serial correlation to make the HAC
    standard error larger than HC1 and to remove significance. This routine
    reports the actual pattern instead of assuming one.
    """
    rows, diag_rows = [], []
    for bw in bandwidths:
        for spec, excl in (("Standard", None), ("Donut", DONUT_WINDOW)):
            r, sub, _ = fit_local_linear_rdd(daily, outcome, bw, LOCKDOWN,
                                             use_hac=False, exclude_window=excl)
            base = {"outcome": outcome, "bandwidth": bw, "spec": spec,
                    "tau": round(r.effect, 6), "n_days": r.n_days}
            for L in lags:
                _, _, fit = fit_local_linear_rdd(
                    daily, outcome, bw, LOCKDOWN,
                    use_hac=(L > 0), hac_lags=max(L, 1), exclude_window=excl,
                )
                rows.append({**base, "estimator": "HC1" if L == 0 else f"HAC({L})",
                             "se": round(float(fit.bse["treated"]), 6),
                             "p": round(float(fit.pvalues["treated"]), 6)})
            # Residual autocorrelation on the standard specification only.
            if spec == "Standard":
                resid = np.asarray(
                    fit_local_linear_rdd(daily, outcome, bw, LOCKDOWN,
                                         use_hac=False)[2].resid, dtype=float)
                for k in range(1, 6):
                    ac = float(np.corrcoef(resid[:-k], resid[k:])[0, 1]) if len(resid) > k else np.nan
                    diag_rows.append({"outcome": outcome, "bandwidth": bw,
                                      "lag": k, "acf": round(ac, 4)})
                try:
                    lb = sm_diagnostic.acorr_ljungbox(resid, lags=[1, 3, 5], return_df=True)
                    for lag_val, row in lb.iterrows():
                        diag_rows.append({"outcome": outcome, "bandwidth": bw,
                                          "lag": f"LB({lag_val})",
                                          "acf": round(float(row["lb_pvalue"]), 4)})
                except Exception as exc:  # pragma: no cover
                    print(f"  [warn] Ljung-Box failed at bw={bw}: {exc}")
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def load_hourly() -> Optional[pd.DataFrame]:
    """Load the hourly aggregate series, if available."""
    for path in HOURLY_FILE_CANDIDATES:
        if os.path.exists(path):
            h = pd.read_csv(path, encoding="utf-8-sig")
            h.columns = [c.lstrip("\ufeff").strip() for c in h.columns]
            h["hour"] = pd.to_datetime(h["hour"])
            h = h.sort_values("hour").reset_index(drop=True)
            # Use the explicit ratio so the denominator matches the daily table.
            h["panic_rate"] = np.where(h["covid_posts"] > 0,
                                       h["panic_posts"] / h["covid_posts"], 0.0)
            print(f"  hourly series: {path}  n={len(h)}  "
                  f"{h['hour'].min()} ~ {h['hour'].max()}")
            return h
    print("  [warn] hourly series not found; skipping the hourly block.")
    return None


def fit_hourly_rdd(
    hourly: pd.DataFrame,
    cutoff: pd.Timestamp,
    bandwidth_hours: int,
    outcome: str = "panic_rate",
    exclusions: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    use_hac: bool = True,
    hac_lags: int = 24,
    min_posts: int = 1,
) -> Optional[Dict[str, float]]:
    """Hourly local-linear RDD reusing the shared estimation core.

    ``hac_lags`` defaults to 24 so that the covariance estimator covers a full
    diurnal cycle in an hourly series.
    """
    d = hourly.loc[hourly["covid_posts"] >= min_posts].copy()
    d["running"] = (d["hour"] - cutoff).dt.total_seconds() / 3600.0
    d = d.loc[d["running"].abs() <= bandwidth_hours].copy()
    if exclusions:
        for lo, hi in exclusions:
            d = d.loc[~((d["hour"] >= lo) & (d["hour"] < hi))].copy()
    if len(d) < 20:
        return None
    weight = np.maximum(1 - np.abs(d["running"].values) / bandwidth_hours, 0.0) \
        * np.clip(d["covid_posts"].values, 1, None)
    fit = local_linear_core(d["running"].values, d[outcome].values, weight,
                            use_hac, hac_lags)
    return {
        "tau": float(fit.params["treated"]),
        "se": float(fit.bse["treated"]),
        "p": float(fit.pvalues["treated"]),
        "n_hours": int(len(d)),
        "n_pre": int((d["running"] < 0).sum()),
        "n_post": int((d["running"] >= 0).sum()),
    }


def run_hourly_analysis(
    outcome: str = "panic_rate",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Hourly discontinuity at both policy moments, plus placebo cutoffs."""
    hourly = load_hourly()
    if hourly is None:
        return None, None, None

    rows = []
    for cut_name, cut in (("enforce_10:00", HOURLY_ENFORCE),
                          ("announce_00:00", HOURLY_ANNOUNCE)):
        for bw in HOURLY_BANDWIDTHS:
            res = fit_hourly_rdd(hourly, cut, bw, outcome)
            if res is None:
                continue
            rows.append({"cutoff": cut_name, "bandwidth_h": bw,
                         "tau": round(res["tau"], 6), "se": round(res["se"], 6),
                         "p": round(res["p"], 6), "n_hours": res["n_hours"]})
    main_tbl = pd.DataFrame(rows)

    placebo = []
    for day in HOURLY_PLACEBO_DATES:
        cut = pd.Timestamp(day + " 10:00:00")
        for bw in PLACEBO_BANDWIDTHS:
            res = fit_hourly_rdd(hourly, cut, bw, outcome)
            if res is None:
                continue
            placebo.append({"cutoff_date": day, "is_real": int(day == "2020-01-23"),
                            "bandwidth_h": bw, "tau": round(res["tau"], 6),
                            "se": round(res["se"], 6), "p": round(res["p"], 6),
                            "n_hours": res["n_hours"]})
    placebo_tbl = pd.DataFrame(placebo)

    hourly = hourly.copy()
    hourly["hours_from_enforcement"] = (
        hourly["hour"] - HOURLY_ENFORCE).dt.total_seconds() / 3600.0
    profile = hourly.loc[hourly["hours_from_enforcement"].abs() <= 168, [
        "hour", "hours_from_enforcement", "total_posts", "covid_posts",
        "panic_posts", "panic_rate", "overall_panic_score"]]
    return main_tbl, placebo_tbl, profile


def run_denominator_comparison(
    daily: pd.DataFrame,
    bandwidths: List[int] = (7, 14, 21, 30),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Same days, two denominator conventions.

    ``panic_posts / covid_posts`` is the definition used throughout the paper.
    ``panic_posts / total_posts * 100`` is the older convention, reported only to
    show how strongly the topic-penetration drift contaminates the series.
    """
    d = daily.copy()
    d["panic_rate_covid"] = np.where(d["covid_posts"] > 0,
                                     d["panic_posts"] / d["covid_posts"], 0.0)
    d["panic_rate_total_pct"] = np.where(d["total_posts"] > 0,
                                         d["panic_posts"] / d["total_posts"] * 100.0, 0.0)
    d["covid_share"] = np.where(d["total_posts"] > 0,
                                d["covid_posts"] / d["total_posts"], np.nan)

    rows = []
    for outcome in ("panic_rate_covid", "panic_rate_total_pct"):
        for bw in bandwidths:
            for spec, excl in (("Standard", None), ("Donut", DONUT_WINDOW)):
                r, _, _ = fit_local_linear_rdd(d, outcome, bw, LOCKDOWN,
                                               use_hac=True, hac_lags=3,
                                               exclude_window=excl)
                rows.append({"denominator": outcome, "bandwidth": bw, "spec": spec,
                             "tau": round(r.effect, 6), "se": round(r.se, 6),
                             "p": round(r.pvalue, 6), "n_days": r.n_days})
    windows = [("2019-12", "2019-12-01", "2019-12-31"),
               ("2020-01-01~19", "2020-01-01", "2020-01-19"),
               ("01-20~22", "2020-01-20", "2020-01-22"),
               ("01-23~26", "2020-01-23", "2020-01-26"),
               ("01-27~02-10", "2020-01-27", "2020-02-10")]
    seg = []
    for name, a, b in windows:
        s = d.loc[(d["date"] >= a) & (d["date"] <= b)]
        seg.append({"window": name, "days": len(s),
                    "covid_share": round(float(s["covid_share"].mean()), 4),
                    "rate_total_pct": round(float(s["panic_rate_total_pct"].mean()), 4),
                    "rate_covid": round(float(s["panic_rate_covid"].mean()), 6)})
    return pd.DataFrame(rows), pd.DataFrame(seg)


def run_validity_filter_comparison(
    daily: pd.DataFrame,
    outcome: str = "panic_rate",
    bandwidths: List[int] = (7, 14, 21, 30),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Effect of dropping days whose denominator is structurally zero.

    A day with ``covid_posts == 0`` is assigned ``panic_rate = 0`` by the
    definition ``panic_posts / covid_posts``; that zero reflects the absence of
    COVID-related discussion, not the absence of panic. All 29 such days fall in
    December 2019, and six of them sit inside a +-30-day window, on the
    pre-cutoff side. This routine quantifies how much that matters.
    """
    d = daily.copy()
    d["valid_day"] = (d["covid_posts"] > 0).astype(int)
    avail = (d.loc[d["date"] >= START_DATE]
             .groupby(d["date"].dt.to_period("M"))
             .agg(days=("date", "size"),
                  zero_covid_days=("valid_day", lambda s: int((s == 0).sum())),
                  median_covid_posts=("covid_posts", "median"),
                  total_posts=("total_posts", "sum"))
             .reset_index())
    avail["month"] = avail["date"].astype(str)
    avail = avail.drop(columns=["date"])

    rows = []
    for bw in bandwidths:
        win = d.loc[d["days_from_lockdown"].abs() <= bw]
        n_zero = int((win["covid_posts"] == 0).sum())
        for label, kw in (("all days", {}),
                          ("valid days only", {"min_covid_posts": 1}),
                          ("valid days only (donut)", {"min_covid_posts": 1,
                                                       "exclude_window": DONUT_WINDOW})):
            r, sub, _ = fit_local_linear_rdd(d, outcome, bw, LOCKDOWN,
                                             use_hac=True, hac_lags=3, **kw)
            rows.append({"bandwidth": bw, "spec": label,
                         "n_days": r.n_days,
                         "structural_zeros_in_window": n_zero,
                         "tau": round(r.effect, 6), "se": round(r.se, 6),
                         "p": round(r.pvalue, 6)})
    return pd.DataFrame(rows), avail


def build_daily_metrics_streaming(
    files: Optional[List[dict]] = None,
    chunk_size: int = 200_000,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Stream the raw inference outputs into the daily series.

    The container running this pipeline is capped at **2 GB of RAM**
    (``/sys/fs/cgroup/memory.max``), which is far below the ~26 GB of server
    outputs; a pandas ``read_csv`` loop still peaks too high because every row
    has to be split on its delimiters, and the source files contain long free-text
    fields. This routine therefore walks the files with the stdlib ``csv``
    reader, which keeps memory flat regardless of file size.

    Every quantity needed downstream is an additive per-day sum::

        total_posts             <- row count
        covid_posts             <- sum(covid_related)
        panic_posts             <- sum(panic)
        overall_panic_score     <- sum(panic_prob) / count(panic_prob)
        conditional_panic_score <- sum(panic_prob | panic==1) / count(panic==1)

    Returns ``(daily, per_file, total_rows)``. ``daily`` carries the same columns
    as the in-memory path, so every figure and RDD block is unaffected.

    ``chunk_size`` is accepted for interface compatibility and is unused.
    """
    files = files if files is not None else DATA_FILES
    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    d0, d1 = START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d")
    # acc[date] = [total, covid, panic, prob_sum, prob_n, cond_sum, cond_n]
    acc: Dict[str, List[float]] = {}
    per_file, total_rows = [], 0

    def parse_day(raw: str) -> Optional[str]:
        s = raw.strip().replace("/", "-")
        head = s.split(" ")[0].split("T")[0]
        parts = head.split("-")
        if len(parts) != 3:
            return None
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return None
        if not (2000 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31):
            return None
        return "%04d-%02d-%02d" % (y, m, d)

    for cfg in files:
        path, panic_col = cfg["path"], cfg["panic_col"]
        if not os.path.exists(path):
            print(f"[skip] missing: {path}")
            continue
        enc = cfg.get("encoding") or "utf-8-sig"
        kept = 0
        with open(path, "r", encoding=enc, errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            try:
                hdr = [c.lstrip("\ufeff").strip() for c in next(reader)]
            except StopIteration:
                continue
            if "created_at" not in hdr or panic_col not in hdr:
                print(f"[skip] missing columns in {path}")
                continue
            i_t = hdr.index("created_at")
            i_p = hdr.index(panic_col)
            i_c = hdr.index("covid_related") if "covid_related" in hdr else None
            i_pp = hdr.index("panic_prob") if "panic_prob" in hdr else None
            if i_c is None and not cfg.get("assume_covid", False):
                print(f"[skip] no covid_related: {path}")
                continue
            need = max([i for i in (i_t, i_p, i_c, i_pp) if i is not None])

            for row in reader:
                if len(row) <= need:
                    continue
                day = parse_day(row[i_t])
                if day is None or day < d0 or day > d1:
                    continue
                cov = 1.0 if i_c is None else (1.0 if row[i_c].strip() == "1" else 0.0)
                try:
                    pan = float(row[i_p])
                except ValueError:
                    pan = 0.0
                pan = 1.0 if pan >= 0.5 else 0.0
                prob = pan
                if i_pp is not None:
                    try:
                        prob = float(row[i_pp])
                    except ValueError:
                        prob = pan
                b = acc.get(day)
                if b is None:
                    b = acc[day] = [0.0] * 7
                b[0] += 1.0
                b[1] += cov
                b[2] += pan
                b[3] += prob
                b[4] += 1.0
                if pan == 1.0:
                    b[5] += prob
                    b[6] += 1.0
                kept += 1
        total_rows += kept
        per_file.append({"file": os.path.basename(path), "rows_in_window": kept})
        print(f"[load] {os.path.basename(path)} rows={kept:,}", flush=True)

    if not acc:
        raise FileNotFoundError("No usable input files were found.")

    cols = ["total_posts", "covid_posts", "panic_posts", "prob_sum", "prob_n",
            "cond_sum", "cond_n"]
    daily = pd.DataFrame(
        [[day] + acc[day] for day in acc], columns=["date"] + cols)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").reindex(all_dates).fillna(0.0)
    daily["overall_panic_score"] = np.where(daily["prob_n"] > 0,
                                            daily["prob_sum"] / daily["prob_n"], 0.0)
    daily["conditional_panic_score"] = np.where(daily["cond_n"] > 0,
                                                daily["cond_sum"] / daily["cond_n"], 0.0)
    daily["panic_rate"] = np.where(daily["covid_posts"] > 0,
                                   daily["panic_posts"] / daily["covid_posts"], 0.0)
    daily = daily.reset_index().rename(columns={"index": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["days_from_lockdown"] = (daily["date"] - LOCKDOWN).dt.days
    daily["days_from_human_to_human"] = (daily["date"] - HUMAN_TO_HUMAN).dt.days
    for col in ["panic_rate", "overall_panic_score", "conditional_panic_score",
                "covid_posts", "panic_posts"]:
        daily[f"{col}_ma7"] = daily[col].rolling(window=7, center=True, min_periods=1).mean()

    print(f"[data] total rows={total_rows:,} | "
          f"date range={daily['date'].min().date()} ~ {daily['date'].max().date()}", flush=True)
    print(f"[data] covid_related=1: {int(daily['covid_posts'].sum()):,} | "
          f"panic=1: {int(daily['panic_posts'].sum()):,}", flush=True)
    return daily, pd.DataFrame(per_file), total_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified Weibo COVID-19 RDD pipeline (daily + donut + hourly).")
    p.add_argument("--donut", action="store_true",
                   help="run the donut comparison (Standard vs 01-20~22 excluded)")
    p.add_argument("--hourly", action="store_true",
                   help="run the hourly discontinuity analysis and placebo cutoffs")
    p.add_argument("--hac", action="store_true",
                   help="run HAC maxlag sensitivity and residual autocorrelation")
    p.add_argument("--denominator", action="store_true",
                   help="compare panic/covid against panic/total*100")
    p.add_argument("--validity", action="store_true",
                   help="drop days with structurally zero COVID posts and compare")
    p.add_argument("--all", action="store_true",
                   help="run every extension block")
    p.add_argument("--figures-only", action="store_true",
                   help="produce only the original figures and skip all extensions")
    p.add_argument("--outdir", default=None, help="override the output directory")
    p.add_argument("--in-memory", action="store_true",
                   help="use the original concatenate-then-groupby loader "
                        "(needs enough RAM for every input file at once)")
    p.add_argument("--chunk-size", type=int, default=200_000,
                   help="rows per chunk for the streaming loader")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir or OUTPUT_DIR

    # Extensions run by default; --figures-only switches them off.
    if args.figures_only:
        do_donut = do_hourly = do_hac = do_denom = do_valid = False
    elif args.all or not any([args.donut, args.hourly, args.hac, args.denominator,
                              args.validity]):
        do_donut = do_hourly = do_hac = do_denom = do_valid = True
    else:
        do_donut, do_hourly = args.donut, args.hourly
        do_hac, do_denom = args.hac, args.denominator
        do_valid = args.validity

    ensure_dir(outdir)
    print("=" * 80)
    print("UNIFIED PAPER FIGURE PIPELINE")
    print("=" * 80)

    df = None
    if args.in_memory:
        # Original path: requires enough RAM for every file at once.
        df = load_and_prepare()
        daily = build_daily_metrics(df)
        total_rows = int(len(df))
        per_file = pd.DataFrame()
    else:
        # Default path: constant-memory streaming aggregation.
        daily, per_file, total_rows = build_daily_metrics_streaming(
            chunk_size=args.chunk_size)
        if not per_file.empty:
            per_file.to_csv(os.path.join(outdir, "data_rows_per_file.csv"),
                            index=False, encoding="utf-8-sig")

    # Save unified daily data for later use.
    daily.to_csv(os.path.join(outdir, "daily_unified_metrics.csv"), index=False, encoding="utf-8-sig")

    # Diagnostics and summaries.
    event_summary = write_event_summary(daily)
    event_summary.to_csv(os.path.join(outdir, "event_window_summary_prepost.csv"), index=False, encoding="utf-8-sig")

    diagnostics = {
        "rows": int(total_rows),
        "days": int(len(daily)),
        "date_range": f"{daily['date'].min().date()} ~ {daily['date'].max().date()}",
        "panic_rate_min": float(daily["panic_rate"].min()),
        "panic_rate_max": float(daily["panic_rate"].max()),
        "overall_panic_score_min": float(daily["overall_panic_score"].min()),
        "overall_panic_score_max": float(daily["overall_panic_score"].max()),
        "conditional_panic_score_min": float(daily["conditional_panic_score"].min()),
        "conditional_panic_score_max": float(daily["conditional_panic_score"].max()),
        "adf_panic_rate": adf_check(daily["panic_rate"]),
        "adf_overall_panic_score": adf_check(daily["overall_panic_score"]),
    }

    # Figures
    plot_long_run_trends(daily)
    plot_policy_window(daily)
    rdd_main_results = plot_rdd_main(daily, bw=14)
    bw_results = plot_bandwidth_sensitivity(daily, bandwidths=[7, 14, 21, 30])
    cutoff_results = plot_cutoff_date_sensitivity(daily, bandwidth=21)

    # Single file summary for manuscript drafting.
    summary = {
        "diagnostics": diagnostics,
        "event_summary": event_summary.to_dict(orient="records"),
        "rdd_main": rdd_main_results,
        "bandwidth_results": bw_results.to_dict(orient="records"),
        "cutoff_sensitivity": cutoff_results.to_dict(orient="records"),
    }

    # ---- Block 2: donut / HAC / denominator / hourly ----
    if do_donut:
        print("\n[donut] Standard vs donut (01-20~22 removed), HAC(3) and HC1")
        donut_tbl = run_donut_comparison(daily)
        donut_tbl.to_csv(os.path.join(outdir, "rdd_donut_comparison.csv"),
                         index=False, encoding="utf-8-sig")
        print(donut_tbl[donut_tbl["cov"] == "HAC"].to_string(index=False))
        summary["donut_comparison"] = donut_tbl.to_dict(orient="records")

    if do_hac:
        print("\n[hac] maxlag sensitivity and residual autocorrelation")
        hac_tbl, acf_tbl = run_hac_sensitivity(daily)
        hac_tbl.to_csv(os.path.join(outdir, "rdd_hac_sensitivity.csv"),
                       index=False, encoding="utf-8-sig")
        acf_tbl.to_csv(os.path.join(outdir, "rdd_residual_acf.csv"),
                       index=False, encoding="utf-8-sig")
        print(hac_tbl[hac_tbl["spec"] == "Standard"].to_string(index=False))
        print("\n  residual ACF (standard specification):")
        print(acf_tbl[acf_tbl["lag"].astype(str).str.isdigit()].to_string(index=False))
        summary["hac_sensitivity"] = hac_tbl.to_dict(orient="records")
        summary["residual_acf"] = acf_tbl.to_dict(orient="records")

    if do_denom:
        print("\n[denominator] panic/covid vs panic/total*100")
        denom_tbl, denom_seg = run_denominator_comparison(daily)
        denom_tbl.to_csv(os.path.join(outdir, "rdd_denominator_comparison.csv"),
                         index=False, encoding="utf-8-sig")
        denom_seg.to_csv(os.path.join(outdir, "rdd_denominator_windows.csv"),
                         index=False, encoding="utf-8-sig")
        print(denom_seg.to_string(index=False))
        summary["denominator_comparison"] = denom_tbl.to_dict(orient="records")
        summary["denominator_windows"] = denom_seg.to_dict(orient="records")

    if do_valid:
        print("\n[validity] dropping days with structurally zero COVID posts")
        val_tbl, avail_tbl = run_validity_filter_comparison(daily)
        val_tbl.to_csv(os.path.join(outdir, "rdd_validity_filter.csv"),
                       index=False, encoding="utf-8-sig")
        avail_tbl.to_csv(os.path.join(outdir, "data_availability_by_month.csv"),
                         index=False, encoding="utf-8-sig")
        print(val_tbl.to_string(index=False))
        print("\n  monthly data availability:")
        print(avail_tbl.to_string(index=False))
        summary["validity_filter"] = val_tbl.to_dict(orient="records")
        summary["data_availability"] = avail_tbl.to_dict(orient="records")

    if do_hourly:
        print("\n[hourly] discontinuity at the announcement and enforcement moments")
        h_main, h_placebo, h_profile = run_hourly_analysis()
        if h_main is not None:
            h_main.to_csv(os.path.join(outdir, "rdd_hourly.csv"),
                          index=False, encoding="utf-8-sig")
            h_placebo.to_csv(os.path.join(outdir, "rdd_hourly_placebo.csv"),
                             index=False, encoding="utf-8-sig")
            h_profile.to_csv(os.path.join(outdir, "hourly_event_profile.csv"),
                             index=False, encoding="utf-8-sig")
            print(h_main.to_string(index=False))
            print("\n  placebo cutoffs sorted by |tau| within each bandwidth:")
            print(h_placebo.assign(abs_tau=h_placebo["tau"].abs())
                  .sort_values(["bandwidth_h", "abs_tau"])
                  .to_string(index=False))
            summary["hourly_main"] = h_main.to_dict(orient="records")
            summary["hourly_placebo"] = h_placebo.to_dict(orient="records")

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 80)
    print("DONE")
    print(f"Output dir: {outdir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
