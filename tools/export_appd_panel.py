"""Export the Appendix D panel (10,777 users) as a per-post, de-identified file.

Users are selected by the Appendix D rule: >= 5 posts before 2020-01-23 and
>= 3 posts after. For those users every post in the corpus is written out with
its tier (from the paper's calibrated thresholds) and five-window label, so the
released file matches the panel the appendix was computed on.

Output: /root/autodl-tmp/appd_rebuild/appd_panel_posts.csv.gz
"""
import gzip
import os

import pandas as pd

BASE = "/root/autodl-tmp"
OUT = "/root/autodl-tmp/appd_rebuild"
CUT = pd.Timestamp("2020-01-23")
FILES = [f"{BASE}/temp19.csv"] + [f"{BASE}/temp{i}.csv" for i in range(1, 13)]

T1, T2, BIN = 0.4722, 0.9799, 0.30          # paper Section 3.3
WINDOWS = [                                  # Table 1
    ("Baseline",       "2019-12-02 00:00", "2020-01-19 23:59:59"),
    ("Signal",         "2020-01-20 00:00", "2020-01-22 23:59:59"),
    ("Announcement",   "2020-01-23 00:00", "2020-01-23 09:59:59"),
    ("Implementation", "2020-01-23 10:00", "2020-01-26 23:59:59"),
    ("Stable",         "2020-01-27 00:00", "2020-02-10 23:59:59"),
]

users = set(pd.read_csv(f"{OUT}/panel_users.csv", dtype={"user_id": str}).user_id)
print(f"panel users: {len(users):,}", flush=True)


def label_window(t):
    for name, a, b in WINDOWS:
        if (t >= pd.Timestamp(a)) and (t <= pd.Timestamp(b)):
            return name
    return "Outside"


def tier_of(p):
    if pd.isna(p) or p < BIN:
        return "non-panic"
    if p < T1:
        return "low"
    if p < T2:
        return "medium"
    return "high"


out_path = f"{OUT}/appd_panel_posts.csv.gz"
written = 0
with gzip.open(out_path, "wt", encoding="utf-8", newline="", compresslevel=6) as fh:
    header_done = False
    for path in FILES:
        if not os.path.exists(path):
            continue
        n = 0
        for chunk in pd.read_csv(path, usecols=["user_id", "created_at", "covid_related",
                                                "panic", "panic_prob"],
                                 dtype={"user_id": str}, chunksize=1_000_000):
            n += len(chunk)
            sub = chunk[chunk["user_id"].isin(users)].copy()
            if sub.empty:
                continue
            t = pd.to_datetime(sub["created_at"], format="mixed", errors="coerce")
            sub["created_at"] = t.dt.strftime("%Y-%m-%d %H:%M:%S")
            sub = sub[t.notna()]
            sub["covid_related"] = pd.to_numeric(sub["covid_related"], errors="coerce").fillna(0).astype(int)
            sub["panic_prob"] = pd.to_numeric(sub["panic_prob"], errors="coerce").round(4)
            sub["tier"] = sub["panic_prob"].map(tier_of)
            sub["window"] = t[t.notna()].map(label_window)
            sub["is_in_window"] = sub["window"] != "Outside"
            sub["period"] = (t[t.notna()] >= CUT).map({False: "pre", True: "post"})
            sub = sub[["user_id", "created_at", "covid_related", "panic_prob",
                       "tier", "window", "is_in_window", "period"]]
            sub.to_csv(fh, index=False, header=not header_done)
            header_done = True
            written += len(sub)
        print(f"  {os.path.basename(path)}: read {n:,}, kept running total {written:,}", flush=True)

print("\n=== DONE ===", flush=True)
print(f"rows written: {written:,}", flush=True)
print(f"file: {out_path} ({os.path.getsize(out_path)/1024/1024:.1f} MB)", flush=True)
