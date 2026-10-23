"""Rebuild the Appendix D panel from the monthly temp files.

Definition under test (paper, Appendix D):
    active users with >= 5 posts before 2020-01-23 and >= 3 posts after
    (reported: 10,777 users / 503,234 posts)

Stage 1  read 2019-12 and 2020-01 only, count each user's pre-cutoff posts,
         keep users with pre >= 5  (the pre-period lies entirely inside these
         two files, so no other file can contribute).
Stage 2  stream every monthly file, and for the stage-1 candidates only, count
         post-cutoff posts and split both periods by covid_related, so the
         non-epidemic (O) share of Table 13 and the "epidemic posts in both
         periods" statistic can be checked without a second pass.

Output: appd_rebuild/panel_users.csv + a printed summary.
"""
import os

import pandas as pd

BASE = "/root/autodl-tmp"
OUT = "/root/autodl-tmp/appd_rebuild"
CUT = pd.Timestamp("2020-01-23")
os.makedirs(OUT, exist_ok=True)

PRE_FILES = [f"{BASE}/temp19.csv", f"{BASE}/temp1.csv"]
ALL_FILES = [f"{BASE}/temp19.csv"] + [f"{BASE}/temp{i}.csv" for i in range(1, 13)]
COLS = ["user_id", "created_at", "covid_related"]


def parse(s):
    return pd.to_datetime(s, format="mixed", errors="coerce")


# ---------------- Stage 1 ----------------
pre_counts = {}
for path in PRE_FILES:
    if not os.path.exists(path):
        print(f"[skip] {path}", flush=True)
        continue
    n = 0
    for chunk in pd.read_csv(path, usecols=["user_id", "created_at"],
                             dtype={"user_id": str}, chunksize=1_000_000):
        t = parse(chunk["created_at"])
        m = t < CUT
        if m.any():
            for uid, c in chunk.loc[m, "user_id"].value_counts().items():
                pre_counts[uid] = pre_counts.get(uid, 0) + int(c)
        n += len(chunk)
    print(f"[stage1] {os.path.basename(path)}: {n:,} rows; users {len(pre_counts):,}",
          flush=True)

cands = {u for u, c in pre_counts.items() if c >= 5}
print(f"[stage1] users with >=5 pre-cutoff posts: {len(cands):,}", flush=True)

# ---------------- Stage 2 ----------------
# per user: total, pre, post, covid1_pre, covid1_post
agg = {}
for path in ALL_FILES:
    if not os.path.exists(path):
        continue
    n = 0
    for chunk in pd.read_csv(path, usecols=COLS, dtype={"user_id": str},
                             chunksize=1_000_000):
        n += len(chunk)
        sub = chunk[chunk["user_id"].isin(cands)]
        if sub.empty:
            continue
        t = parse(sub["created_at"])
        cov = pd.to_numeric(sub["covid_related"], errors="coerce").fillna(0).astype(int)
        is_pre = (t < CUT).to_numpy()
        is_cov = (cov == 1).to_numpy()
        uid = sub["user_id"].to_numpy()
        for u, pr, cv in zip(uid, is_pre, is_cov):
            a = agg.get(u)
            if a is None:
                a = agg[u] = [0, 0, 0, 0, 0]      # total, pre, post, cov_pre, cov_post
            a[0] += 1
            if pr:
                a[1] += 1
                if cv:
                    a[3] += 1
            else:
                a[2] += 1
                if cv:
                    a[4] += 1
    print(f"[stage2] {os.path.basename(path)}: {n:,} rows; users seen {len(agg):,}",
          flush=True)

rows = []
for u in cands:
    a = agg.get(u)
    if a and a[2] >= 3:
        rows.append(dict(user_id=u, total=a[0], pre=a[1], post=a[2],
                         covid_pre=a[3], covid_post=a[4]))

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/panel_users.csv", index=False)

pre_all, post_all = df.pre.sum(), df.post.sum()
print("\n=== RESULT ===", flush=True)
print(f"users (pre>=5 & post>=3): {len(df):,}")
print(f"posts total:              {df.total.sum():,}")
print(f"posts pre / post:         {pre_all:,} / {post_all:,}")
print(f"O share pre:  {100 * (1 - df.covid_pre.sum() / pre_all):.2f}%")
print(f"O share post: {100 * (1 - df.covid_post.sum() / post_all):.2f}%")
both = ((df.covid_pre > 0) & (df.covid_post > 0)).mean() * 100
print(f"users with epidemic posts in both periods: {both:.2f}%")
print("paper reports: 10,777 users / 503,234 posts; O 8.70% -> 16.43%; 98.93% both")
