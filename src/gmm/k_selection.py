# -*- coding: utf-8 -*-
"""抽取 panic 阳性子集的 panic_prob，并做完整的 K=1~8 模型选型。

目的：
  1) 服务器版 param_tuning1.py 的 K_RANGE = range(2,6)，K=5 是搜索上界，
     论文"BIC/AIC/silhouette 一致支持 K=5"的表述建立在受限搜索空间上。
     这里扩到 K=1~8，给出可写进论文的完整选型表。
  2) 同时保存 panic=1 的 panic_prob 全量向量（485,478 条）与
     panic=0 的抽样子集，供本地做截断效应检验与零模型对比。

输出（/root/autodl-tmp/）：
  panic1_probs.csv         panic=1 的 panic_prob（全量）
  panic0_probs_sample.csv  panic=0 的 panic_prob（抽样 200,000）
  k_selection_full.csv     K=1~8 的 BIC/AIC/Silhouette
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

DATA_DIR = "/root/autodl-tmp/"
FILES = ["temp19.csv"] + ["temp%d.csv" % i for i in range(1, 13)]
CHUNK = 200000
N_PANIC0_SAMPLE = 200000

t0 = time.time()
panic1, panic0 = [], []
rng = np.random.RandomState(42)

for f in FILES:
    p = os.path.join(DATA_DIR, f)
    if not os.path.exists(p):
        print("  [skip] missing", f, flush=True)
        continue
    n1 = n0 = 0
    for chunk in pd.read_csv(p, usecols=["panic", "panic_prob"], chunksize=CHUNK,
                             encoding="utf-8-sig", on_bad_lines="skip", low_memory=False):
        chunk["panic"] = pd.to_numeric(chunk["panic"], errors="coerce")
        chunk["panic_prob"] = pd.to_numeric(chunk["panic_prob"], errors="coerce")
        chunk.dropna(subset=["panic", "panic_prob"], inplace=True)
        p1 = chunk.loc[chunk["panic"] == 1, "panic_prob"].to_numpy()
        p0 = chunk.loc[chunk["panic"] == 0, "panic_prob"].to_numpy()
        if len(p1):
            panic1.append(p1)
            n1 += len(p1)
        if len(p0):
            keep = min(len(p0), 20000)
            panic0.append(rng.choice(p0, keep, replace=False))
            n0 += keep
    print("  %-14s panic=1 %7d | panic=0 sampled %6d  [%.0fs]" % (f, n1, n0, time.time() - t0),
          flush=True)

x1 = np.concatenate(panic1)
x0 = np.concatenate(panic0)
print("\npanic=1 total %d | panic=0 sample %d" % (len(x1), len(x0)), flush=True)
pd.DataFrame({"panic_prob": x1}).to_csv(DATA_DIR + "panic1_probs.csv", index=False)
pd.DataFrame({"panic_prob": x0}).to_csv(DATA_DIR + "panic0_probs_sample.csv", index=False)

print("\n[分布] panic=1: min=%.4f max=%.4f mean=%.4f median=%.4f"
      % (x1.min(), x1.max(), x1.mean(), np.median(x1)), flush=True)

# ---- 完整 K 选型 ----
X = x1.reshape(-1, 1)
idx = rng.choice(len(x1), min(20000, len(x1)), replace=False)
rows = []
for k in range(1, 9):
    gm = GaussianMixture(n_components=k, covariance_type="full", random_state=42,
                         n_init=10, max_iter=500).fit(X)
    sil = np.nan
    if k >= 2:
        sil = silhouette_score(X[idx], gm.predict(X[idx]))
    means = np.sort(gm.means_.ravel())
    rows.append(dict(K=k, BIC=round(gm.bic(X), 1), AIC=round(gm.aic(X), 1),
                     Silhouette=round(float(sil), 4) if sil == sil else None,
                     means=[round(float(m), 4) for m in means]))
    print("K=%d BIC=%.1f AIC=%.1f Sil=%.4f means=%s"
          % (k, gm.bic(X), gm.aic(X), sil if sil == sil else -1, np.round(means, 4).tolist()),
          flush=True)

pd.DataFrame(rows).to_csv(DATA_DIR + "k_selection_full.csv", index=False)
print("\nsaved: panic1_probs.csv / panic0_probs_sample.csv / k_selection_full.csv")
print("elapsed %.0fs" % (time.time() - t0))
