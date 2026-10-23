# -*- coding: utf-8 -*-
"""导出精简中间产物：日度计数 + 恐慌阳性 (date, prob) 列表。

用法：python3 export_subset.py <输入csv> <输出前缀>
产出：<前缀>_daily.csv （date,n_covid,n_panic）
      <前缀>_pos.csv   （date,panic_prob，仅 panic_prob>=0.3）
"""
import csv
import sys
import collections

inp, outpre = sys.argv[1], sys.argv[2]


def norm_day(s):
    s = (s or '').strip().split(' ')[0]
    if '/' in s:
        p = s.split('/')
        if len(p) == 3:
            return f'{int(p[0]):04d}-{int(p[1]):02d}-{int(p[2]):02d}'
    return s[:10]


dc = collections.Counter()
dp = collections.Counter()
probs = []
n = 0

with open(inp, newline='', encoding='utf-8', errors='ignore') as f:
    r = csv.reader(f)
    h = next(r)
    it, ic, ip = h.index('created_at'), h.index('covid_related'), h.index('panic_prob')
    for row in r:
        n += 1
        try:
            if row[ic].strip() in ('1', '1.0', 'True', 'true'):
                d = norm_day(row[it])
                if not d:
                    continue
                dc[d] += 1
                p = float(row[ip])
                if p >= 0.3:
                    dp[d] += 1
                    probs.append((d, p))
        except (IndexError, ValueError):
            continue

with open(f'{outpre}_daily.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['date', 'n_covid', 'n_panic'])
    for d in sorted(dc):
        w.writerow([d, dc[d], dp[d]])

with open(f'{outpre}_pos.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['date', 'panic_prob'])
    for d, p in probs:
        w.writerow([d, p])

print(f'OK {outpre}: rows={n} covid={sum(dc.values())} pos={len(probs)}', flush=True)
