# Active-user panel behind Appendix D

`user_panel_5windows.csv.gz` — the per-post, de-identified panel that Appendix D is
computed on. Gzip-compressed CSV, 6.2 MB, **503,234 rows / 10,777 users**, sorted by
`created_at`.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `user_id` | 16-char hex | Pseudonym. Replaces the Weibo UID at source; the mapping is not retained by the authors. |
| `created_at` | `YYYY-MM-DD HH:MM:SS` | Posting time, minute resolution. |
| `covid_related` | 0 / 1 | COVID-relevance flag. `0` is the `O Non-epidemic` row of Table 13. |
| `panic_prob` | float ∈ [0,1] | PanicBERT posterior. |
| `tier` | `non-panic` / `low` / `medium` / `high` | GMM tier (`panic_prob` split at 0.30, 0.4722, 0.9799). |
| `window` | `Baseline` / `Signal` / `Announcement` / `Implementation` / `Stable` / `Outside` | Five-window scheme of Table 1. |
| `is_in_window` | bool | `True` inside the five windows. |
| `period` | `pre` / `post` | Before / after 2020-01-23, the split used in Appendix D. |

No post text, no raw identifier and no location field is present. Posts from the same
user are linkable through `user_id`, which is the point of a panel.

## Panel definition

A user enters the panel if it wrote **at least 5 posts before 2020-01-23** and **at
least 3 posts after**. Every post those users wrote anywhere in the corpus then enters
the file. This reproduces the appendix exactly:

| Quantity | This file | Appendix D |
|---|---|---|
| Users | **10,777** | 10,777 |
| Posts | **503,234** | 503,234 |
| Posts pre / post | 71,551 / 431,683 | — |
| Non-epidemic share, pre → post | **8.70% → 16.43%** | 8.70 → 16.43 |
| Users with epidemic posts in both periods | **98.93%** | 98.93% |
| Tier counts (non-panic / low / medium / high) | 498,768 / 635 / 3,051 / 780 | — |

The 98.93% figure uses all 10,777 users as the denominator (10,662 users post epidemic
content in both periods; 18 users have no epidemic post at all). Computed only over
users who post epidemic content, the share is 99.10% — the same quantity with the 18
users excluded, so quote whichever denominator the sentence uses.

## What this file cannot reproduce

Table 13 splits the epidemic posts into `A / B / C / D / N`. Those labels are **not**
included here and cannot be regenerated from the corpus: the labelling stage (rule
pass plus LLM fallback) is not part of the released code, and the per-post label file
was not preserved. The keyword classifiers in `../../src/topic/topic_model.py` are the
ones the paper describes as *reference hints that do not decide the final label*; run
on this panel they yield roughly 2% `A` against the 43% in Table 13, so they are not a
substitute. Reproducing Table 13 therefore requires re-running the LLM labelling stage
over the 426,090 epidemic posts in this file.

## Window boundaries

`Baseline` 2019-12-02 – 2020-01-19 · `Signal` 2020-01-20 – 2020-01-22 ·
`Announcement` 2020-01-23 00:00–10:00 · `Implementation` 2020-01-23 10:00 – 2020-01-26 ·
`Stable` 2020-01-27 – 2020-02-10.

## Rebuilding

On the full monthly corpus (`temp19.csv` = 2019-12, `temp1..temp12.csv` = 2020):

```bash
python tools/rebuild_appd_panel.py    # selects the 10,777 users, writes panel_users.csv
python tools/export_appd_panel.py     # writes this per-post file
```

Both scripts are in `../../tools/`. The tier and window columns are derived inside the
export script, so the file can be regenerated end to end without any intermediate
artefact.

`../../src/panel/` contains an alternative construction (`active_user_panel.py` plus
`normal_user_panel.py`, ≥10 posts inside the five windows and ≥1 baseline post) that
yields a different 11,164-user panel. It is used for the per-window user-tracking
descriptives, not for Appendix D.
