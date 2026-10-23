# Stratified five-window user panel

`user_tier_panel_5windows.csv.gz` — the de-identified user panel behind the
per-user analyses (Appendix D of the manuscript). Gzip-compressed CSV, 6.1 MB on
disk, 503,988 rows.

## What is in it

| Column | Type | Meaning |
|---|---|---|
| `user_id` | 16-char hex | Pseudonym. Replaces the original Weibo UID at source; the mapping is not retained by the authors and the identifier cannot be resolved back. |
| `created_at` | `YYYY-MM-DD HH:MM:SS` | Posting time, minute resolution. |
| `panic_prob` | float ∈ [0,1] | PanicBERT posterior for that post. |
| `tier` | `non-panic` / `low` / `medium` / `high` | GMM tier. `non-panic` = `panic_prob < 0.30`; `low` / `medium` / `high` split the panic-positive range at the two calibrated boundaries. |
| `window` | `Baseline` / `Signal` / `Announcement` / `Implementation` / `Stable` / `Outside` | Five-window scheme of Table 1; `Outside` means the post falls outside all five. |
| `is_in_window` | bool | `True` for the five-window span, `False` for posts the same user wrote elsewhere in the corpus. |

No post text, no raw user identifier and no location field is present. Every row is a
single post; posts from the same user are linkable through `user_id`, which is the
point of a panel.

## Panel definition

A user enters the panel if, over the corpus,

- at least **10 posts inside the five windows**, and
- at least **1 post in the baseline window**.

Every post those users wrote anywhere in the corpus then enters the file, with its
window label and `is_in_window` flag. This is what `src/panel/active_user_panel.py`
(per-window counting) and `src/panel/normal_user_panel.py` (`--min-win 10 --min-base 1`)
implement; both are the scripts that produced this file.

## Counts

| Quantity | This file | Appendix D of the manuscript |
|---|---|---|
| Users | **11,164** | 10,777 |
| Posts (all) | **503,988** | 503,234 |
| Posts in window | 214,253 | — |
| non-panic / low / medium / high (all posts) | 500,582 / 621 / 2,213 / 572 | — |

The released snapshot is not the same user set as the analysis panel of Appendix D. The
appendix describes an inclusion rule of at least 5 posts before 23 January and at least
3 after; the file released here uses the rule implemented by the two scripts above
(≥10 posts inside the five windows and ≥1 baseline post). Both draw on the same corpus,
but they are different panels, which is why the counts differ (11,164 vs 10,777 users).
The appendix figures were computed on its own panel and are not reproducible from this
file without re-running the labelling stage.

## Window boundaries

`Baseline` 2019-12-02 – 2020-01-19 · `Signal` 2020-01-20 – 2020-01-22 ·
`Announcement` 2020-01-23 00:00–10:00 · `Implementation` 2020-01-23 10:00 – 2020-01-26 ·
`Stable` 2020-01-27 – 2020-02-10. Boundaries are taken from Table 1 of the manuscript;
the `window` column in this file is the authoritative membership, so use it rather
than recomputing from timestamps.

## Reproducing this file

On the corpus, run `src/panel/active_user_panel.py` (stages 1 and 2, `--min-posts 10`),
then `src/panel/normal_user_panel.py --min-win 10 --min-base 1`, then map the Chinese
tier labels to the English ones used here:

```
非恐慌 → non-panic    低恐慌 → low    中恐慌 → medium    高恐慌 → high
```

The released file is sorted by `created_at` so that line-level diffs stay stable.
