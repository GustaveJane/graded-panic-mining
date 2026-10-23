# Reproduction guide

This document maps every computational step in the paper to the code that produces it.

---

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set the corpus location once in `config/config.yaml`:

```yaml
paths:
  raw_data_dir: "/path/to/weibo-corpus"
```

All randomness is seeded explicitly (`random_state=0` for mixtures, fixed bootstrap seeds elsewhere).

---

## 1. Stage-by-stage sequence

| # | Stage | Command | Output |
|---|---|---|---|
| 1a | Cleaning & de-duplication | `python -m src.data.clean` | cleaned post table |
| 1b | COVID-relevance filtering | `python -m src.data.filter_covid` | `covid_related` flag |
| 1c | User-level cleaning | `python -m src.data.user_clean` | active-user panel base |
| 2a | PanicBERT fine-tuning | `python -m src.classifier.train` | `models/panic_bert/` |
| 2b | Corpus inference | within `src.classifier` | per-post `panic_prob` |
| 2c | LLM semi-automatic coding | `python -m src.annotation.build_doc_taxonomy` | L1/L2/L3 labels |
| 2d | GMM stratification | `python -m src.gmm.fit_gmm_full` | tiers + thresholds `t1=0.4722, t2=0.9799` |
| 3a | BERTopic modelling | `python -m src.topic.topic_model` | 781 topics |
| 3b | Tier-crossed sampling + IPW | `python -m src.topic.topic_sample` | weighted sample (278,971) |
| 3c | Topic reporting | `python -m src.topic.topic_report` | Tables 6/7/22/26, Figs 4/7/9 |
| 4a | RDD main analysis | `python -m src.rdd.rdd_main` | Tables 11/13/15/16, Figs 2/5/6 |
| 4b | MSE-optimal Fig. 3 | `python -m src.rdd.fig3_mse` | Fig. 3 |
| 4c | Hourly RDD | `python -m src.rdd.hourly` | Table 17, Fig 8 |
| 4d | HAC diagnostics | `python -m src.rdd.hac_sensitivity` | Table 12 |
| 4e | Denominator audit | `python -m src.rdd.denominator` | Tables 10/14 |
| 4f | Holiday placebo | `python -m src.rdd.placebo` | Table 18 |
| 4g | Anchor events | `python -m src.events.event_study` | Tables 19/20 |
| 4h | User panel | `python -m src.panel.user_tracking` | Table 23, Fig 12 |
| 4i | Transition matrix | `python -m src.panel.transition_matrix` | Fig 11 |
| 4j | Appraisal mediation | `python -m src.appraisal.mediation` | Table 24 |
| 5a | Second-event transfer | `python -m src.transfer.xinfadi` | transfer section |
| 5b | Label-noise budget | `python -m src.robustness.label_noise` | Fig. 13 |
| 5c | Tier-crossed ablation | `python -m src.robustness.ablation` | Tables 2/25 |

---

## 2. Paper element → code mapping

### Tables

| Paper table | Produced by |
|---|---|
| T1 study comparison | hand-curated (no code) |
| T2 tier-crossed vs random | `src/robustness/ablation.py` |
| T3 binary detection metrics | `src/classifier/train.py` |
| T4 blind human validation | `src/validation/construct_validity.py` |
| T5 panic levels × windows | `src/rdd/rdd_main.py` |
| T6 content nature × windows | `src/topic/topic_report.py` |
| T7 content × panic tier | `src/topic/topic_report.py` |
| T8 GMM model selection | `src/gmm/k_selection.py` |
| T9 tier composition (4 rules) | `src/gmm/fit_gmm_full.py` |
| T10 denominator conventions | `src/rdd/denominator.py` |
| T11 MSE-optimal bandwidths | `src/rdd/rdd_main.py` |
| T12 HAC lag sensitivity | `src/rdd/hac_sensitivity.py` |
| T13 inference conventions | `src/rdd/rdd_main.py` |
| T14 monthly availability | `src/data/clean.py` |
| T15 daily/hourly window | `src/rdd/rdd_main.py`, `src/rdd/hourly.py` |
| T16 donut RDD | `src/rdd/rdd_main.py` |
| T17 hourly RDD | `src/rdd/hourly.py` |
| T18 holiday placebo | `src/rdd/placebo.py` |
| T19 anchor events | `src/events/event_study.py` |
| T20 pre/post policy | `src/events/event_study.py` |
| T21 lockdown lexicon | `config/config.yaml` |
| T22 panic appeals × windows | `src/topic/topic_report.py` |
| T23 transition matrix | `src/panel/transition_matrix.py` |
| T24 DLUT convergent validity | `src/appraisal/mediation.py` |
| T25 topic coherence ablation | `src/robustness/ablation.py` |
| T26 content composition | `src/topic/topic_report.py` |

### Figures

| Paper figure | Produced by |
|---|---|
| F1 framework | conceptual diagram (no code) |
| F2 temporal evolution | `src/rdd/rdd_main.py` |
| F3 RDD main (MSE-optimal) | `src/rdd/fig3_mse.py` |
| F4 topic × panic tier | `src/topic/topic_report.py` |
| F5 cutoff-date sensitivity | `src/rdd/rdd_main.py` |
| F6 lockdown keywords (daily) | `src/rdd/rdd_main.py` |
| F7 A/B subcategories × windows | `src/topic/topic_report.py` |
| F8 lockdown keywords (hourly) | `src/rdd/hourly.py` |
| F9 overall topic composition | `src/topic/topic_report.py` |
| F10 blind human validation | `src/validation/construct_validity.py` |
| F11 transition matrix | `src/panel/transition_matrix.py` |
| F12 panic appeals pre/post | `src/panel/user_tracking.py` |
| F13 label-noise sensitivity | `src/robustness/label_noise.py` |

> `src/rdd/rdd_main.py` is a single 1,314-line analysis script that reproduces several of the tables and figures above in one pass; the per-element mapping is indicative of which block inside it is responsible.

---

## 3. Expected intermediate artefacts

| Artefact | Produced by | Used by |
|---|---|---|
| `daily_unified_metrics.csv` | `src.rdd.rdd_main` | all daily analyses |
| `rdd_bandwidth_results.csv` | `src.rdd.rdd_main` | T11 |
| `daily_stats_detail.csv` | `src.data.clean` | denominator audit |
| tier-labelled post table | `src.gmm.fit_gmm_full` | topic stage, panels |
| `bh_full_hourly.csv` | `src.rdd.hourly` | T17 |

---

## 4. Runtime notes

- Stage 1–2 dominate wall-clock time (corpus-scale BERT inference); a GPU is strongly recommended.
- Stages 3–5 are CPU-bound and complete in minutes on the aggregated series.
- The transfer demonstration (`src/transfer/`) can be run on a monthly-sliced corpus; see the module docstring for the expected input layout.

---

## 5. Transferring to a new corpus

1. Quota-sample ~5,000 posts from the target event, front-loaded to its high-panic window.
2. **Re-estimate** the GMM thresholds — they are corpus-specific by construction; do not reuse `t1 = 0.4722 / t2 = 0.9799` without re-fitting.
3. Re-derive or translate the appraisal lexicon, and re-validate it against human ratings before interpreting the mechanism stage.
4. Stages 1–3 transfer unconditionally; Stage 4 requires a clearly dated, well-publicised intervention.
