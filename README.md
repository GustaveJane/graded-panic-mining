# SSCM: Stratification–Semantics Coupled Mining framework

Code for the paper **"From Binary Sentiment to Graded Panic: A Stratification-Semantics Coupled Mining Framework for Crisis Corpora"** (target: *Big Data Mining and Analytics*).

SSCM turns a binary panic classifier into a **graded intensity measurement**, couples the grades with topic semantics through tier-crossed sampling, and links both to a descriptive policy-effect analysis. The Wuhan lockdown corpus (65,175,112 Weibo posts, Dec 2019 – Dec 2020) serves as the **case study**; the framework itself is designed to be transferred to other crisis corpora.

```
Stage 1  Data acquisition & preprocessing
         └─ dedup · bot/ad removal · 11 regex normalisation · key-variable extraction
Stage 2  Panic identification & stratification
         └─ PanicBERT fine-tuning (pani_prob) → GMM on the panic-positive subset → 4 tiers
Stage 3  Topic framework analysis
         └─ BERTopic (781 topics) × tier-crossed quota sampling → IPW reweighting
Stage 4  Policy-effect analysis
         └─ local-linear RDD · anchor-event study · user panel · appraisal mediation
```

---

## Repository layout

```
src/
├── data/           Stage 1  cleaning, COVID-relevance filtering, user cleaning
├── classifier/     Stage 2  PanicBERT fine-tuning + lexicon augmentation
├── annotation/     Stage 2  LLM semi-automatic coding (L1/L2/L3)
├── gmm/            Stage 2  mixture fitting, K selection, threshold calibration
├── topic/          Stage 3  BERTopic, tier-crossed sampling, topic reporting
├── validation/     §4.1    blind human validation of the intensity tiers
├── rdd/            Stage 4  local-linear RDD, hourly, placebo, HAC, denominators
├── events/         Stage 4  anchor-event response study
├── panel/          Stage 4  user-level tracking and transition matrix
├── appraisal/      Stage 4  appraisal-lexicon mediation analysis
├── transfer/       §4.5    second-event transfer demonstration (Beijing Xinfadi)
├── robustness/     Appendix  label-noise budget, tier-crossed ablation
├── pipeline/                stage entry points (thin placeholders that document the module map)
└── utils/                   shared loaders and plotting helpers
```

`scripts/legacy/` keeps the original single-file scripts before modularisation, for provenance. Files that were byte-identical duplicates of a module under `src/` have been dropped here, and the remaining scripts have been renamed to English while keeping their content unchanged. They are not part of the supported interface.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.12 is the tested version. A CUDA-capable GPU is recommended for Stage 2 (BERT fine-tuning); everything else runs on CPU.

---

## Data

**The raw Weibo corpus is not redistributed here.** It is derived from *Weibo-COV 2.0*, whose licence governs redistribution. See [`docs/data_availability.md`](docs/data_availability.md) for:

- how to request access to the source corpus,
- the schema the pipeline expects (one row per post, with `created_at`, `covid_related`, `panic_prob`),
- how to point the code at your own copy via `config/config.yaml`.

No post identifiers, user IDs or raw text are stored in this repository.

---

## Quick start

```bash
# 1. point the config at your local corpus
$EDITOR config/config.yaml        # set paths.raw_data_dir

# 2. run the stages in order
python -m src.data.filter_covid
python -m src.classifier.train
python -m src.gmm.fit_gmm_full
python -m src.topic.topic_model
python -m src.rdd.rdd_main

# or drive everything from the orchestrator
python -m src.pipeline.run_pipeline
```

Each stage writes to the directories declared in `config/config.yaml`. The stage entry points under `src/pipeline/step*.py` are thin placeholders that document the mapping.

See [`docs/reproduction.md`](docs/reproduction.md) for the full sequence, expected intermediate artefacts and the commands behind each paper table/figure.

---

## Transfer to a new corpus

`docs/reproduction.md` documents the transfer protocol. In short:

1. quota-sample ~5,000 posts from the target event, front-loaded to its high-panic window;
2. **re-estimate** the GMM thresholds on the target corpus — they are corpus-specific by design;
3. re-derive or translate the appraisal lexicon before interpreting the mechanism stage.

The second-event demonstration on the Beijing Xinfadi outbreak (June 2020) is in `src/transfer/`.

---

## Citation

```bibtex
@article{sscm2026,
  title   = {From Binary Sentiment to Graded Panic: A Stratification-Semantics
             Coupled Mining Framework for Crisis Corpora},
  journal = {Big Data Mining and Analytics},
  year    = {2026}
}
```

A `CITATION.cff` will be added once the paper is accepted.

---

## Licence

MIT — see [`LICENSE`](LICENSE).
