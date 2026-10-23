"""Build the `supplementary/` release package.

Every artefact here corresponds to an explicit promise in the manuscript
("Data and Code Availability" and the in-text "released in the supplementary
repository" statements).  Nothing is invented: each file is copied from, or
generated out of, material that already exists in this project.

Run:  python tools/build_supplementary.py
"""
from __future__ import annotations

import csv
import pathlib
import re
import shutil

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary"
DOCS = pathlib.Path(r"C:/Users/17853/WorkBuddy/2026-08-20-09-20-57/docs")


def reset_dir(p: pathlib.Path) -> None:
    # Idempotent: every artefact has a fixed name, so re-running simply overwrites.
    # (No rmtree: the build must not destroy anything if it is interrupted.)
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 1. Lexicon: parse the LaTeX appendix (authoritative 191-entry list)
# --------------------------------------------------------------------------
def build_lexicon() -> None:
    src = DOCS / "06_附录_词表191.tex"
    txt = src.read_text(encoding="utf-8")
    pat = re.compile(
        r"\\(?:textbf\{([^}]*)\}|quad\s+([A-Za-z][A-Za-z\- ]*?))\s*"
        r"(?:\(([^)]*)\))?\s*&\s*(\d+)\s*&\s*(.+?)\s*\\\\"
    )
    rows = []
    for m in pat.finditer(txt):
        zh, en_plain, en, n, forms = m.groups()
        zh = (zh or "").strip()
        en = (en or en_plain or "").strip()
        kind = "evaluative" if zh else "structural"
        for f in [x.strip() for x in forms.split(",") if x.strip()]:
            rows.append(dict(surface_form=f, dimension_en=en, dimension_zh=zh or en,
                             kind=kind, scored=1 if kind == "evaluative" else 0,
                             group_n=int(n)))

    # One surface form belongs to one dimension: evaluative assignment wins,
    # structural groups only claim what is left (matches the paper caption).
    prio = {"evaluative": 0, "structural": 1}
    rows.sort(key=lambda r: (prio[r["kind"]], r["dimension_en"]))
    seen, out = {}, []
    for r in rows:
        if r["surface_form"] in seen:
            seen[r["surface_form"]]["also_in"] = r["dimension_en"]
            continue
        r["also_in"] = ""
        seen[r["surface_form"]] = r
        out.append(r)

    out.sort(key=lambda r: (r["kind"] != "evaluative", r["dimension_en"]))
    d = OUT / "lexicon"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "lexicon.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["surface_form", "dimension_en", "dimension_zh",
                                          "kind", "scored", "group_n", "also_in"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(out)

    shutil.copy2(src, d / "lexicon_191_appendix.tex")
    alt = DOCS / "06b_附录_词表191_中英对照版.tex"
    if alt.exists():
        shutil.copy2(alt, d / "lexicon_191_appendix_cn_en.tex")
    rules = DOCS / "07_词表构建_方法段与赋值规则.md"
    if rules.exists():
        shutil.copy2(rules, d / "lexicon_construction_rules.md")

    n_eval = sum(r["kind"] == "evaluative" for r in out)
    print(f"[lexicon] {len(out)} unique surface forms ({n_eval} evaluative + "
          f"{len(out) - n_eval} structural)")


# --------------------------------------------------------------------------
# 2. Codebook
# --------------------------------------------------------------------------
def build_codebook() -> None:
    d = OUT / "codebook"
    d.mkdir(parents=True, exist_ok=True)
    src = DOCS / "05_编码框架_L2正式定义与边界规则.md"
    shutil.copy2(src, d / "codebook_L1_L2_L3.md")
    print("[codebook] copied")


# --------------------------------------------------------------------------
# 3. Prompts: lift the deployed templates verbatim out of the annotation script
# --------------------------------------------------------------------------
def build_prompts() -> None:
    d = OUT / "prompts"
    d.mkdir(parents=True, exist_ok=True)
    src = ROOT / "src" / "data" / "filter_covid.py"
    lines = src.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if "3种详细Prompt模板" in l)
    end = next(i for i, l in enumerate(lines) if "return [{\"role\": \"user\"" in l
               and i > start)
    body = "\n".join(lines[start:end]).rstrip()

    header = (
        "# Deployed panic-annotation prompt (Chinese template, verbatim)\n"
        "# Source: src/data/filter_covid.py, section 3 (three template variants).\n"
        "# 'fewshot' is the deployed variant; it carries the 12 worked examples\n"
        "# reproduced in Table 30 of the manuscript. 'example' carries 10 further\n"
        "# worked examples with full reasoning, released here as promised.\n"
        "# 'keyword' is the keyword-driven fallback variant.\n"
        "# Text is UTF-8; the placeholder {items} is filled with 10 numbered posts.\n\n"
    )
    (d / "panic_annotation_prompt_zh.py.txt").write_text(header + body + "\n",
                                                         encoding="utf-8")

    # few-shot examples -> TSV
    rows = []
    fs = re.search(r'elif template_type == "fewshot":(.*?)【待判定文本】', body, re.S)
    if fs:
        for m in re.finditer(r"\d+\.\s*文本：“(.+?)”\s*→\s*(\d)", fs.group(1)):
            rows.append(dict(source="fewshot (Table 30)", text=m.group(1), label=int(m.group(2))))
    ex = re.search(r'elif template_type == "example":(.*?)【输出要求', body, re.S)
    if ex:
        for m in re.finditer(r"【示例(\d+)：判定为(\d)[^】]*】\s*文本：“(.+?)”\s*分析：(.+?)\n(?:---|\n)",
                             ex.group(1), re.S):
            rows.append(dict(source=f"example (detailed #{m.group(1)})",
                             text=m.group(3), label=int(m.group(2)),
                             rationale=" ".join(m.group(4).split())))
    with open(d / "few_shot_examples.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "text", "label", "rationale"],
                           delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[prompts] templates extracted; {len(rows)} worked examples "
          f"({sum(r['source'].startswith('fewshot') for r in rows)} few-shot + "
          f"{sum(r['source'].startswith('example') for r in rows)} detailed)")


# --------------------------------------------------------------------------
# 4. Tables
# --------------------------------------------------------------------------
PAPER_EVENTS = [
    ("E1", "2019-12-31", "First official notice of unknown pneumonia", None, None, None),
    ("E2", "2020-01-20", "Human-to-human transmission confirmed", 2.63, 1.46, 0.84),
    ("E3", "2020-01-23", "Wuhan lockdown announced", 2.66, -1.13, -0.70),
    ("E4", "2020-01-27", "Spring Festival holiday extended", 2.81, -1.70, -0.94),
    ("E5", "2020-02-07", "Death of Dr. Li Wenliang; hospital crisis peak", 2.17, -1.12, -0.78),
    ("E6", "2020-03-24", "Hubei reopening announcement", 1.23, -0.47, -1.08),
    ("E7", "2020-04-04", "National mourning day", 1.06, -0.49, -1.16),
    ("E8", "2020-04-08", "Wuhan reopening", 0.88, -0.26, -0.97),
    ("E9", "2020-05-10", "Shulan cluster outbreak (Jilin)", 0.73, 0.14, 0.60),
    ("E10", "2020-05-21", "Two Sessions convene", 0.74, -0.23, -0.92),
    ("E11", "2020-06-11", "Beijing Xinfadi outbreak", 0.65, 1.71, 11.74),
]


def build_tables() -> None:
    outdir = OUT / "tables"
    outdir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ROOT / "full_gmm_thresholds.csv", outdir / "gmm_thresholds.csv")
    shutil.copy2(ROOT / "gmm_diagnostics_summary.csv", outdir / "gmm_diagnostics.csv")

    d = pd.read_csv(ROOT / "experiments/results/rdd/daily_unified_metrics.csv",
                    encoding="utf-8-sig")
    d.to_csv(outdir / "daily_covid_panic_2019_2020.csv", index=False, encoding="utf-8")

    # Recompute the eleven anchor-event responses with the recipe stated in the
    # Table 21 note, so the released table is regenerated rather than transcribed.
    d["date"] = pd.to_datetime(d["date"])
    elig = d[d["covid_posts"] >= 500].set_index("date").sort_index()
    rows = []
    for eid, date, name, base_p, delta_p, z_p in PAPER_EVENTS:
        ev = pd.Timestamp(date)
        pre = elig.loc[(elig.index >= ev - pd.Timedelta(days=32)) &
                       (elig.index <= ev - pd.Timedelta(days=3))]
        post = elig.loc[(elig.index >= ev) & (elig.index <= ev + pd.Timedelta(days=2))]
        if len(pre) < 5 or len(post) == 0:
            rows.append(dict(ID=eid, date=date, event=name, baseline_pct="", delta_pp="",
                             z_rate="", z_volume="", paper_baseline_pct="",
                             paper_delta_pp="", paper_z_rate="", note="insufficient baseline"))
            continue
        b = pre["panic_rate"].mean() * 100
        dl = (post["panic_rate"].mean() - pre["panic_rate"].mean()) * 100
        zr = dl / (pre["panic_rate"].std(ddof=1) * 100) if pre["panic_rate"].std(ddof=1) else np.nan
        vb = pre["total_posts"].mean()
        vz = ((post["total_posts"].mean() - vb) / pre["total_posts"].std(ddof=1)
              if pre["total_posts"].std(ddof=1) else np.nan)
        rows.append(dict(ID=eid, date=date, event=name,
                         baseline_pct=round(b, 2), delta_pp=round(dl, 2),
                         z_rate=round(float(zr), 2), z_volume=round(float(vz), 2),
                         paper_baseline_pct=base_p, paper_delta_pp=delta_p,
                         paper_z_rate=z_p, note=""))
    ev = pd.DataFrame(rows)
    ev.to_csv(outdir / "event_response_table.csv", index=False, encoding="utf-8")

    rec = pd.to_numeric(ev["delta_pp"], errors="coerce")
    pap = pd.to_numeric(ev["paper_delta_pp"], errors="coerce")
    diff = (rec - pap).abs().dropna()
    print(f"[tables] events recomputed: {len(ev)}; max |delta_pp - paper| = {diff.max():.2f} pp")
    print(ev[["ID", "baseline_pct", "delta_pp", "z_rate", "paper_delta_pp"]].to_string(index=False))
    print(f"[tables] daily rows: {len(d)} ({d['date'].min().date()} .. {d['date'].max().date()})")


# --------------------------------------------------------------------------
# 5. Preprocessing rules actually present in this repository
# --------------------------------------------------------------------------
def build_preprocessing_note() -> None:
    d = OUT / "preprocessing"
    d.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "src" / "data" / "clean.py").read_text(encoding="utf-8")
    emo = re.search(r"EMOTION_PATTERNS\s*=\s*\{(.*?)\n\}", src, re.S)
    note = [
        "# Preprocessing rule inventory",
        "",
        "This file inventories the preprocessing rules that are actually implemented in",
        "this repository. It is generated from the source, so the rules below are the",
        "rules the code runs — not a description written after the fact.",
        "",
        "## 1. Stage implemented in `src/data/clean.py`",
        "",
        "- Removal of repost boilerplate: `转发理由...`, `转发微博`, `Repost` (case-insensitive).",
        "- Topic-key construction: punctuation and whitespace stripped before comparison.",
        "- Separation of a reposted news body from the user's own comment",
        "  (markers: `转发理由`, `转发微博`, `Repost`, `轉發微博`, `全文转发理由`).",
        "- Six-way news-type classification: domestic epidemic / international epidemic /",
        "  policy release / data statistics / science communication / social events.",
        "- Emotion-pattern detection for the comment layer (patterns reproduced below).",
        "",
        "```python",
        ("EMOTION_PATTERNS = {" + emo.group(1) + "\n}") if emo else "(pattern block not found)",
        "```",
        "",
        "## 2. Stage implemented in `src/data/filter_covid.py`",
        "",
        "- Reads the monthly `cleaned_YYYY-MM.csv` files, strips newlines from column names,",
        "  drops empty text, and draws a fixed-size sample per month (`random_state=42`).",
        "- Runs the three-round LLM voting annotation; the deployed prompt is released",
        "  verbatim under `../prompts/`.",
        "",
        "## 3. Upstream cleaning that is NOT in this repository",
        "",
        "The corpus-level deduplication, advertisement/bot removal and text normalisation",
        "produced the `cleaned_YYYY-MM.csv` inputs consumed above. That script ran on the",
        "compute server and is not part of this repository, so its exact rule list cannot",
        "be released here. Anything in the manuscript that refers to \"complete rule",
        "lists\" for those three steps cannot yet be honoured by this repository.",
        "",
    ]
    (d / "preprocessing_rules.md").write_text("\n".join(note), encoding="utf-8")
    print("[preprocessing] rule inventory written")


def main() -> None:
    reset_dir(OUT)
    build_lexicon()
    build_codebook()
    build_prompts()
    build_tables()
    build_preprocessing_note()
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
