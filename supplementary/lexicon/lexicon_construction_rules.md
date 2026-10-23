# 词表构建：方法段、赋值规则与附录表

**词表来源**：`D:\xinguanbiaozhu\cognitive_proxy\rebuild\lexicon_cognitive.py`（v1, 2026-09-18）
**规模核验**：五维评估词 163 + 结构性特征 30 − 重叠 2 = **191 去重词条**（与 "191-term lexicon" 一致）
**日期**：2026-10-09

---

## 一、方法部分：词典构造段（英文，可直接粘贴）

```latex
\paragraph{Lexicon construction.} No published Chinese lexicon measures the appraisal
constructs required here: the DLUT affective lexicon ontology \cite{35} and the Chinese
LIWC dictionaries \cite{36,37} index emotional valence, intensity or general psychological
categories, but none of them covers the certainty and controllability dimensions on which
the proposed mechanism rests. We therefore constructed a task-specific lexicon along the
dimensional structure of appraisal theory, taking certainty and situational control from
Smith and Ellsworth \cite{38}, together with threat appraisal and coping potential, as the
dimensions to be measured. Seed terms were taken from the appraisal literature and from the
crisis vocabulary documented during the outbreak, and each seed was expanded with
morphological and synonymic variants attested in the platform register, yielding 191 unique
surface forms across five appraisal dimensions---threat (35), uncertainty (38), control
(39), helplessness (25) and hope (26)---and five structural features. Candidate terms were
retained only if they describe severity, certainty, controllability or coping orientation,
and were excluded if their frequency is driven by reference to the event itself; the
complete list is given in Appendix~\ref{app:lexicon}.
```

**句数核对**：4 句。① 为什么自建（公开词典缺 certainty/controllability 两维）；② 维度来源（Smith & Ellsworth 的 certainty 与 situational control + threat/coping）；③ **种子词来源与扩充方法** + 规模；④ 保留/排除判据 + 指向附录。

---

## 二、⚠️ 关于"两人独立编码一致性"

**必须如实说明：这次词典构建没有做过双人独立编码。**

我把 `D:\xinguanbiaozhu` 全库的 `reports/`、`cognitive_proxy/` 翻了一遍，**没有任何与词典相关的双人编码、Cohen's κ、Krippendorff α 记录**。词典是单人构建的，其后做的两项经验检验是：

| 已有的检验 | 内容 | 结果 |
|---|---|---|
| **DLUT 平行收敛效度** | 同一批 278,971 帖、同一套计数规则，只换词典（自建 191 词 vs DLUT 情感词汇本体 27,350 词条） | threat ~ 惧 +0.341、helplessness ~ 哀 +0.421、hope ~ 乐/好 +0.326/+0.338（皆 p<0.001，方向符合）；control 与 uncertainty **在 DLUT 中无对手方** ⇒ 这正是本文必须自建词典的实证理由 |
| **520 条构念效度标注** | 双人独立盲标 GMM 分层强度（1–5） | 二次加权 κ = 0.963，α = 0.963，ICC(2,1) = 0.963 |

注意后者验证的是 **GMM 分层**，不是词表本身，**不能挪用来充当词表编码信度**。

### 若要在方法里写这句，需要现在补做（约 30 分钟人工）

我已生成可直接使用的盲标表：

**`D:\xinguanbiaozhu\cognitive_proxy\rebuild\lexicon_two_coder_sheet.csv`**
- 163 行（五维评估词），`item_id, term, coder_A_dim, coder_B_dim, disagreement_resolution`
- **顺序已打乱、原维度已隐藏**，两位编码员各自独立把每个词归入 threat / control / uncertainty / helplessness / hope 之一
- 回收后我可以直接算 Cohen's κ、Krippendorff α（含 95% CI）与逐维混淆矩阵

拿到数后可补这一句（放在第 3 句之后）：

```latex
Two coders independently assigned every candidate term to one of the five dimensions while
blind to its source, and disagreements were resolved by discussion (Cohen's $\kappa$ = XXX,
$n$ = 163).
```

> 我不替你编这个 κ 值。要么补做后填真实数字，要么就用第一版（不提编码一致性，改以 DLUT 收敛效度 + 附录完整词表交代可信度）。

---

## 三、赋值规则（appendix 或方法脚注用）

### 3.1 归类规则（每个词形恰好一个维度）

1. 每个候选词形**只归入一个**评估维度，不在维度间重复。
2. 保留判据：描述 **严重性 / 确定性 / 可控性 / 应对取向** 之一。
3. 排除判据（三选一即剔除）：① 疾病名、地名、机构名；② 防护或政策物名（口罩、隔离、封城等）；③ 与 COVID 相关度过滤词表重叠者。
4. 主题名词**一律不参与匹配**，排除清单随附录一并列出（`疫情, 新冠, 新型冠状, 病毒, 肺炎, 武汉, 湖北, 口罩, 封城, 隔离, 确诊, 病例, 核酸, 疫苗, COVID, corona`）。
   *理由*：语料中疫情帖占比从 54% 升到 94%，含主题词的词典会把"信息洪流"误读为"威胁评估上升"。

### 3.2 匹配与计数

| 项 | 规则 |
|---|---|
| 匹配方式 | Aho–Corasick 多模式串匹配，**不做分词**，直接在未分词文本上扫 |
| 长串优先 | 词组优先于单词；单字词（如"我""吗"）只在结构性特征中使用 |
| 计数 | 命中次数（hits） |
| 归一化 | **ratio-of-sums**：`density_d = hits_d / total_chars × 1000`（每千字命中数） |
| 为何不用逐帖均值 | 避免短帖权重过大 |
| 聚合层 | 7 个分层：all / covid / panic / nonpanic / covid_nonpanic / covid_panic / original |
| 结构性特征 | 单独计数，仅作**描述性协变量**，不进入任何评估维度得分 |

### 3.3 已知的计数边界（主动披露）

- **8 组词条互为子串**（如 `死亡` ⊂ `死亡率`、`我` ⊂ `我们`），会造成重复计数。
- 量化：**五维重复计数上限 = 0.0555 次/千字 = 五维总和的 0.34%**（可忽略）。
- 唯一实质受影响的是结构性维度 `first_person`（多计占其自身 **18.09%**），但它只是描述性协变量，不影响机制结论。

```latex
\paragraph{Counting rule.} Densities are computed by multi-pattern string matching over
unsegmented text and normalised as hits per 1{,}000 characters within each stratum. Eight
entries are substrings of another entry in the same dimension. For the five appraisal
dimensions this double counting is bounded above by 0.056 hits per 1{,}000 characters, or
0.34\% of the five-dimension total. The only dimension materially affected is the structural
first-person feature, where the plural pronoun contains the singular and accounts for 18.1\%
of its matches; this enters only as a descriptive covariate.
```

### 3.4 词典覆盖率

扫描口径：2019-12-01 ~ 2020-02-12，13,277,642 帖 / 2,507,425,222 字。

| 维度 | 命中次数 | 每千字 | 占全部字符 |
|---|---|---|---|
| threat | 5,454,669 | 2.1754 | 0.218% |
| uncertainty | 4,949,456 | 1.9739 | 0.197% |
| control | 16,465,260 | 6.5666 | 0.657% |
| helplessness | 1,219,283 | 0.4863 | 0.049% |
| hope | 12,763,039 | 5.0901 | 0.509% |
| **五维合计** | **40,851,707** | **16.29** | **1.629%** |

分层：疫情帖层 16.49 次/千字；仅原创帖层 19.77 次/千字。

---

## 四、附录表

**文件**：`docs/06_附录_词表191.tex`（可直接粘贴的 `table*` 片段，`\label{app:lexicon}`）

结构：五维评估词逐维列出全部词形（35 / 39 / 38 / 25 / 26）→ 结构性特征 5 组 → 排除主题词清单。caption 已声明"每个词形恰好归入一个维度""维度独立计分（每千字命中数）""结构性特征不进入得分"。

### ⚠️ 编译层面的硬问题：表内有中文，**pdflatex 会直接报错**

`Unicode character ... not set up for use with LaTeX`。三条路选一条：

| 方案 | 说明 | 代价 |
|---|---|---|
| **A（推荐）** | 附录表所在文档改用 **XeLaTeX/LuaLaTeX + xeCJK**（BDMA/IEEE 模板通常可用 `\usepackage{xeCJK}`） | 需确认期刊允许；正文其余部分不受影响 |
| B | 把词表改成**纯英文译文表**（每个词给一个英文对应词） | 需人工过一遍 191 词译文；且译文会损失原词的语料证据力 |
| C | 词表放**补充材料**（Supplementary），正文只留维度规模与代表词 | 附录里就看不到完整词表，与"附录新增完整词表"的诉求不符 |

未修改任何论文文件，等你确认走哪条路。
