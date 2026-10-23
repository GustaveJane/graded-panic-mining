#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topic_model.py — BERTopic 聚类 + 人工编码（两阶段版本）

逻辑完全对应你的框架：
    BERTopic 聚类 → 人工编码主题（A/B/C/D → B1-B4 → P1-P8） → 映射回文档 → 分布矩阵

阶段一（Stage 1）— 聚类 + 生成编码模板：
    python topic_model.py --stage 1
    1) 读取 topic_corpus.csv，清洗过滤
    2) 训练 BERTopic，得到每个文档的 topic_id
    3) 输出每个主题的：词表、代表文本、文档数、恐慌均值、关键词规则参考提示
    4) 生成 topic_list.csv（人工编码表），level1_code / level2_code / panic_code 留空待填

阶段二（Stage 2）— 读取人工编码并生成结果：
    python topic_model.py --stage 2                # 默认读取 topic_list.csv
    python topic_model.py --stage 2 --encoding xx.csv
    1) 校验人工编码（必填项、取值范围、条件约束），输出 encoding_validation_report.csv
    2) 构建主题级映射 topic_taxonomy_mapping.csv
    3) 把编码映射回每条文档，构建 doc_taxonomy_mapping.csv
    4) 生成分布矩阵：主题/一层框架/二层框架/恐慌诉求 × 风险组

说明：
- 关键词规则保留为“参考提示”列（ref_*），只辅助人工判断，不决定最终编码
- 人工只填写 level1_code / level2_code / panic_code / notes 四列
- 输出目录默认 topic_outputs/，可用环境变量 TOPIC_OUTPUT_DIR 修改

输出文件：
    Stage 1:
        topic_list.csv                    # 人工编码模板（含说明行与参考列）
        topic_info.csv / topic_keywords.csv
        topic_representative_docs.csv
        doc_topic_mapping.csv             # 文档-主题 底表
        embeddings.npy / bertopic_model/
    Stage 2:
        topic_taxonomy_mapping.csv        # 主题-人工编码 映射表
        doc_taxonomy_mapping.csv          # 文档-人工编码 映射表（后续事件分析主表）
        encoding_validation_report.csv
        topic_distribution_by_group.csv
        taxonomy_level1_distribution_by_group.csv
        taxonomy_level2_distribution_by_group.csv
        panic_type_distribution_by_group.csv
"""

import os
import re
import argparse
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None

BASE_DIR = Path(os.environ.get('TOPIC_BASE_DIR', Path(__file__).resolve().parent))
INPUT_PATH = Path(os.environ.get('TOPIC_INPUT_PATH', BASE_DIR / 'topic_corpus.csv'))
OUTPUT_DIR = Path(os.environ.get('TOPIC_OUTPUT_DIR', BASE_DIR / 'topic_outputs'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CLUSTER_SIZE = 40
MIN_SAMPLES = 5
N_NEIGHBORS = 15
N_COMPONENTS = 5
TOP_N_WORDS = 15
RANDOM_STATE = 42
N_REP_DOCS = 5
HIGH_PANIC_TH = 0.8

# ==============================
# 人工编码框架（与论文框架一致）
# ==============================
NATURE_CN = {
    'A': '原创情绪表达',
    'B': '新闻转发/通报',
    'C': '互助/求助',
    'D': '闲谈/生活吐槽',
    'N': '噪声/未归类',
}

NATURE_DETAIL_CN = {
    'A1': '情绪宣泄',
    'A2': '纪念/致敬',
    'A3': '希望/鼓励',
    'A4': '讽刺/调侃',
    'B1': '国内疫情数据',
    'B2': '国际疫情动态',
    'B3': '政策干预信息',
    'B4': '医疗/科普信息',
    'C1': '互助/求助',
    'D1': '日常生活吐槽',
}

PANIC_CN = {
    'P1': '医疗资源求助',
    'P2': '生活物资焦虑',
    'P3': '复工/复学焦虑',
    'P4': '隔离/封控影响',
    'P5': '情绪宣泄',
    'P6': '信息寻求',
    'P7': '症状/确诊担忧',
    'P8': '出行/交通焦虑',
}

# 校验规则
L1_RULES = {
    # level1: (默认 level2, 允许的 level2 集合)；None 表示必须人工填写
    'A': ('A1', {'A1', 'A2', 'A3', 'A4'}),
    'B': (None, {'B1', 'B2', 'B3', 'B4'}),
    'C': ('C1', {'C1'}),
    'D': ('D1', {'D1'}),
    'N': ('', set()),
}
VALID_PANIC = {f'P{i}' for i in range(1, 9)}

ENCODING_INSTRUCTIONS = [
    '# ==================== 人工编码说明（请先阅读再填写）====================',
    '# 1. 第一层 level1_code（必填）：A=原创情绪表达  B=新闻转发/通报  C=互助/求助  D=闲谈/生活吐槽  N=噪声/未归类',
    '# 2. 第二层 level2_code：',
    '#    A类→A1情绪宣泄 / A2纪念致敬 / A3希望鼓励 / A4讽刺调侃',
    '#    B类→B1国内疫情数据 / B2国际疫情动态 / B3政策干预信息 / B4医疗科普信息（B类必填）',
    '#    C类→C1  D类→D1  N类留空',
    '# 3. 第三层 panic_code（无单一主导诉求可留空）：P1医疗资源求助 P2生活物资焦虑 P3复工复学焦虑 P4隔离封控影响 P5情绪宣泄 P6信息寻求 P7症状确诊担忧 P8出行交通焦虑',
    '# 4. 判断依据：以 top_words + representative_text 的整体语义为准；ref_* 列为关键词规则的参考提示，仅供参考，可以不同意。',
    '# 5. 只填写 level1_code / level2_code / panic_code / notes 四列，其余列不要改动。',
    '# 6. 填写完成后运行：python topic_model.py --stage 2',
    '# ================================================================',
]

# ==============================
# 关键词规则（仅作为人工编码的参考提示）
# ==============================
HELP_KW = ['求助', '求救', '帮忙', '支援', '救命', '床位', '收治', '医院', '门诊', '急诊', '转发求助', '资源', '捐', '捐款', '募捐', '援助']
EMOTION_KW = ['慌', '害怕', '恐慌', '崩溃', '失眠', '心痛', '难过', '无助', '紧张', '焦虑', '烦', '疯', '想哭', '可怕', '惊慌', '瑟瑟发抖', '太难了', '心塞', '丧', 'emo']
NEWS_KW = ['确诊', '病例', '新增', '通报', '地图', '发布会', '卫健委', '本土', '输入', '密接', '阳性', '累计', '无症状', '封闭管理', '战时机制', '防控', '国家卫健委', '钟南山', '华南海鲜市场', '冷链', '核酸阳性']
DAILY_KW = ['复工', '开学', '复学', '上班', '下班', '在家', '回家', '出门', '口罩', '吃', '睡', '胖', '日常', '便当', '电影', '追剧', '生活', '吐槽', '工作', '学校', '考试', '交通', '忘戴口罩', '疫情在家', '疫情期间', '想回家']

B1_KW = ['国内', '本土', '省', '市', '县', '区', '全国', '确诊', '新增', '病例', '无症状', '阳性', '累计', '疫情地图', '最新疫情地图', '通报', '密接', '活动史', '分离到', '首例', '扩散', '疫情反弹']
B2_KW = ['美国', '日本', '英国', '意大利', '全球', '国际', '海外', 'WHO', '世卫', '欧美', '伦敦', '东京', '纽约', '巴黎', '特朗普', '白宫', '钻石公主', '朝日新闻']
B3_KW = ['复工', '复学', '开学', '隔离', '封闭', '管控', '返岗', '返校', '复产', '延迟', '通知', '政策', '战时', '停工', '居家隔离', '限行', '暂停', '恢复', '封城', '中风险', '高风险']
B4_KW = ['疫苗', '核酸', '病毒', '症状', '发烧', '咳嗽', '流感', '传播', '检测', '咽拭子', '气溶胶', '诊断', '防护', '口罩', '布洛芬', '治疗', '药', '医学', '科普', '肺炎', '感染', '医护', '病原', '医疗', '新冠', '新型冠状病毒']

P1_KW = ['医院', '床位', '收治', '住院', '门诊', '急诊', '医生', '看病', '治疗', '药', '救治', '救救', '病房', '诊所', '挂号', '检查', '就医', '隔离病房', '发热门诊']
P2_KW = ['口罩', '囤', '抢', '物资', '买不到', '缺货', '超市', '菜', '食品', '药店', '生活用品', '抢购', '纸巾', '消毒液', '洗手液', '防护服']
P3_KW = ['复工', '复学', '开学', '上班', '下班', '返工', '返校', '考试', '毕业', '上课', '工作', '上学', '延迟开学', '复工第一天', '要复工', '马上复工']
P4_KW = ['隔离', '封闭', '封城', '居家', '管控', '禁足', '不让出门', '封控', '小区封闭', '静默', '战时机制', '闭环', '限制出行']
P5_KW = ['慌', '害怕', '恐慌', '崩溃', '失眠', '疯', '难过', '想哭', '无助', '烦', '可怕', '惊慌', '焦虑', '心痛', '瑟瑟发抖', '绝望', '怕了', '怕死']
P6_KW = ['求助', '求问', '怎么办', '有谁知道', '谁知道', '哪里', '如何', '有没有', '为什么', '是不是', '能不能', '了解下', '问一下', '想知道', '请问', '求解']
P7_KW = ['发烧', '咳嗽', '头疼', '腹泻', '嗓子', '症状', '感染', '阳性', '密接', '疑似', '核酸', '检测', '不舒服', '头晕', '鼻塞', '扁桃体', '肺炎', '胸闷', '难受', '体温']
P8_KW = ['出门', '回家', '火车', '高铁', '飞机', '车站', '路上', '返程', '通勤', '春运', '交通', '旅游', '回程', '返乡', '出行', '坐车']

A_MEMORIAL_KW = ['李文亮', '缅怀', '致敬', '逆行者', '战疫', '希望与爱', '山河无恙', '加油', '守护', '祈祷', '感谢', '爱心', '捐款', '共抗疫情', '最美逆行者', '天使', '英雄']
A_HOPE_KW = ['希望', '相信', '好起来', '共克时艰', '平安', '祝福', '春天', '光亮', '会好的', '坚持', '挺过去', '加油', '别怕', '温暖', '重逢', '结束吧']
A_SATIRE_KW = ['哈哈', '呵呵', '笑死', '调侃', '嘲讽', '吐槽', '无语', '离谱', '离大谱', '梗', '戏精', '恶心', '真香', '整活', '喷', '开玩笑']

SYSTEM_NOISE_RE = [
    r'转发理由[:：].*$',
    r'//@.*$',
    r'http[s]?://\S+',
    r'www\.\S+',
    r'\[.*?\]',
    r'【.*?】',
    r'#.*?#',
    r'@\S+',
]

# ==============================
# 文本工具
# ==============================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.strip()
    for pat in SYSTEM_NOISE_RE:
        text = re.sub(pat, ' ', text)
    text = text.replace('\u200b', '').replace('\ufeff', '')
    text = ' '.join(text.split())
    return text.strip()


def is_useful_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 8:
        return False
    keep = ''.join(ch for ch in t if ch.isalnum() or '\u4e00' <= ch <= '\u9fff')
    return len(keep) >= 4


def kw_score(text: str, keywords) -> int:
    return sum(1 for kw in keywords if kw in text)


# ---------- 参考提示用的关键词分类（不决定最终标签） ----------
def classify_content_nature(text: str) -> str:
    scores = {
        'C': kw_score(text, HELP_KW),
        'A': kw_score(text, EMOTION_KW),
        'B': kw_score(text, NEWS_KW),
        'D': kw_score(text, DAILY_KW),
    }
    if scores['C'] > 0 and scores['C'] >= max(scores['A'], scores['B'], scores['D']):
        return 'C'
    if scores['B'] > 0 and scores['B'] >= max(scores['A'], scores['D']):
        return 'B'
    if scores['A'] > 0 and scores['A'] >= scores['D']:
        return 'A'
    return 'D'


def classify_nature_detail(text: str, nature: str) -> str:
    if nature == 'A':
        scores = {
            'A1': kw_score(text, EMOTION_KW),
            'A2': kw_score(text, A_MEMORIAL_KW),
            'A3': kw_score(text, A_HOPE_KW),
            'A4': kw_score(text, A_SATIRE_KW),
        }
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'A1'
    if nature == 'B':
        scores = {
            'B1': kw_score(text, B1_KW),
            'B2': kw_score(text, B2_KW),
            'B3': kw_score(text, B3_KW),
            'B4': kw_score(text, B4_KW),
        }
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'B1'
    if nature == 'C':
        return 'C1'
    return 'D1'


def classify_panic_type(text: str) -> str:
    scores = {
        'P1': kw_score(text, P1_KW),
        'P2': kw_score(text, P2_KW),
        'P3': kw_score(text, P3_KW),
        'P4': kw_score(text, P4_KW),
        'P5': kw_score(text, P5_KW),
        'P6': kw_score(text, P6_KW),
        'P7': kw_score(text, P7_KW),
        'P8': kw_score(text, P8_KW),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else ''


def tokenize_zh(text: str):
    text = clean_text(text)
    if not text:
        return ['_empty_']
    if jieba is not None:
        tokens = []
        for w in jieba.lcut(text):
            w = w.strip()
            if len(w) <= 1:
                continue
            if re.fullmatch(r'[0-9]+', w):
                continue
            tokens.append(w)
        return tokens if tokens else ['_empty_']
    tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z]{2,}|[0-9]{2,}', text)
    return tokens if tokens else ['_empty_']


def make_topic_label(nature_code, detail_code):
    if detail_code in NATURE_DETAIL_CN:
        return f'{detail_code} {NATURE_DETAIL_CN[detail_code]}'
    return f'{nature_code} {NATURE_CN.get(nature_code, nature_code)}'


# ==============================
# 文件读写工具
# ==============================
def read_csv_flexible(path: Path) -> pd.DataFrame:
    """兼容 utf-8-sig / gbk / utf-8 编码，兼容 Excel 保存的 xlsx。"""
    if str(path).lower().endswith('.xlsx'):
        return pd.read_excel(path)
    for enc in ('utf-8-sig', 'gbk', 'utf-8'):
        try:
            return pd.read_csv(path, comment='#', encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'无法解码编码表：{path}')


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding='utf-8-sig')


# ==============================
# Stage 1：聚类 + 生成编码模板
# ==============================
def run_stage1(args):
    print('=' * 80)
    print('topic_model.py — Stage 1：BERTopic 聚类 + 生成人工编码表')
    print('=' * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f'未找到 {INPUT_PATH}，请先运行 topic_sample.py 生成语料')

    print('\n【Step 1】读取语料并清洗 ...')
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    if 'content' not in df.columns:
        raise ValueError('topic_corpus.csv 中缺少 content 列')
    df = df.dropna(subset=['content']).copy()
    df['content'] = df['content'].astype(str).map(clean_text)
    df = df[df['content'].map(is_useful_text)].copy()
    df['doc_id'] = range(len(df))
    docs = df['content'].tolist()
    print(f'  有效文档数: {len(docs):,}')

    print('\n【Step 2】加载/计算 embedding ...')
    from sentence_transformers import SentenceTransformer
    embed_path = OUTPUT_DIR / 'embeddings.npy'
    embeddings = None
    if embed_path.exists() and not args.recompute_embeddings:
        cached = np.load(embed_path)
        if cached.shape[0] == len(docs):
            embeddings = cached
            print(f'  使用缓存: {embed_path} ({len(docs):,} 条)')
    if embeddings is None:
        embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = embed_model.encode(docs, show_progress_bar=True, batch_size=256)
        np.save(embed_path, embeddings)
        print(f'  embeddings 已缓存到 {embed_path}')

    print('\n【Step 3】训练 BERTopic ...')
    topic_model, topics, _ = train_topic_model(docs, embeddings)
    df['topic'] = topics
    n_topics = len([t for t in set(topics) if t != -1])
    print(f'  有效主题数: {n_topics}, 噪声文档数(-1): {int(sum(1 for t in topics if t == -1)):,}')

    print('\n【Step 4】保存主题信息 ...')
    topic_info = topic_model.get_topic_info()
    write_csv(topic_info, OUTPUT_DIR / 'topic_info.csv')

    topic_words = []
    for tid in sorted(set(topics)):
        if tid == -1:
            continue
        words = topic_model.get_topic(tid) or []
        top_words = ', '.join([f'{w}({s:.4f})' for w, s in words[:TOP_N_WORDS]])
        topic_words.append({'topic_id': tid, 'top_words': top_words})
    write_csv(pd.DataFrame(topic_words), OUTPUT_DIR / 'topic_keywords.csv')

    print('\n【Step 5】构建人工编码模板 topic_list.csv ...')
    template = build_template(topic_model, topics, df)
    write_template_with_instructions(template, OUTPUT_DIR / 'topic_list.csv')
    print(f'  模板已保存: {OUTPUT_DIR / "topic_list.csv"}')

    print('\n【Step 6】保存文档-主题底表 ...')
    doc_cols = ['doc_id', 'content', 'risk_group', 'risk_group_name', 'panic_prob', 'topic']
    doc_cols += [c for c in ['_source_file', 'user_id', 'created_at', 'crawl_time'] if c in df.columns]
    write_csv(df[doc_cols], OUTPUT_DIR / 'doc_topic_mapping.csv')

    rep_rows = []
    for tid in sorted(set(topics)):
        sub = df[df['topic'] == tid]
        if len(sub) == 0:
            continue
        samples = sub.sample(n=min(N_REP_DOCS, len(sub)), random_state=RANDOM_STATE)
        for _, row in samples.iterrows():
            rep_rows.append({
                'topic_id': tid,
                'risk_group_name': row['risk_group_name'],
                'panic_prob': row['panic_prob'],
                'content': row['content'][:220],
            })
    write_csv(pd.DataFrame(rep_rows), OUTPUT_DIR / 'topic_representative_docs.csv')

    print('\n【Step 7】保存模型 ...')
    topic_model.save(str(OUTPUT_DIR / 'bertopic_model'))

    print('\n' + '=' * 80)
    print('✅ Stage 1 完成')
    print(f'  - 主题数: {n_topics}')
    print(f'  - 编码模板: {OUTPUT_DIR / "topic_list.csv"}')
    print('  - 下一步：打开 topic_list.csv，人工填写 level1_code / level2_code / panic_code')
    print('  - 填写后运行: python topic_model.py --stage 2')
    print('=' * 80)


def build_template(topic_model, topics, df) -> pd.DataFrame:
    topic_info = topic_model.get_topic_info()
    topic_name_map = dict(zip(topic_info['Topic'], topic_info['Name']))
    rows = []
    for tid in sorted(set(topics), key=lambda t: (t == -1, t)):
        sub = df[df['topic'] == tid].copy()
        if len(sub) == 0:
            continue

        if tid == -1:
            name = '噪声/未归类'
            top_words = ''
        else:
            name = str(topic_name_map.get(tid, f'Topic_{tid}'))
            words = topic_model.get_topic(tid) or []
            top_words = ', '.join([f'{w}({s:.4f})' for w, s in words[:TOP_N_WORDS]])

        rep_docs = sub.sample(n=min(N_REP_DOCS, len(sub)), random_state=RANDOM_STATE)
        rep_text = ' || '.join([str(d)[:80] for d in rep_docs['content'].tolist()[:N_REP_DOCS]])

        probs = sub['panic_prob'].dropna()
        mean_panic = round(float(probs.mean()), 6) if len(probs) else np.nan
        median_panic = round(float(probs.median()), 6) if len(probs) else np.nan
        high_share = round(float((probs >= HIGH_PANIC_TH).mean()), 6) if len(probs) else np.nan

        # 关键词规则参考提示（仅参考）
        hint_text = f'{name} {top_words} {rep_text}'
        ref1 = classify_content_nature(hint_text)
        ref2 = classify_nature_detail(hint_text, ref1)
        refp = classify_panic_type(hint_text)

        # -1（噪声）默认编码为 N
        prefilled = {'level1_code': 'N', 'level2_code': '', 'panic_code': ''} if tid == -1 else {'level1_code': '', 'level2_code': '', 'panic_code': ''}

        rows.append({
            'topic_id': tid,
            'topic_name': name,
            'top_words': top_words,
            'representative_text': rep_text,
            'n_docs': len(sub),
            'share': round(len(sub) / len(df), 6),
            'mean_panic': mean_panic,
            'median_panic': median_panic,
            'high_panic_share': high_share,
            'ref_level1_code': ref1,
            'ref_level1_label': NATURE_CN.get(ref1, ''),
            'ref_level2_code': ref2,
            'ref_level2_label': NATURE_DETAIL_CN.get(ref2, ''),
            'ref_panic_code': refp,
            'ref_panic_label': PANIC_CN.get(refp, ''),
            **prefilled,
            'notes': '',
        })
    return pd.DataFrame(rows)


def write_template_with_instructions(df: pd.DataFrame, path: Path):
    buf = StringIO()
    for line in ENCODING_INSTRUCTIONS:
        buf.write(line + '\n')
    df.to_csv(buf, index=False)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(buf.getvalue())


# ==============================
# BERTopic 训练
# ==============================
def train_topic_model(docs, embeddings):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    umap_model = UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=N_COMPONENTS,
        metric='cosine',
        random_state=RANDOM_STATE,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

    def _build_and_fit(min_df, max_df):
        vectorizer_model = CountVectorizer(
            tokenizer=tokenize_zh,
            token_pattern=None,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
        )
        model = BERTopic(
            language='multilingual',
            top_n_words=TOP_N_WORDS,
            min_topic_size=MIN_CLUSTER_SIZE,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            calculate_probabilities=False,
            verbose=True,
        )
        topics, probs = model.fit_transform(docs, embeddings=embeddings)
        return model, topics, probs

    try:
        return _build_and_fit(min_df=2, max_df=0.95)
    except ValueError as e:
        if 'no terms remain' in str(e).lower():
            print('  ⚠️ CountVectorizer 词表过窄，自动降级到 min_df=1 / max_df=1.0')
            return _build_and_fit(min_df=1, max_df=1.0)
        raise


# ==============================
# Stage 2：读取人工编码 → 映射 → 分布矩阵
# ==============================
COL_ALIASES = {
    'topic_id': ['topic_id', '主题id', 'topic', '主题编号', '编号'],
    'level1_code': ['level1_code', '第一层编码', 'l1_code', '一级编码', '内容性质'],
    'level2_code': ['level2_code', '第二层编码', 'l2_code', '二级编码', '新闻细分'],
    'panic_code': ['panic_code', '第三层编码', 'p_code', '恐慌编码', '诉求'],
    'notes': ['notes', '备注', '说明'],
}


def pick_col(df: pd.DataFrame, candidates) -> str | None:
    colmap = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if str(cand).strip().lower() in colmap:
            return colmap[str(cand).strip().lower()]
    return None


def run_stage2(args):
    print('=' * 80)
    print('topic_model.py — Stage 2：人工编码校验 + 文档映射 + 分布矩阵')
    print('=' * 80)

    enc_path = Path(args.encoding) if args.encoding else OUTPUT_DIR / 'topic_list.csv'
    if not enc_path.exists():
        raise FileNotFoundError(f'未找到人工编码表 {enc_path}')

    enc = read_csv_flexible(enc_path)

    # 列名规范化（兼容 Excel 改列名的情况）
    rename = {}
    for key, aliases in COL_ALIASES.items():
        col = pick_col(enc, aliases)
        if col is not None and col != key:
            rename[col] = key
    enc = enc.rename(columns=rename)

    for req in ['topic_id', 'level1_code']:
        if req not in enc.columns:
            raise ValueError(f'编码表中缺少必要列 {req}，请检查文件: {enc_path}')

    print('\n【Step 1】校验人工编码 ...')
    enc, report = validate_encoding(enc)
    write_csv(report, OUTPUT_DIR / 'encoding_validation_report.csv')
    print(f'  校验报告: {OUTPUT_DIR / "encoding_validation_report.csv"}')

    n_error = int((report['severity'] == 'error').sum())
    n_warn = int((report['severity'] == 'warning').sum())
    if n_error > 0:
        detail = report[report['severity'] == 'error']['message'].head(20).tolist()
        raise ValueError(f'编码表存在 {n_error} 处错误，请修正后重跑：\n  ' + '\n  '.join(detail))
    print(f'  错误 0 处，警告 {n_warn} 处' + (f'（见 encoding_validation_report.csv）' if n_warn else ''))

    # 主题-人工编码映射表
    print('\n【Step 2】构建主题级映射表 ...')
    enc['topic_display_label'] = enc.apply(
        lambda r: make_topic_label(r['content_nature'], r['nature_detail']) if pd.notna(r['nature_detail']) and r['nature_detail'] else make_topic_label(r['content_nature'], ''), axis=1)
    tax_cols = ['topic_id', 'topic_name', 'topic_display_label', 'top_words', 'representative_text',
                'n_docs', 'share', 'mean_panic', 'median_panic', 'high_panic_share',
                'content_nature', 'content_nature_cn', 'nature_detail', 'nature_detail_cn',
                'panic_type', 'panic_type_cn']
    tax_cols = [c for c in tax_cols if c in enc.columns]
    tax = enc[tax_cols].sort_values('topic_id')
    write_csv(tax, OUTPUT_DIR / 'topic_taxonomy_mapping.csv')
    print(f'  已保存: {OUTPUT_DIR / "topic_taxonomy_mapping.csv"}')

    print('\n【Step 3】映射到文档 ...')
    doc_path = OUTPUT_DIR / 'doc_topic_mapping.csv'
    if not doc_path.exists():
        raise FileNotFoundError(f'未找到 {doc_path}，请先运行 --stage 1')
    doc = pd.read_csv(doc_path, low_memory=False)

    missing_topics = sorted(set(doc['topic'].unique()) - set(enc['topic_id'].unique()))
    if missing_topics:
        raise ValueError(f'文档中存在未编码的主题: {missing_topics}，请返回编码表补齐')

    merge_cols = ['content_nature', 'content_nature_cn', 'nature_detail', 'nature_detail_cn', 'panic_type', 'panic_type_cn']
    enc_map = enc[['topic_id'] + merge_cols].rename(columns={'topic_id': 'topic'})
    doc = doc.merge(enc_map, on='topic', how='left')

    base_cols = ['doc_id', 'content', 'risk_group', 'risk_group_name', 'panic_prob', 'topic']
    extra_cols = [c for c in doc.columns if c not in base_cols + merge_cols]
    doc_cols = base_cols + merge_cols + extra_cols
    write_csv(doc[doc_cols], OUTPUT_DIR / 'doc_taxonomy_mapping.csv')
    print(f'  已保存: {OUTPUT_DIR / "doc_taxonomy_mapping.csv"} ({len(doc):,} 行)')

    print('\n【Step 4】生成分布矩阵 ...')
    write_distributions(doc)

    print('\n【Step 5】编码结果摘要 ...')
    print_encoding_summary(enc)

    print('\n' + '=' * 80)
    print('✅ Stage 2 完成')
    print('  后续可直接运行 event_broadcast_analysis.py')
    print('=' * 80)


def validate_encoding(enc: pd.DataFrame):
    enc = enc.copy()
    enc['topic_id'] = pd.to_numeric(enc['topic_id'], errors='coerce')
    for col in ['level1_code', 'level2_code', 'panic_code']:
        if col not in enc.columns:
            enc[col] = ''
        enc[col] = enc[col].fillna('').astype(str).str.strip().str.upper()

    report_rows = []
    resolved = []

    for _, row in enc.iterrows():
        tid = row['topic_id']
        l1 = row['level1_code']
        l2 = row['level2_code']
        p = row['panic_code']

        if pd.isna(tid):
            report_rows.append({'topic_id': np.nan, 'severity': 'error', 'message': 'topic_id 为空'})
            continue

        if l1 == '':
            report_rows.append({'topic_id': tid, 'severity': 'error', 'message': f'topic {tid} 缺少 level1_code'})
            resolved.append({**row.to_dict(), 'level2_code': '', 'panic_code': ''})
            continue
        if l1 not in L1_RULES:
            report_rows.append({'topic_id': tid, 'severity': 'error', 'message': f'topic {tid} 的 level1_code={l1} 非法，应为 A/B/C/D/N'})
            resolved.append({**row.to_dict(), 'level2_code': '', 'panic_code': ''})
            continue

        default_l2, allowed_l2 = L1_RULES[l1]

        # 第二层校验
        if l1 == 'B' and l2 == '':
            report_rows.append({'topic_id': tid, 'severity': 'error', 'message': f'topic {tid} 是 B 类，必须填写 level2_code (B1/B2/B3/B4)'})
            l2_resolved = ''
        elif l2 != '' and l2 not in allowed_l2:
            report_rows.append({'topic_id': tid, 'severity': 'error', 'message': f'topic {tid} 的 level2_code={l2} 不属于 {l1} 类的合法取值 {sorted(allowed_l2)}'})
            l2_resolved = ''
        elif l2 == '' and l1 == 'A':
            report_rows.append({'topic_id': tid, 'severity': 'warning', 'message': f'topic {tid} 的 A 类未填 level2_code，已默认 A1（建议人工确认）'})
            l2_resolved = 'A1'
        elif l2 == '':
            l2_resolved = default_l2
        else:
            l2_resolved = l2

        # 第三层校验
        if p != '' and p not in VALID_PANIC:
            report_rows.append({'topic_id': tid, 'severity': 'error', 'message': f'topic {tid} 的 panic_code={p} 非法，应为 P1-P8 或留空'})
            p_resolved = ''
        else:
            p_resolved = p

        resolved.append({
            **row.to_dict(),
            'level2_code': l2_resolved,
            'panic_code': p_resolved,
        })

    enc = pd.DataFrame(resolved)
    enc['content_nature'] = enc['level1_code']
    enc['content_nature_cn'] = enc['level1_code'].map(NATURE_CN).fillna('')
    enc['nature_detail'] = enc['level2_code']
    enc['nature_detail_cn'] = enc['level2_code'].map(NATURE_DETAIL_CN).fillna('')
    enc['panic_type'] = enc['panic_code']
    enc['panic_type_cn'] = enc['panic_code'].map(PANIC_CN).fillna('')

    report = pd.DataFrame(report_rows)
    return enc, report


def write_distributions(doc: pd.DataFrame):
    valid = doc[doc['topic'] != -1].copy()

    # 主题分布
    topic_dist = pd.crosstab(valid['risk_group_name'], valid['topic'], normalize='index')
    write_csv(topic_dist, OUTPUT_DIR / 'topic_distribution_by_group.csv')
    topic_count = pd.crosstab(valid['risk_group_name'], valid['topic'])
    write_csv(topic_count, OUTPUT_DIR / 'topic_count_by_group.csv')

    # 一层框架
    l1_dist = pd.crosstab(doc['risk_group_name'], doc['content_nature_cn'], normalize='index')
    l1_dist = l1_dist.reindex(columns=[NATURE_CN['A'], NATURE_CN['B'], NATURE_CN['C'], NATURE_CN['D']], fill_value=0)
    write_csv(l1_dist, OUTPUT_DIR / 'taxonomy_level1_distribution_by_group.csv')
    l1_count = pd.crosstab(doc['risk_group_name'], doc['content_nature_cn'])
    l1_count = l1_count.reindex(columns=[NATURE_CN['A'], NATURE_CN['B'], NATURE_CN['C'], NATURE_CN['D']], fill_value=0)
    write_csv(l1_count, OUTPUT_DIR / 'taxonomy_level1_count_by_group.csv')

    # 二层框架（仅 B 类）
    bdf = doc[doc['content_nature'] == 'B'].copy()
    l2_order = [NATURE_DETAIL_CN['B1'], NATURE_DETAIL_CN['B2'], NATURE_DETAIL_CN['B3'], NATURE_DETAIL_CN['B4']]
    if len(bdf) > 0:
        l2_dist = pd.crosstab(bdf['risk_group_name'], bdf['nature_detail_cn'], normalize='index').reindex(columns=l2_order, fill_value=0)
        write_csv(l2_dist, OUTPUT_DIR / 'taxonomy_level2_distribution_by_group.csv')
        l2_count = pd.crosstab(bdf['risk_group_name'], bdf['nature_detail_cn']).reindex(columns=l2_order, fill_value=0)
        write_csv(l2_count, OUTPUT_DIR / 'taxonomy_level2_count_by_group.csv')
    else:
        write_csv(pd.DataFrame(columns=l2_order), OUTPUT_DIR / 'taxonomy_level2_distribution_by_group.csv')
        write_csv(pd.DataFrame(columns=l2_order), OUTPUT_DIR / 'taxonomy_level2_count_by_group.csv')

    # 恐慌诉求
    pdf = doc[doc['panic_type'] != ''].copy()
    p_order = [PANIC_CN[f'P{i}'] for i in range(1, 9)]
    if len(pdf) > 0:
        p_dist = pd.crosstab(pdf['risk_group_name'], pdf['panic_type_cn'], normalize='index').reindex(columns=p_order, fill_value=0)
        write_csv(p_dist, OUTPUT_DIR / 'panic_type_distribution_by_group.csv')
        p_count = pd.crosstab(pdf['risk_group_name'], pdf['panic_type_cn']).reindex(columns=p_order, fill_value=0)
        write_csv(p_count, OUTPUT_DIR / 'panic_type_count_by_group.csv')
    else:
        write_csv(pd.DataFrame(columns=p_order), OUTPUT_DIR / 'panic_type_distribution_by_group.csv')
        write_csv(pd.DataFrame(columns=p_order), OUTPUT_DIR / 'panic_type_count_by_group.csv')

    print('  topic_distribution_by_group.csv')
    print('  taxonomy_level1_distribution_by_group.csv')
    print('  taxonomy_level2_distribution_by_group.csv')
    print('  panic_type_distribution_by_group.csv')


def print_encoding_summary(enc: pd.DataFrame):
    print('  一层框架编码分布:')
    for code, label in NATURE_CN.items():
        n = int((enc['content_nature'] == code).sum())
        print(f'    {code} {label}: {n} 个主题')
    print('  二层框架编码分布（B 类）:')
    for code, label in [('B1', '国内疫情数据'), ('B2', '国际疫情动态'), ('B3', '政策干预信息'), ('B4', '医疗/科普信息')]:
        n = int((enc['nature_detail'] == code).sum())
        print(f'    {code} {label}: {n} 个主题')
    print('  恐慌诉求编码分布:')
    for i in range(1, 9):
        code = f'P{i}'
        n = int((enc['panic_type'] == code).sum())
        if n:
            print(f'    {code} {PANIC_CN[code]}: {n} 个主题')


def main():
    global MIN_CLUSTER_SIZE
    parser = argparse.ArgumentParser(description='topic_model.py — BERTopic 聚类 + 人工编码（两阶段）')
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True,
                        help='1=聚类并生成人工编码表；2=读取人工编码并生成分析结果')
    parser.add_argument('--encoding', type=str, default=None,
                        help='Stage 2 使用的人工编码文件路径（默认 topic_outputs/topic_list.csv）')
    parser.add_argument('--recompute-embeddings', action='store_true',
                        help='忽略缓存 embeddings.npy，重新计算')
    parser.add_argument('--min-cluster-size', type=int, default=40,
                        help='BERTopic 最小主题大小（默认 40）')
    args = parser.parse_args()
    MIN_CLUSTER_SIZE = args.min_cluster_size

    if args.stage == 1:
        run_stage1(args)
    else:
        run_stage2(args)


if __name__ == '__main__':
    main()
