import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
import sys
import random
import time
import warnings

warnings.filterwarnings('ignore')


# ==================== 双重输出：控制台 + 文件 ====================
class TeeOutput:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()


sys.stdout = TeeOutput('/root/autodl-tmp/topic_analysis_progress.log')

# ==================== 微博文本清洗 ====================
WEIBO_PATTERNS = [
    (r'转发理由[：:].*', ''),
    (r'转发微博', ''),
    (r'//@.*?:', ''),
    (r'@.*?\s', ''),
    (r'@.*?[，。！？,\.!?]', ''),
    (r'#.*?#', ''),
    (r'http[s]?://\S+', ''),
    (r'\[.*?\]', ''),
    (r'【.*?】', ''),
    (r'[a-zA-Z0-9]+', ''),
    (r'\s+', ' '),
]


def clean_weibo_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    for pattern, repl in WEIBO_PATTERNS:
        text = re.sub(pattern, repl, text)
    return text.strip()


def simple_cut(text):
    if pd.isna(text):
        return []
    text = clean_weibo_text(text)
    words = re.findall(r'[\u4e00-\u9fff]+', text)
    return [w for w in words if 2 <= len(w) <= 6 and len(set(w)) > 1]


# ==================== 配置 ====================
DAILY_STATS_PATH = '/root/autodl-tmp/daily_stats_detail_2019_2020.csv'
DATA_DIR = '/root/autodl-tmp/'
FILE_LIST = [
    'temp19.csv', 'temp1.csv', 'temp2.csv', 'temp3.csv',
    'temp4.csv', 'temp5.csv', 'temp6.csv', 'temp7.csv',
    'temp8.csv', 'temp9.csv', 'temp10.csv', 'temp11.csv', 'temp12.csv'
]
OUTPUT_PATH = '/root/autodl-tmp/panic_trend_tfidf_v4.png'
OUTPUT_FEATURES = '/root/autodl-tmp/tfidf_daily_features_v4.csv'

CHUNK_SIZE = 100000
SAMPLE_SIZE = 300000

start_time = time.time()


def print_progress(step, current, total, extra=""):
    """打印进度条"""
    pct = current / total * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = '█' * filled + '░' * (bar_len - filled)
    elapsed = time.time() - start_time
    print(f"\r[{step}] [{bar}] {pct:.1f}% {current}/{total} {extra} | {elapsed:.0f}s", end="", flush=True)


# ==================== 1. 读取日级统计 ====================
print("=" * 70)
print("COVID-19 PANIC TEXT FEATURE ANALYSIS - REALTIME PROGRESS")
print("=" * 70)
print(f"\n[1/5] Loading daily stats...")
daily_df = pd.read_csv(DAILY_STATS_PATH)
time_col = [c for c in daily_df.columns if c.lower() in ['created_at', 'date', 'time']][0]
daily_df['date'] = pd.to_datetime(daily_df[time_col])
daily_df = daily_df[(daily_df['date'] >= '2019-12-01') & (daily_df['date'] <= '2020-12-31')].reset_index(drop=True)

if 'panic_prob' in daily_df.columns and 'panic_rate' not in daily_df.columns:
    daily_df['panic_rate'] = daily_df['panic_prob'] * 100
if 'total_posts' not in daily_df.columns:
    daily_df['total_posts'] = daily_df.get('post_count', 0)

print(f"✓ Daily stats: {len(daily_df)} days | {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")

# ==================== 2. 第一次遍历：采样 + 统计 ====================
print(f"\n[2/5] PASS 1: Sampling + Statistics (Processing {len(FILE_LIST)} files)...")
daily_feature_accum = {}
sampled_texts = []
sampled_panic = []
sample_count = 0
total_posts = 0


def process_chunk_pass1(chunk):
    global sample_count, sampled_texts, sampled_panic

    content_col = 'content'
    panic_col = 'panic_prob'
    time_col = 'created_at'

    if content_col not in chunk.columns or time_col not in chunk.columns:
        return 0

    chunk['parsed_time'] = pd.to_datetime(chunk[time_col], errors='coerce')
    chunk = chunk[chunk['parsed_time'].notna()]
    chunk = chunk[(chunk['parsed_time'] >= '2019-12-01') & (chunk['parsed_time'] <= '2020-12-31')]
    n = len(chunk)
    if n == 0:
        return 0

    chunk['tokens'] = chunk[content_col].fillna('').astype(str).apply(
        lambda x: ' '.join(simple_cut(x))
    )

    for date, group in chunk.groupby(chunk['parsed_time'].dt.date):
        date_key = pd.Timestamp(date)
        if date_key not in daily_feature_accum:
            daily_feature_accum[date_key] = {
                'text_count': 0, 'total_length': 0, 'exclamation_sum': 0,
                'question_sum': 0, 'first_person_sum': 0, 'time_ref_sum': 0,
                'sentence_count_sum': 0, 'panic_prob_sum': 0,
            }
        acc = daily_feature_accum[date_key]
        acc['text_count'] += len(group)
        acc['total_length'] += group[content_col].str.len().sum()
        acc['exclamation_sum'] += group[content_col].str.count('！').sum()
        acc['question_sum'] += group[content_col].str.count('？').sum()
        acc['first_person_sum'] += (group[content_col].str.count('我') + group[content_col].str.count('我们')).sum()
        acc['time_ref_sum'] += (group[content_col].str.count('今天') + group[content_col].str.count('现在') + group[
            content_col].str.count('最近')).sum()
        acc['sentence_count_sum'] += group[content_col].apply(lambda x: len(re.split(r'[。！？\n]', str(x)))).sum()
        if panic_col in group.columns:
            acc['panic_prob_sum'] += group[panic_col].sum()

    if sample_count < SAMPLE_SIZE:
        needed = SAMPLE_SIZE - sample_count
        sample = chunk.sample(min(needed, n), random_state=42)
        sampled_texts.extend(sample['tokens'].tolist())
        if panic_col in sample.columns:
            sampled_panic.extend(sample[panic_col].tolist())
        else:
            sampled_panic.extend([0.5] * len(sample))
        sample_count += len(sample)

    return n


for file_idx, file_name in enumerate(FILE_LIST):
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"\n✗ Skip {file_name}: not found")
        continue

    # 估算总行数
    try:
        total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
    except:
        total_rows = 0

    file_posts = 0
    chunk_count = 0

    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE)):
        n = process_chunk_pass1(chunk)
        file_posts += n
        total_posts += n
        chunk_count += 1

        if total_rows > 0:
            processed_rows = min((i + 1) * CHUNK_SIZE, total_rows)
            print_progress(f"File {file_idx + 1}/{len(FILE_LIST)}", processed_rows, total_rows,
                           f"| {file_name} | Posts: {total_posts:,}")

    print(f"\n✓ {file_name}: {file_posts:,} posts | Sample: {len(sampled_texts):,}")

print(f"\n{'=' * 70}")
print(f"✓ PASS 1 COMPLETE: {total_posts:,} posts | Sample: {len(sampled_texts):,}")
print(f"{'=' * 70}")

# ==================== 3. 恐慌分层 TF-IDF ====================
print(f"\n[3/5] PASS 2: Panic-Stratified TF-IDF...")
print_progress("TF-IDF", 0, 3)

# 分层
high_panic = [(t, p) for t, p in zip(sampled_texts, sampled_panic) if p > 0.6 and len(t) > 5]
mid_panic = [(t, p) for t, p in zip(sampled_texts, sampled_panic) if 0.3 <= p <= 0.6 and len(t) > 5]
low_panic = [(t, p) for t, p in zip(sampled_texts, sampled_panic) if p < 0.3 and len(t) > 5]

print(f"\n  High panic: {len(high_panic):,} | Mid: {len(mid_panic):,} | Low: {len(low_panic):,}")

# 高恐慌过采样
if len(high_panic) < 5000:
    print(f"  ⚠ High panic too small, oversampling...")
    random.seed(42)
    high_panic = random.choices(high_panic, k=min(10000, len(high_panic) * 3))
    print(f"  ✓ After oversampling: {len(high_panic):,}")

print_progress("TF-IDF", 1, 3, "| Fitting high-panic vectorizer...")


# 计算TF-IDF
def get_tfidf_vectorizer(texts, max_features=300):
    if len(texts) < 10:
        return None, None
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=5, max_df=0.7, ngram_range=(1, 1))
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


high_texts = [t for t, p in high_panic]
high_vec, high_tfidf = get_tfidf_vectorizer(high_texts, 500)
high_word_scores = dict(
    zip(high_vec.get_feature_names_out(), np.array(high_tfidf.mean(axis=0)).flatten())) if high_vec else {}

print_progress("TF-IDF", 2, 3, "| Fitting low-panic vectorizer...")

low_texts = [t for t, p in low_panic]
low_vec, low_tfidf = get_tfidf_vectorizer(low_texts, 500)
low_word_scores = dict(
    zip(low_vec.get_feature_names_out(), np.array(low_tfidf.mean(axis=0)).flatten())) if low_vec else {}

print_progress("TF-IDF", 3, 3, "| Finding panic-specific words...")

# 找恐慌特有词
panic_specific = []
for word, score in sorted(high_word_scores.items(), key=lambda x: x[1], reverse=True):
    low_score = low_word_scores.get(word, 0)
    specificity = score / (low_score + 0.001)
    if specificity > 3.0 and score > 0.005 and len(word) <= 6:
        panic_specific.append((word, score, low_score, specificity))

print(f"\n{'=' * 70}")
print(f"✓ PASS 2 COMPLETE: {len(panic_specific)} panic-specific words found")
print(f"{'=' * 70}")
print(f"\nTop 15 panic words:")
for word, h, l, s in panic_specific[:15]:
    print(f"  {word:12s} high={h:.4f} low={l:.4f} ratio={s:.1f}")

# ==================== 4. 构建动态维度 ====================
print(f"\n[4/5] PASS 3: Building dynamic dimensions...")

DYNAMIC_THREAT = []
DYNAMIC_UNCERTAINTY = []
DYNAMIC_HELPLESSNESS = []
DYNAMIC_ANGER = []
DYNAMIC_SADNESS = []

threat_keywords = ['病毒', '感染', '传染', '肺炎', '确诊', '病例', '死亡', '医院', '隔离', '封城', '口罩', '买不到',
                   '武汉', '疫情', '爆发', '扩散', '严重', '危急', '发烧', '症状', '害怕', '恐惧', '恐慌', '惊慌',
                   '恐怖', '担心', '担忧', '焦虑', '不安', '紧张']
uncertainty_keywords = ['可能', '也许', '未知', '不确定', '不知道', '风险', '隐患', '万一', '或许', '大概', '说不定']
helplessness_keywords = ['没办法', '无能为力', '只能', '只好', '被迫', '不得不', '绝望', '无望', '无助', '等死', '被困',
                         '崩溃', '束手无策', '走投无路']
anger_keywords = ['愤怒', '气愤', '抗议', '反对', '不满', '谴责', '问责', '追责', '甩锅', '隐瞒', '撒谎', '欺骗',
                  '失职', '渎职']
sadness_keywords = ['难过', '悲伤', '痛心', '哀悼', '悼念', '牺牲', '离世', '去世', '逝世', '走好', '安息', '泪目',
                    '哭泣', '流泪']

for word, score, _, _ in panic_specific:
    matched = False
    for kw in threat_keywords:
        if kw in word:
            DYNAMIC_THREAT.append(word)
            matched = True
            break
    if matched: continue

    for kw in uncertainty_keywords:
        if kw in word:
            DYNAMIC_UNCERTAINTY.append(word)
            matched = True
            break
    if matched: continue

    for kw in helplessness_keywords:
        if kw in word:
            DYNAMIC_HELPLESSNESS.append(word)
            matched = True
            break
    if matched: continue

    for kw in anger_keywords:
        if kw in word:
            DYNAMIC_ANGER.append(word)
            matched = True
            break
    if matched: continue

    for kw in sadness_keywords:
        if kw in word:
            DYNAMIC_SADNESS.append(word)
            matched = True
            break
    if matched: continue

    if score > 0.02:
        DYNAMIC_THREAT.append(word)
    else:
        DYNAMIC_UNCERTAINTY.append(word)

FINAL_DIMENSIONS = {
    'threat': list(
        set(DYNAMIC_THREAT + ['病毒', '新冠', '肺炎', '感染', '传染', '确诊', '病例', '死亡', '医院', '隔离', '封城',
                              '武汉', '疫情', '害怕', '恐惧', '恐慌', '担心', '焦虑'])),
    'uncertainty': list(
        set(DYNAMIC_UNCERTAINTY + ['可能', '也许', '未知', '不确定', '不知道', '风险', '隐患', '万一'])),
    'control': ['政府', '措施', '防控', '相信', '希望', '好转', '治愈', '胜利', '疫苗', '核酸', '检测', '防护', '有效',
                '成功'],
    'helplessness': list(
        set(DYNAMIC_HELPLESSNESS + ['没办法', '绝望', '无助', '等死', '被困', '崩溃', '束手无策', '走投无路'])),
    'anger': list(set(DYNAMIC_ANGER)),
    'sadness': list(set(DYNAMIC_SADNESS)),
}

FINAL_DIMENSIONS = {k: v for k, v in FINAL_DIMENSIONS.items() if v}

print(f"✓ Dimensions: {', '.join([f'{k}({len(v)})' for k, v in FINAL_DIMENSIONS.items()])}")

# 预编译正则
final_patterns = {}
for dim, words in FINAL_DIMENSIONS.items():
    sorted_words = sorted(set(words), key=len, reverse=True)[:100]
    final_patterns[dim] = re.compile('|'.join(re.escape(w) for w in sorted_words))

# ==================== 5. 第二次遍历：动态维度映射 ====================
print(f"\n[5/5] PASS 4: Mapping dimensions to daily level...")

daily_dynamic = {dim: {} for dim in FINAL_DIMENSIONS.keys()}
daily_counts = {}

for file_idx, file_name in enumerate(FILE_LIST):
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        continue

    try:
        total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
    except:
        total_rows = 0

    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE)):
        if 'content' not in chunk.columns or 'created_at' not in chunk.columns:
            continue

        chunk['parsed_time'] = pd.to_datetime(chunk['created_at'], errors='coerce')
        chunk = chunk[chunk['parsed_time'].notna()]
        chunk = chunk[(chunk['parsed_time'] >= '2019-12-01') & (chunk['parsed_time'] <= '2020-12-31')]

        if len(chunk) == 0:
            continue

        for date, group in chunk.groupby(chunk['parsed_time'].dt.date):
            date_key = pd.Timestamp(date)
            if date_key not in daily_counts:
                daily_counts[date_key] = 0
                for dim in FINAL_DIMENSIONS.keys():
                    daily_dynamic[dim][date_key] = 0

            day_texts = group['content'].fillna('').astype(str).apply(clean_weibo_text)
            daily_counts[date_key] += len(day_texts)

            for dim, pattern in final_patterns.items():
                matches = day_texts.str.count(pattern)
                daily_dynamic[dim][date_key] += matches.sum()

        if total_rows > 0:
            processed = min((i + 1) * CHUNK_SIZE, total_rows)
            print_progress(f"Map {file_idx + 1}/{len(FILE_LIST)}", processed, total_rows, f"| {file_name}")

print(f"\n{'=' * 70}")
print(f"✓ PASS 4 COMPLETE")
print(f"{'=' * 70}")

# 归一化
for date, count in daily_counts.items():
    if count > 0:
        for dim in FINAL_DIMENSIONS.keys():
            daily_dynamic[dim][date] /= count

# ==================== 6. 合并 + 可视化 ====================
print(f"\n[FINAL] Merging and plotting...")

daily_text_features = []
for date, acc in sorted(daily_feature_accum.items()):
    count = acc['text_count']
    if count == 0:
        continue
    row = {
        'date': date,
        'text_length_mean': acc['total_length'] / count,
        'exclamation_ratio': acc['exclamation_sum'] / count,
        'question_ratio': acc['question_sum'] / count,
        'first_person_ratio': acc['first_person_sum'] / count,
        'time_ref_ratio': acc['time_ref_sum'] / count,
        'sentence_count_mean': acc['sentence_count_sum'] / count,
        'panic_prob_mean': acc['panic_prob_sum'] / count,
    }
    daily_text_features.append(row)

text_feature_df = pd.DataFrame(daily_text_features)

dynamic_df_data = []
for date in sorted(daily_counts.keys()):
    row = {'date': date}
    for dim in FINAL_DIMENSIONS.keys():
        row[f'{dim}_density'] = daily_dynamic[dim].get(date, 0)
    dynamic_df_data.append(row)
dynamic_df = pd.DataFrame(dynamic_df_data)

daily_merged = pd.merge(daily_df, text_feature_df, on='date', how='left')
daily_merged = pd.merge(daily_merged, dynamic_df, on='date', how='left')
daily_merged = daily_merged.fillna(0)

if daily_merged['panic_rate'].max() < 1:
    daily_merged['panic_rate'] = daily_merged['panic_rate'] * 100

daily_merged.to_csv(OUTPUT_FEATURES, index=False)

# 统计
print(f"\nDimension peak dates:")
for dim in FINAL_DIMENSIONS.keys():
    col = f'{dim}_density'
    peak_idx = daily_merged[col].idxmax()
    print(f"  {dim:12s} peak={daily_merged[col].max():.3f} on {daily_merged.loc[peak_idx, 'date'].date()}")

# 可视化
COLOR_PANIC = '#3A506B'
COLOR_VOLUME = '#C97B63'
COLOR_GRID = '#E8E8E8'

n_dims = len(FINAL_DIMENSIONS)
CMAP_L1 = LinearSegmentedColormap.from_list('l1',
                                            ['#E8F4F8', '#A8D8EA', '#F9B4B4', '#E74C3C', '#8B0000'] if n_dims <= 5 else
                                            ['#E8F4F8', '#A8D8EA', '#AA96DA', '#FCBAD3', '#FFFFD2', '#FF9A8B',
                                             '#FF6A88'])

data = daily_merged.copy()
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(5, 1, height_ratios=[3, 1.5, 1.2, 1.2, 0.3], hspace=0.08)

# 主图
ax_main = fig.add_subplot(gs[0])
ax_main.plot(data['date'], data['panic_rate'], color=COLOR_PANIC, linewidth=2, alpha=0.9, label='Panic Rate (%)',
             zorder=5)
ax_main.fill_between(data['date'], data['panic_rate'], alpha=0.12, color=COLOR_PANIC)
ax_main.set_ylabel('Panic Rate (%)', color=COLOR_PANIC, fontsize=12, fontweight='bold')
ax_main.tick_params(axis='y', labelcolor=COLOR_PANIC)
ax_main.set_ylim(0, data['panic_rate'].max() * 1.25)

ax_vol = ax_main.twinx()
ax_vol.bar(data['date'], data['total_posts'], alpha=0.25, color=COLOR_VOLUME, width=1.0, label='Daily Post Count',
           zorder=1)
ax_vol.set_ylabel('Daily Post Count', color=COLOR_VOLUME, fontsize=12, fontweight='bold')
ax_vol.tick_params(axis='y', labelcolor=COLOR_VOLUME)

EVENTS = {
    '2020-01-20': ('Dr. Zhong Nanshan\nconfirms H2H', '#C97B63'),
    '2020-01-23': ('Wuhan Lockdown', '#3A506B'),
    '2020-03-24': ('Hubei reopening\nannouncement', '#6B8E23'),
    '2020-04-08': ('Wuhan officially\nreopens', '#8B4513'),
}
for date_str, (label, color) in EVENTS.items():
    event_date = pd.to_datetime(date_str)
    ax_main.axvline(x=event_date, color=color, linestyle='--', linewidth=1.8, alpha=0.7, zorder=3)
    ax_main.annotate(label, xy=(event_date, ax_main.get_ylim()[1] * 0.92), xytext=(12, 0), textcoords='offset points',
                     fontsize=9, color=color, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.85, linewidth=1.2))

ax_main.set_title('COVID-19 Panic Rate & TF-IDF Dynamic Dimensions (v4)\n(Dec 2019 - Dec 2020)',
                  fontsize=15, fontweight='bold', pad=15, color='#2C3E50')
ax_main.grid(True, alpha=0.25, linestyle='-', color=COLOR_GRID, zorder=0)
lines1, labels1 = ax_main.get_legend_handles_labels()
lines2, labels2 = ax_vol.get_legend_handles_labels()
ax_main.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fancybox=True, shadow=True,
               fontsize=10)
ax_main.set_xticklabels([])
ax_main.tick_params(axis='x', length=0)

# L1: 动态维度
ax_l1 = fig.add_subplot(gs[1], sharex=ax_main)
dim_cols = [f'{d}_density' for d in FINAL_DIMENSIONS.keys()]
dim_labels = [d.capitalize() for d in FINAL_DIMENSIONS.keys()]

for feat in dim_cols:
    if feat not in data.columns:
        data[feat] = 0

l1_data = data[dim_cols].values.T
l1_data_norm = (l1_data - l1_data.min(axis=1, keepdims=True)) / (
            l1_data.max(axis=1, keepdims=True) - l1_data.min(axis=1, keepdims=True) + 1e-10)

im1 = ax_l1.imshow(l1_data_norm, aspect='auto', cmap=CMAP_L1, interpolation='nearest',
                   extent=[mdates.date2num(data['date'].min()), mdates.date2num(data['date'].max()), -0.5,
                           len(dim_cols) - 0.5])
ax_l1.set_yticks(range(len(dim_cols)))
ax_l1.set_yticklabels(dim_labels, fontsize=9, fontweight='bold')
ax_l1.set_ylabel(f'L1: TF-IDF Dimensions ({n_dims} dims)', fontsize=10, fontweight='bold', color='#2C3E50')
ax_l1.tick_params(axis='x', length=0)
ax_l1.set_xticklabels([])

# L2-L3 省略，与之前相同...

# X轴
ax_x = fig.add_subplot(gs[4], sharex=ax_main)
ax_x.set_xlim([pd.to_datetime('2019-12-01'), pd.to_datetime('2021-01-05')])
ticks = pd.date_range(start='2019-12-01', end='2020-12-31', freq='MS')
ax_x.set_xticks(ticks)
ax_x.set_xticklabels([d.strftime('%b %Y') for d in ticks], rotation=45, ha='right', fontsize=9)
ax_x.tick_params(axis='y', length=0)
ax_x.set_yticklabels([])
for spine in ['top', 'right', 'left']:
    ax_x.spines[spine].set_visible(False)
ax_x.spines['bottom'].set_color('#CCCCCC')

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

elapsed = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"✓ ALL DONE in {elapsed:.0f}s ({elapsed / 60:.1f}min)")
print(f"  Output: {OUTPUT_PATH}")
print(f"  Features: {OUTPUT_FEATURES}")
print(f"  Peak panic: {data['panic_rate'].max():.2f}% on {data.loc[data['panic_rate'].idxmax(), 'date'].date()}")
print(f"{'=' * 70}")

# 恢复输出
sys.stdout = sys.__stdout__
print(f"\nLog saved to: /root/autodl-tmp/topic_analysis_progress.log")