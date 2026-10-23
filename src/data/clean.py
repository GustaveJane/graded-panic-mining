#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERTopic主题深度清洗 v5增强版
目标：
1. 基于fp_v4跨层级合并同一源内容
2. 新闻转发六分类：国内疫情/国际疫情/政策发布/数据统计/科普辟谣/社会事件
3. 分离新闻正文与转发评论，提取恐慌情绪
4. 最终100个独立主题，层级间不重复
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
INPUT_FILE = '/root/autodl-tmp/output_bertopic_cleaned_v4/完整清洗记录_v4.csv'
OUTPUT_DIR = '/root/autodl-tmp/output_bertopic_cleaned_v4/'
FINAL_TOPICS_TARGET = 100

# ==================== 情绪模式定义 ====================

EMOTION_PATTERNS = {
    '害怕/恐惧': [
        r'害怕|恐惧|恐慌|吓死|瑟瑟发抖|慌|担心|焦虑|紧张|惊恐|畏惧|胆怯|后怕',
        r'太可怕了|太恐怖|太吓人|我的天|天呐|天啊|卧槽|我靠|妈呀|救命|完了',
        r'不敢|不敢去|不敢出门|不敢看|不敢想|不敢碰|不敢吃|不敢用',
        r'瑟瑟发抖|抖|哆嗦|发颤|腿软|腿抖|手软|手抖|心跳加速|心慌|心悸'
    ],
    '愤怒/不满': [
        r'气死|生气|愤怒|恼火|愤怒|无语|恶心|痛恨|恨|讨厌|烦|厌恶|憎恶',
        r'妈的|卧槽|我靠|他妈的|傻逼|煞笔|脑残|智障|混蛋|垃圾|废物|不要脸',
        r'凭什么|为什么|怎么这样|太不公平|太过分了|太离谱|太荒唐|太可笑',
        r'抗议|反对|抵制|谴责|声讨|批判|抨击|揭露|曝光|举报|投诉|上访'
    ],
    '悲伤/绝望': [
        r'难过|悲伤|想哭|泪目|心疼|悲哀|绝望|崩溃|无助|无奈|无力|心累',
        r'太惨了|太可怜|太心酸|太心疼|太悲哀|太绝望|太无助|太无奈|太无力',
        r'哭|流泪|眼泪|哭泣|痛哭|大哭|嚎啕大哭|泪流满面|泣不成声|哽咽',
        r'绝望|没希望|没救了|完了|结束了|完蛋了|死定了|活不下去了|不想活了'
    ],
    '希望/祈祷': [
        r'求求|拜托|保佑|祈祷|希望|加油|挺住|坚持|挺过去|战胜|克服|度过',
        r'会好的|没事的|不要怕|别担心|别慌|冷静|稳住|坚持住|撑住|扛住',
        r'相信|信任|信心|信念|坚信|确信|肯定|一定|必然|绝对|毫无疑问',
        r'祝福|祝愿|祈福|祷告|许愿|心想事成|万事如意|平安|健康|康复|痊愈'
    ],
    '质疑/怀疑': [
        r'不信|不相信|怀疑|质疑|疑问|疑惑|困惑|不解|不明白|不清楚|不知道',
        r'真的吗|假的吧|不可能|怎么会|为什么|凭什么|怎么回事|什么情况',
        r'隐瞒|掩盖|粉饰|美化|造假|撒谎|欺骗|蒙蔽|忽悠|糊弄|敷衍|搪塞',
        r'真相|事实|实情|内幕|黑幕|秘密|隐情|猫腻|水分|泡沫|虚假|谎言'
    ],
    '冷漠/麻木': [
        r'无所谓|不在乎|不关心|不在意|不重要|没关系|随便|都行|爱咋咋地',
        r'习惯了|麻木|麻痹|迟钝|无感|没感觉|没反应|无动于衷|置若罔闻',
        r'看淡|看透|看开|放下|释然|超脱|佛系|随缘|顺其自然|听天由命',
        r'关我屁事|与我无关|懒得管|不想管|不管了|随他去|爱谁谁'
    ]
}

# ==================== 新闻分类规则 ====================

NEWS_PATTERNS = {
    '国内疫情': [
        r'武汉|湖北|黄冈|孝感|襄阳|宜昌|荆州|黄石|鄂州|荆门|咸宁|随州|恩施|十堰|天门|仙桃|潜江',
        r'确诊.*例|新增.*例|死亡病例|治愈出院|封城|解封|复工|雷神山|火神山|方舱',
        r'钟南山|张文宏|李兰娟|卫健委|疾控中心|医院|收治|核酸检测|阳性|阴性|隔离|密切接触者'
    ],
    '国际疫情': [
        r'美国|意大利|西班牙|德国|法国|英国|日本|韩国|印度|巴西|俄罗斯|澳大利亚|加拿大',
        r'特朗普|拜登|约翰逊|默克尔|马克龙|菅义伟|莫迪|普京|文在寅|安倍晋三|鲍里斯|Boris',
        r'全球确诊|海外疫情|境外输入|世卫组织|WHO|欧盟|G7|G20|联合国|群体免疫|封国|旅行禁令'
    ],
    '政策发布': [
        r'国务院|党中央|政治局|常委会|指挥部|领导小组|办公厅|发布|印发|出台|通知|公告|决定|措施|方案',
        r'一级响应|二级响应|三级响应|防控指挥部|防疫指挥部|应急指挥部',
        r'延长假期|延迟开学|推迟复工|交通管制|封闭管理|封控|管控|健康码|行程码|通行码|复工证明'
    ],
    '数据统计': [
        r'累计确诊|累计死亡|治愈率|死亡率|病死率|感染率|传播率|R0|疫情地图|数据更新|实时数据|统计图表',
        r'破万|破十万|破百万|里程碑|超过.*万|突破.*万|达.*例|至.*例|新增.*例|现有.*例|重症.*例'
    ],
    '科普辟谣': [
        r'辟谣|谣言|假消息|不实信息|澄清|真相|假的|骗人的|别信|不要相信|科普|知识|预防|防护|口罩|消毒|洗手|通风',
        r'专家解读|医生提醒|科学|研究表明|实验证明|临床研究|医学|双黄连|板蓝根|熏醋|喝酒|吸烟|吃大蒜|维生素C'
    ],
    '社会事件': [
        r'李文亮|吹哨人|医生去世|医护牺牲|志愿者|捐款|捐赠|物资|防护服|N95|红十字会|慈善|捐赠物资|分配|调配',
        r'火神山医院|雷神山医院|方舱医院|建设|施工|基建狂魔|中国速度|歧视|地域歧视|武汉人|湖北人|排外|驱赶'
    ]
}


# ==================== 函数定义 ====================

def extract_key_sentence(text, max_len=80):
    """从文本中提取关键句作为源内容标识"""
    if not text or pd.isna(text):
        return "EMPTY"
    text = str(text)
    text = re.sub(r'转发理由.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'转发微博', '', text)
    text = re.sub(r'Repost', '', text, flags=re.IGNORECASE)
    key = text[:max_len].strip()
    key = re.sub("[，。！？、；：“”‘’（）【】\\s]", "", key)
    return key if key else "EMPTY"


def separate_content_and_comment(sample_docs):
    """分离新闻正文和转发评论"""
    if pd.isna(sample_docs) or not sample_docs:
        return "", "", {}, {}

    text = str(sample_docs)
    repost_markers = ['转发理由', '转发微博', 'Repost', '轉發微博', '全文转发理由']

    news_body = text
    comment = ""

    for marker in repost_markers:
        if marker in text:
            parts = text.split(marker, 1)
            news_body = parts[0].strip()
            comment = parts[1].strip() if len(parts) > 1 else ""
            break

    if not comment and '转发' in text:
        idx = text.find('转发')
        if idx > 0:
            news_body = text[:idx].strip()
            comment = text[idx:].strip()

    all_emotions = extract_emotions(news_body + " " + comment)
    comment_emotions = extract_emotions(comment) if comment else {}

    return news_body, comment, all_emotions, comment_emotions


def extract_emotions(text):
    """提取文本中的情绪信号"""
    if not text or pd.isna(text):
        return {}
    text = str(text)
    emotions = {}
    for emotion_type, patterns in EMOTION_PATTERNS.items():
        matches = []
        for p in patterns:
            matches.extend(re.findall(p, text))
        if matches:
            emotions[emotion_type] = len(matches)
    return emotions


def is_news_content(text, top_words):
    """判断是否是新闻内容"""
    combined = str(text) + ' ' + str(top_words)

    news_markers = ['刚刚', '突发', '据报道', '据央视', '据新华社', '据人民日报',
                    '通报', '公告', '发布会', '确诊', '新增', '死亡病例',
                    '治愈出院', '核酸检测', '阳性', '隔离', '封城']
    has_numbers = bool(re.search(r'\d+例|\d+人|\d+万|\d+千', combined))
    has_location = bool(re.search(r'武汉|湖北|北京|上海|广州|深圳|杭州|成都|重庆|西安|南京', combined))
    has_time_event = bool(re.search(r'今日|今天|昨日|昨天|近日|目前|现在|已经|正在', combined))
    has_repost = '转发理由' in combined or '转发微博' in combined
    has_source = bool(re.search(r'据.*?报道|据.*?消息|据.*?通报|据.*?公告', combined))

    score = sum([
        any(m in combined for m in news_markers) * 2,
        has_numbers * 2,
        has_location * 1,
        has_time_event * 1,
        has_repost * 2,
        has_source * 2
    ])

    return score >= 4


def classify_news_type(text, top_words):
    """新闻细分分类"""
    combined = str(text) + ' ' + str(top_words)

    intl_keywords = ['美国', '意大利', '西班牙', '德国', '法国', '英国', '日本', '韩国',
                     '印度', '巴西', '俄罗斯', '特朗普', '拜登', '约翰逊', '默克尔',
                     '全球', '海外', '境外输入', 'WHO', '世卫组织', '欧盟']
    if any(k in combined for k in intl_keywords):
        return '国际疫情'

    policy_keywords = ['国务院', '通知', '公告', '决定', '措施', '方案', '指挥部',
                       '一级响应', '二级响应', '延长假期', '延迟开学', '推迟复工',
                       '健康码', '通行码', '复工证明']
    if any(k in combined for k in policy_keywords):
        return '政策发布'

    data_keywords = ['累计确诊', '累计死亡', '治愈率', '死亡率', '疫情地图',
                     '数据更新', '统计图表', '破万', '破十万', '新增.*例']
    if any(k in combined for k in data_keywords):
        return '数据统计'

    science_keywords = ['辟谣', '谣言', '科普', '知识', '预防', '防护', '专家解读',
                        '医生提醒', '科学', '研究表明', '双黄连', '板蓝根']
    if any(k in combined for k in science_keywords):
        return '科普辟谣'

    social_keywords = ['李文亮', '医护', '志愿者', '捐款', '捐赠', '物资', '红十字会',
                       '火神山', '雷神山', '歧视', '地域歧视', '复工', '复产', '经济']
    if any(k in combined for k in social_keywords):
        return '社会事件'

    return '国内疫情'


def determine_primary_category(source_key, all_docs, all_comments, original_categories):
    """基于内容特征确定主要类别"""
    combined_all = ' '.join(all_docs + all_comments)

    if re.search(r'孕期|医院|求助|救治|收治|床位|透析|ICU|求医|看病|急诊', combined_all):
        return '医疗资源/求助'
    elif re.search(r'复工|开学|上班|返校|复课|返岗|工作|课程|学业', combined_all):
        return '复工焦虑'
    elif re.search(r'口罩|买菜|物资|超市|囤货|断供|短缺|粮食|食物', combined_all):
        return '生活物资/买菜'
    elif re.search(r'害怕|担心|焦虑|恐惧|吓死|慌|紧张|惊恐|崩溃|绝望', combined_all):
        return '情绪宣泄'
    elif re.search(r'隔离|居家|封城|封控|管控|足不出户|宅家|封闭|居家办公', combined_all):
        return '生活影响'
    elif re.search(r'出行|交通|航班|火车|地铁|公交|自驾|返乡|返程|旅游|旅行', combined_all):
        return '出行/交通'
    else:
        cat_counts = Counter(c for c in original_categories if c and c != '未分类')
        return cat_counts.most_common(1)[0][0] if cat_counts else '未分类'


def similar(a, b):
    """计算字符串相似度"""
    if not a or not b:
        return 0
    return SequenceMatcher(None, str(a), str(b)).ratio()


# ==================== 主流程 ====================

def main():
    print("=" * 70)
    print("BERTopic主题深度清洗 v5增强版")
    print("核心：跨层级源内容合并 + 新闻细分 + 情绪提取")
    print("=" * 70)

    # Step 0: 加载数据
    print("\n" + "=" * 70)
    print("Step 0: 加载数据")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)
    print(f"总记录数: {len(df)}")
    print(f"层级分布: {df['Tier'].value_counts().to_dict()}")

    # Step 1: 基于fp_v4的源内容识别
    print("\n" + "=" * 70)
    print("Step 1: 基于fp_v4的源内容识别")
    print("=" * 70)

    kept_df = df[df['clean_status'] != '删除'].copy()
    print(f"保留主题数: {len(kept_df)}")

    source_groups = defaultdict(list)

    for idx, row in kept_df.iterrows():
        topic_id = row['Topic_ID']
        tier = row['Tier']
        fp_v4 = str(row.get('fp_v4', ''))
        fingerprint = str(row.get('fingerprint', ''))
        valid = row.get('valid_fp', False)
        sample_docs = str(row.get('Sample_Docs', ''))

        if fp_v4 and fp_v4 != 'nan' and len(fp_v4) > 5:
            source_key = fp_v4[:120]
        elif valid and fingerprint and fingerprint != 'nan' and len(fingerprint) > 10:
            source_key = fingerprint[:120]
        else:
            source_key = extract_key_sentence(sample_docs)

        source_groups[source_key].append({
            'original_idx': idx,
            'Topic_ID': topic_id,
            'Tier': tier,
            'Count': row['Count'],
            'Sample_Docs': sample_docs,
            'Top_Words': str(row.get('Top_Words', '')),
            'Primary_Category_v3': str(row.get('Primary_Category_v3', '未分类')),
            'clean_status': str(row.get('clean_status', '保留')),
            'fp_v4': fp_v4
        })

    print(f"识别到 {len(source_groups)} 个源内容组")

    # Step 2: 新闻识别与分类
    print("\n" + "=" * 70)
    print("Step 2: 新闻识别与细分")
    print("=" * 70)

    classified_groups = {}

    for source_key, items in source_groups.items():
        combined_text = ' '.join([item['Sample_Docs'][:400] for item in items[:3]])
        combined_words = ' '.join([item['Top_Words'] for item in items])

        is_news = is_news_content(combined_text, combined_words)

        if is_news:
            news_type = classify_news_type(combined_text, combined_words)
        else:
            news_type = None

        classified_groups[source_key] = {
            'items': items,
            'is_news': is_news,
            'news_type': news_type
        }

    news_count = sum(1 for info in classified_groups.values() if info['is_news'])
    non_news_count = sum(1 for info in classified_groups.values() if not info['is_news'])
    print(f"新闻主题: {news_count} 个")
    print(f"非新闻主题: {non_news_count} 个")

    # Step 3: 分离内容与情绪提取
    print("\n" + "=" * 70)
    print("Step 3: 分离新闻正文与转发评论，提取情绪")
    print("=" * 70)

    # Step 4: 构建最终主题
    print("\n" + "=" * 70)
    print("Step 4: 构建最终主题")
    print("=" * 70)

    final_topics = []

    for source_key, info in classified_groups.items():
        items = info['items']
        is_news = info['is_news']
        news_type = info['news_type']

        all_docs = []
        all_comments = []
        all_emotions = defaultdict(int)
        all_comment_emotions = defaultdict(int)
        tiers = []
        categories = []
        total_count = 0

        for item in items:
            news_body, comment, emotions, comment_emotions = separate_content_and_comment(item['Sample_Docs'])

            all_docs.append(news_body)
            if comment:
                all_comments.append(comment)

            for k, v in emotions.items():
                all_emotions[k] += v
            for k, v in comment_emotions.items():
                all_comment_emotions[k] += v

            tiers.append(item['Tier'])
            categories.append(item['Primary_Category_v3'])
            total_count += item['Count']

        tier_counts = Counter(tiers)
        primary_tier = tier_counts.most_common(1)[0][0]

        primary_category = determine_primary_category(source_key, all_docs, all_comments, categories)

        if all_comments:
            rep_text = max(all_comments, key=len)
        else:
            rep_text = max(all_docs, key=len)

        top_emotions = ' | '.join([f"{k}({v})" for k, v in sorted(all_emotions.items(), key=lambda x: -x[1])[:5]])
        top_comment_emotions = ' | '.join(
            [f"{k}({v})" for k, v in sorted(all_comment_emotions.items(), key=lambda x: -x[1])[:5]])

        if is_news:
            topic_name = f"[{news_type}] {source_key[:40]}"
        else:
            topic_name = f"[{primary_category}-{primary_tier}] {source_key[:40]}"

        final_topics.append({
            'Topic_Name': topic_name,
            'Source_Key': source_key[:100],
            'Is_News': is_news,
            'News_Type': news_type,
            'Primary_Tier': primary_tier,
            'Primary_Category': primary_category,
            'Tier_Distribution': dict(tier_counts),
            'Total_Doc_Count': total_count,
            'Original_Topic_Count': len(items),
            'Representative_Text': rep_text[:500],
            'News_Body_Sample': all_docs[0][:300] if all_docs else '',
            'Comments_Sample': ' || '.join(all_comments[:3]) if all_comments else '',
            'Top_Emotions': top_emotions,
            'Top_Comment_Emotions': top_comment_emotions,
            'Emotion_Diversity': len(all_emotions),
            'Categories': dict(Counter(c for c in categories if c and c != '未分类'))
        })

    final_df = pd.DataFrame(final_topics)
    print(f"生成主题: {len(final_df)} 个")

    # Step 5: 合并到目标数量
    print("\n" + "=" * 70)
    print("Step 5: 智能合并到100个主题")
    print("=" * 70)

    merged_topics = []

    # 新闻主题：>=500条保留，<<500条合并为"其他"
    news_df = final_df[final_df['Is_News'] == True].copy()
    for news_type in news_df['News_Type'].unique():
        sub_df = news_df[news_df['News_Type'] == news_type].copy()

        big = sub_df[sub_df['Total_Doc_Count'] >= 500]
        for _, row in big.iterrows():
            row_dict = row.to_dict()
            row_dict['Topic_Name'] = f"[{news_type}] {row['Source_Key'][:30]}"
            merged_topics.append(row_dict)

        small = sub_df[sub_df['Total_Doc_Count'] < 500]
        if len(small) > 0:
            total_docs = small['Total_Doc_Count'].sum()
            all_comments = ' || '.join([c for c in small['Comments_Sample'].tolist() if c])[:500]
            rep_text = small['Representative_Text'].iloc[0]

            merged_topic = {
                'Topic_Name': f"[{news_type}] 其他综合",
                'Source_Key': f"{news_type}_others",
                'Is_News': True,
                'News_Type': news_type,
                'Primary_Tier': Counter(small['Primary_Tier']).most_common(1)[0][0],
                'Primary_Category': news_type,
                'Total_Doc_Count': total_docs,
                'Original_Topic_Count': small['Original_Topic_Count'].sum(),
                'Representative_Text': str(rep_text)[:500],
                'News_Body_Sample': '',
                'Comments_Sample': all_comments,
                'Top_Emotions': '',
                'Top_Comment_Emotions': '',
                'Tier_Distribution': dict(Counter(small['Primary_Tier'])),
                'Merged_From': len(small)
            }
            merged_topics.append(merged_topic)

    # 非新闻主题：>=300条保留，<<300条合并为"综合"
    non_news_df = final_df[final_df['Is_News'] == False].copy()

    for category in non_news_df['Primary_Category'].unique():
        sub_df = non_news_df[non_news_df['Primary_Category'] == category].copy()

        for tier in ['高恐慌', '中恐慌', '低恐慌']:
            tier_df = sub_df[sub_df['Primary_Tier'] == tier].copy()
            if len(tier_df) == 0:
                continue

            tier_df = tier_df.sort_values('Total_Doc_Count', ascending=False)

            big = tier_df[tier_df['Total_Doc_Count'] >= 300]
            for _, row in big.iterrows():
                row_dict = row.to_dict()
                row_dict['Topic_Name'] = f"[{category}-{tier}] {row['Source_Key'][:30]}"
                merged_topics.append(row_dict)

            small = tier_df[tier_df['Total_Doc_Count'] < 300]
            if len(small) > 0:
                total_docs = small['Total_Doc_Count'].sum()
                all_comments = ' || '.join([c for c in small['Comments_Sample'].tolist() if c])[:500]
                rep_text = small['Representative_Text'].iloc[0]

                merged_topic = {
                    'Topic_Name': f"[{category}-{tier}] 综合",
                    'Source_Key': f"{category}_{tier}_others",
                    'Is_News': False,
                    'News_Type': None,
                    'Primary_Tier': tier,
                    'Primary_Category': category,
                    'Total_Doc_Count': total_docs,
                    'Original_Topic_Count': small['Original_Topic_Count'].sum(),
                    'Representative_Text': str(rep_text)[:500],
                    'News_Body_Sample': '',
                    'Comments_Sample': all_comments,
                    'Top_Emotions': '',
                    'Top_Comment_Emotions': '',
                    'Tier_Distribution': {tier: len(small)},
                    'Merged_From': len(small)
                }
                merged_topics.append(merged_topic)

    merged_df = pd.DataFrame(merged_topics)
    merged_df = merged_df.sort_values('Total_Doc_Count', ascending=False).reset_index(drop=True)
    merged_df['Final_Topic_ID'] = [f"FT-{i + 1:03d}" for i in range(len(merged_df))]

    print(f"最终主题数: {len(merged_df)}")

    # Step 6: 保存结果
    print("\n" + "=" * 70)
    print("Step 6: 保存结果")
    print("=" * 70)

    output_file = f"{OUTPUT_DIR}final_topics_v5_enhanced.csv"
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"保存: {output_file}")

    # 生成映射表
    mapping_rows = []
    for _, row in merged_df.iterrows():
        final_id = row['Final_Topic_ID']
        source_key = row['Source_Key']

        for sk, info in classified_groups.items():
            if sk == source_key:
                for item in info['items']:
                    mapping_rows.append({
                        'Final_Topic_ID': final_id,
                        'Original_Topic_ID': item['Topic_ID'],
                        'Original_Tier': item['Tier'],
                        'Is_News': row['Is_News'],
                        'News_Type': row['News_Type'],
                        'Primary_Category': row['Primary_Category']
                    })
                break

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_file = f"{OUTPUT_DIR}topic_mapping_v5_enhanced.csv"
    mapping_df.to_csv(mapping_file, index=False, encoding='utf-8-sig')
    print(f"保存映射表: {mapping_file}")

    # 生成报告
    report = f"""# BERTopic主题清洗报告 v5增强版

## 清洗策略
1. **基于fp_v4跨层级合并**：同一源内容在不同恐慌层级强制合并
2. **新闻转发六分类**：国内疫情/国际疫情/政策发布/数据统计/科普辟谣/社会事件
3. **内容-评论分离**：提取新闻正文 vs 转发评论中的恐慌情绪
4. **情绪信号提取**：识别害怕、愤怒、悲伤、希望、质疑、冷漠六类情绪
5. **智能分类修正**：基于内容特征重新分类

## 结果统计
- 原始主题数: {len(df)}
- 保留主题数: {len(kept_df)}
- 源内容组数: {len(source_groups)}
- 最终主题数: **{len(merged_df)}**
- 新闻主题: {merged_df['Is_News'].sum()} 个
- 非新闻主题: {len(merged_df) - merged_df['Is_News'].sum()} 个

## 新闻主题细分
"""

    news_type_dist = merged_df[merged_df['Is_News'] == True]['News_Type'].value_counts()
    for nt, count in news_type_dist.items():
        report += f"- {nt}: {count} 个\n"

    report += f"""
## 非新闻主题细分
"""

    cat_dist = merged_df[merged_df['Is_News'] == False]['Primary_Category'].value_counts()
    for cat, count in cat_dist.items():
        report += f"- {cat}: {count} 个\n"

    report += f"""
## 层级分布
"""

    tier_dist = merged_df['Primary_Tier'].value_counts()
    for tier, count in tier_dist.items():
        report += f"- {tier}: {count} 个\n"

    report += f"""
## Top-20 主题详情
"""

    top20 = merged_df.head(20)
    for _, row in top20.iterrows():
        report += f"""
### {row['Final_Topic_ID']}: {row['Topic_Name']}
- 文档数: {row['Total_Doc_Count']}
- 层级: {row['Primary_Tier']}
- 跨层级分布: {row['Tier_Distribution']}
- 情绪信号: {row['Top_Emotions']}
- 代表文本: {row['Representative_Text'][:150]}...
"""

    report_file = f"{OUTPUT_DIR}cleaning_report_v5_enhanced.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"保存报告: {report_file}")

    print("\n" + "=" * 70)
    print("清洗完成！")
    print("=" * 70)
    print(f"最终主题数: {len(merged_df)}")
    print(f"新闻主题: {merged_df['Is_News'].sum()}")
    print(f"非新闻主题: {len(merged_df) - merged_df['Is_News'].sum()}")

    return merged_df


if __name__ == '__main__':
    final_result = main()
