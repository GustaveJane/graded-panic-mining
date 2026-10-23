# Preprocessing rule inventory

This file inventories the preprocessing rules that are actually implemented in
this repository. It is generated from the source, so the rules below are the
rules the code runs — not a description written after the fact.

## 1. Stage implemented in `src/data/clean.py`

- Removal of repost boilerplate: `转发理由...`, `转发微博`, `Repost` (case-insensitive).
- Topic-key construction: punctuation and whitespace stripped before comparison.
- Separation of a reposted news body from the user's own comment
  (markers: `转发理由`, `转发微博`, `Repost`, `轉發微博`, `全文转发理由`).
- Six-way news-type classification: domestic epidemic / international epidemic /
  policy release / data statistics / science communication / social events.
- Emotion-pattern detection for the comment layer (patterns reproduced below).

```python
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
```

## 2. Stage implemented in `src/data/filter_covid.py`

- Reads the monthly `cleaned_YYYY-MM.csv` files, strips newlines from column names,
  drops empty text, and draws a fixed-size sample per month (`random_state=42`).
- Runs the three-round LLM voting annotation; the deployed prompt is released
  verbatim under `../prompts/`.

## 3. Upstream cleaning that is NOT in this repository

The corpus-level deduplication, advertisement/bot removal and text normalisation
produced the `cleaned_YYYY-MM.csv` inputs consumed above. That script ran on the
compute server and is not part of this repository, so its exact rule list cannot
be released here. Anything in the manuscript that refers to "complete rule
lists" for those three steps cannot yet be honoured by this repository.
