import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置 =====================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
})

MAIN_COLOR = '#3A506B'
DATA_DIR = '/root/autodl-tmp/'
FILE_LIST = [
    'temp19_1.csv', 'temp1.csv', 'temp2.csv', 'temp3.csv',
    'temp4.csv', 'temp5.csv', 'temp6.csv', 'temp7.csv',
    'temp8.csv', 'temp9.csv', 'temp10.csv', 'temp11.csv', 'temp12.csv'
]
USER_DATA_PATH = "/root/autodl-tmp/raw_data_utf8/user.csv"
OUTPUT_PROVINCE = "/root/autodl-tmp/province_panic_peak.png"

PROVINCE_MAP = {
    '北京':'Beijing','上海':'Shanghai','广东':'Guangdong','江苏':'Jiangsu',
    '浙江':'Zhejiang','山东':'Shandong','四川':'Sichuan','湖北':'Hubei',
    '湖南':'Hunan','河南':'Henan','河北':'Hebei','安徽':'Anhui',
    '福建':'Fujian','江西':'Jiangxi','广西':'Guangxi','陕西':'Shaanxi',
    '山西':'Shanxi','贵州':'Guizhou','重庆':'Chongqing','天津':'Tianjin',
    '云南':'Yunnan','辽宁':'Liaoning','黑龙江':'Heilongjiang','吉林':'Jilin',
    '甘肃':'Gansu','内蒙古':'Inner Mongolia','宁夏':'Ningxia','新疆':'Xinjiang',
    '西藏':'Xizang','青海':'Qinghai','海南':'Hainan'
}

# ===================== 极速加载（已修复列不存在问题） =====================
def load_fast():
    dfs = []
    for fname in FILE_LIST:
        p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(p):
            continue

        # 先读表头，判断有哪些列
        header = pd.read_csv(p, nrows=0).columns.tolist()
        use_cols = ['user_id', 'created_at']
        if 'panic' in header:
            use_cols.append('panic')
        if 'panic_pred' in header:
            use_cols.append('panic_pred')

        # 只读取存在的列，不报错
        df = pd.read_csv(p, usecols=use_cols, dtype={'user_id': str})

        # 统一恐慌列名为 panic
        if 'panic_pred' in df.columns and 'panic' not in df.columns:
            df.rename(columns={'panic_pred': 'panic'}, inplace=True)

        # 只保留 2020年1-3月（疫情最高峰）
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df = df[(df['created_at'] >= '2020-01-01') & (df['created_at'] <= '2020-03-31')]

        dfs.append(df[['user_id', 'panic']])
        print(f"Loaded {fname} | 有效行数 {len(df):,}")

    return pd.concat(dfs, ignore_index=True)

# ===================== 主程序 =====================
post_df = load_fast()
user_df = pd.read_csv(USER_DATA_PATH, usecols=['user_id', 'province'], dtype={'user_id': str})
user_df['province'] = user_df['province'].map(PROVINCE_MAP).dropna()

# 合并
df = post_df.merge(user_df, on='user_id', how='inner')

# 按省份计算真实恐慌率
prov = df.groupby('province')['panic'].agg(['count', 'sum'])
prov['panic_rate'] = (prov['sum'] / prov['count'] * 100).round(2)
prov = prov.sort_values('panic_rate', ascending=False).head(15)

# 画图
plt.figure(figsize=(14,7))
prov['panic_rate'].plot(kind='bar', color=MAIN_COLOR, width=0.7)
plt.title('Regional Panic Rate During COVID-19 Peak (Jan–Mar 2020)', fontsize=14, weight='bold')
plt.xlabel('Province')
plt.ylabel('Panic Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_PROVINCE, dpi=150)

print("\n✅ 运行完成！图片已保存到：", OUTPUT_PROVINCE)