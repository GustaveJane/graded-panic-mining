import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# ===================== 全局配置（纯英文，无中文） =====================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

# 配色
MAIN_COLOR = '#3A506B'  # 深蓝灰
ACCENT_COLOR = '#C97B63'  # 暖赭石
FANS_PALETTE = ['#3A506B', '#4D6A8C', '#6084AD', '#739ECD']
AGE_PALETTE = ['#3A506B', '#4D6A8C', '#6084AD', '#739ECD', '#86B8DE', '#99D2FF']

# ===================== 路径配置 =====================
DATA_DIR = '/root/autodl-tmp/'
FILE_LIST = [
    'temp19_1.csv', 'temp1.csv', 'temp2.csv', 'temp3.csv',
    'temp4.csv', 'temp5.csv', 'temp6.csv', 'temp7.csv',
    'temp8.csv', 'temp9.csv', 'temp10.csv', 'temp11.csv', 'temp12.csv'
]
USER_DATA_PATH = "/root/autodl-tmp/raw_data_utf8/user.csv"

# 输出文件
OUTPUT_FANS_BOX = "/root/autodl-tmp/fans_panic_box.png"
OUTPUT_FANS_BAR = "/root/autodl-tmp/fans_panic_bar.png"
OUTPUT_GENDER = "/root/autodl-tmp/gender_panic.png"
OUTPUT_AGE = "/root/autodl-tmp/age_panic.png"
OUTPUT_PROVINCE = "/root/autodl-tmp/province_panic.png"


# ===================== 1. 加载帖子数据 =====================
def load_post_data():
    all_posts = []
    for fname in FILE_LIST:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        df_sample = pd.read_csv(fpath, nrows=0)
        panic_col = 'panic' if 'panic' in df_sample.columns else 'panic_pred'
        df = pd.read_csv(fpath, usecols=['user_id', panic_col])
        df.rename(columns={panic_col: 'panic'}, inplace=True)
        all_posts.append(df)
        print(f"✅ Loaded {fname} | {len(df):,} rows")
    return pd.concat(all_posts, ignore_index=True)


# ===================== 2. 计算用户恐慌率 =====================
def calc_user_panic_rate(post_df):
    user_panic = post_df.groupby("user_id").agg(
        total_posts=("panic", "count"),
        panic_posts=("panic", "sum")
    ).reset_index()
    user_panic["panic_rate"] = (user_panic["panic_posts"] / user_panic["total_posts"] * 100).round(2)
    return user_panic[user_panic["total_posts"] >= 2]


# ===================== 3. 加载并清洗用户数据 =====================
def load_user_data():
    user_df = pd.read_csv(USER_DATA_PATH, usecols=['user_id', 'gender', 'province', 'fans_num', 'birthday'])
    # 清洗生日
    user_df['birthday_clean'] = pd.to_datetime(user_df['birthday'], errors='coerce')
    user_df = user_df[(user_df['birthday_clean'].dt.year >= 1900) & (user_df['birthday_clean'].dt.year <= 2020)]
    user_df['age'] = 2020 - user_df['birthday_clean'].dt.year
    # 粉丝分组
    user_df['fans_group'] = pd.cut(
        user_df['fans_num'],
        bins=[-1, 1000, 10000, 100000, float('inf')],
        labels=['0-1k', '1k-10k', '10k-100k', '100k+']
    )
    # 年龄分组
    user_df['age_group'] = pd.cut(
        user_df['age'],
        bins=[-1, 18, 25, 35, 45, 60, 120],
        labels=['Under 18', '18-25', '26-35', '36-45', '46-60', 'Over 60']
    )

    # 关键：把中文性别 替换成 英文
    gender_map = {"男": "Male", "女": "Female"}
    user_df["gender"] = user_df["gender"].map(gender_map)

    # 清洗gender
    user_df = user_df.dropna(subset=['gender'])
    return user_df


# ===================== 4. 合并数据 =====================
def merge_data(user_df, user_panic_df):
    return pd.merge(user_df, user_panic_df, on="user_id", how="inner")


# ===================== 5. 绘图函数（纯英文） =====================
def draw_plots(final_df):
    # --- 5.1 粉丝数-恐慌率 箱线图 ---
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        x="fans_group", y="panic_rate", data=final_df,
        palette=FANS_PALETTE, width=0.6, fliersize=2, linewidth=1
    )
    plt.title("Follower Scale vs Panic Rate (Boxplot)", pad=12)
    plt.xlabel("Follower Group")
    plt.ylabel("Panic Rate (%)")
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig(OUTPUT_FANS_BOX, dpi=300)
    print(f"✅ Saved: {OUTPUT_FANS_BOX}")

    # --- 5.2 粉丝数-恐慌率 柱状图 ---
    plt.figure(figsize=(8, 5))
    fans_mean = final_df.groupby("fans_group")["panic_rate"].mean().reset_index()
    sns.barplot(
        x="fans_group", y="panic_rate", data=fans_mean,
        color=MAIN_COLOR, width=0.6
    )
    plt.title("Average Panic Rate by Follower Group", pad=12)
    plt.xlabel("Follower Group")
    plt.ylabel("Average Panic Rate (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_FANS_BAR, dpi=300)
    print(f"✅ Saved: {OUTPUT_FANS_BAR}")

    # --- 5.3 性别图 ---
    plt.figure(figsize=(6, 5))
    sns.boxplot(
        x="gender", y="panic_rate", data=final_df,
        palette=[MAIN_COLOR, ACCENT_COLOR], width=0.6, fliersize=2,
        order=['Female', 'Male']
    )
    plt.title("Gender vs Panic Rate", pad=12)
    plt.xlabel("Gender")
    plt.ylabel("Panic Rate (%)")
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig(OUTPUT_GENDER, dpi=300)
    print(f"✅ Saved: {OUTPUT_GENDER}")

    # --- 5.4 年龄图 ---
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        x="age_group", y="panic_rate", data=final_df,
        palette=AGE_PALETTE, width=0.6, fliersize=2
    )
    plt.title("Age Group vs Panic Rate", pad=12)
    plt.xlabel("Age Group")
    plt.ylabel("Panic Rate (%)")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_AGE, dpi=300)
    print(f"✅ Saved: {OUTPUT_AGE}")

    # --- 5.5 省份图 ---
    plt.figure(figsize=(12, 6))
    province_mean = final_df.dropna(subset=["province"]).groupby("province")["panic_rate"].mean().sort_values(
        ascending=False).head(15)
    province_mean.plot(kind="bar", color=MAIN_COLOR, width=0.7)
    plt.title("Regional Differences in Average Panic Rate (Top 15 Provinces)", pad=12)
    plt.xlabel("Province")
    plt.ylabel("Average Panic Rate (%)")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_PROVINCE, dpi=300)
    print(f"✅ Saved: {OUTPUT_PROVINCE}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("===== User Panic Analysis =====")
    post_df = load_post_data()
    user_panic_df = calc_user_panic_rate(post_df)
    user_df = load_user_data()
    final_df = merge_data(user_df, user_panic_df)
    draw_plots(final_df)
    print("\n🎉 All plots saved successfully!")