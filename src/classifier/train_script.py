import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ====================== 已经帮你填好你的路径 ======================
input_file = r"/root/autodl-tmp/covid_temp_check1.csv"
train_save_path = r"/root/autodl-tmp/temp_train1.csv"
val_save_path = r"/root/autodl-tmp/temp_val1.csv"
# =================================================================

# 读取文件（已修复编码问题）
df = pd.read_csv(input_file, encoding='utf-8-sig')

# 用人工复核的 check 列作为最终标签
df = df[['text', 'check']].rename(columns={'check': 'temp_related'})

# 清洗空值
df = df.dropna(subset=['text', 'temp_related'])
print("数据总量：", len(df))

# 8:2拆分训练集/验证集
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['temp_related']
)

# 保存
train_df.to_csv(train_save_path, index=False, encoding='utf-8-sig')
val_df.to_csv(val_save_path, index=False, encoding='utf-8-sig')

print("✅ 数据拆分完成！")
print(f"训练集：{len(train_df)} 条")
print(f"验证集：{len(val_df)} 条")