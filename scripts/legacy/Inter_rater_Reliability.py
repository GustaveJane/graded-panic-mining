# 人工标注数据一致性检验
import pandas as pd
import os
import sys
import io

# 设置控制台编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 第一轮人工复核的5000条数据
df_combined_1 = pd.read_csv(r"D:\\dataFiles\\Disaster_management\\data_preprocessing\\panic_test_4o_1.csv")
# 第二轮人工复核的1065个数据
df_panic_2 = pd.read_excel(r"D:\\dataFiles\\Disaster_management\\data_preprocessing\\panic_test_11.xlsx")
df_unpanic_2 = pd.read_excel(r"D:\\dataFiles\\Disaster_management\\data_preprocessing\\panic_test_00.xlsx")
# 合并两个DataFrame  
df_combined_2 = pd.concat([df_panic_2, df_unpanic_2], ignore_index=True)  

# 数据预处理：确保列名一致
df_combined_1 = df_combined_1.rename(columns={'panic_related': 'label_first'})
df_combined_2 = df_combined_2.rename(columns={'panic_related': 'label_second'})

# 合并两个数据集（inner join，仅保留共有文本）
merged_df = pd.merge(
    df_combined_1[['text', 'label_first']],
    df_combined_2[['text', 'label_second']],
    on='text',
    how='inner'
)

# 计算一致性指标
# 1. 简单一致率 (Simple Agreement Rate)
total_samples = len(merged_df)
agreement_count = (merged_df['label_first'] == merged_df['label_second']).sum()
simple_agreement = agreement_count / total_samples

# 2. Cohen's Kappa（考虑随机一致性）
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(merged_df['label_first'], merged_df['label_second'])

# 3. 混淆矩阵
confusion_matrix = pd.crosstab(
    merged_df['label_first'],
    merged_df['label_second'],
    rownames=['第一轮标注'],
    colnames=['第二轮标注'],
    margins=True
)

# 4. 分类一致性指标
# 恐慌样本一致性
panic_agreement = merged_df[merged_df['label_second'] == 1]['label_first'].mean()
# 非恐慌样本一致性
non_panic_agreement = 1 - merged_df[merged_df['label_second'] == 0]['label_first'].mean()


# 5. 不一致样本分析
disagreement_df = merged_df[merged_df['label_first'] != merged_df['label_second']]
disagreement_rate = len(disagreement_df) / total_samples

# 输出结果
print("\n====== 人工标注一致性检验报告 ======")
print(f"共有样本数量: {total_samples}")
print(f"简单一致率: {simple_agreement:.4f} ({agreement_count}/{total_samples})")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"不一致率: {disagreement_rate:.4f} ({len(disagreement_df)}/{total_samples})")
print(f"恐慌样本一致性: {panic_agreement:.4f}")
print(f"非恐慌样本一致性: {non_panic_agreement:.4f}")

print("\n====== 混淆矩阵 ======")
print(confusion_matrix)

# 保存不一致样本供复查
# disagreement_df.to_csv("D:\\dataFiles\\Disaster_management\\data_preprocessing\\inconsistent_samples.csv", index=False)

# 指标说明
print("\n====== 指标说明 ======")
print("1. 简单一致率: 两次标注结果完全相同的样本比例")
print("2. Cohen's Kappa: 考虑随机一致性的校正指标，取值范围[-1,1]")
print("   >0.8: 几乎完美一致 | >0.6: 显著一致 | >0.4: 中等一致 | <0.4: 一致性较弱")
print("3. 恐慌样本一致性: 当第二轮标注为恐慌时，第一轮也标注为恐慌的比例")
print("4. 非恐慌样本一致性: 当第二轮标注为非恐慌时，第一轮也标注为非恐慌的比例")
print("5. 不一致样本: 已保存至 inconsistent_samples.csv 文件")