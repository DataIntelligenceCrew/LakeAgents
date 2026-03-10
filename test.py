import pandas as pd
import numpy as np
import os

input_path = "/localdisk3/ytang49/opendata/query_table/Public_safety-NYC/rows.csv"
output_dir = "/localdisk3/ytang49/opendata/query_table/Public_safety-NYC_noisy"
output_path = os.path.join(output_dir, "rows.csv")

df = pd.read_csv(input_path)

print("原始数据:", df.shape)

# 随机选 50 条
sample_indices = df.sample(n=50, random_state=42).index
noise_rows = df.loc[sample_indices].copy()

# 对数值列加噪声
numeric_cols = noise_rows.select_dtypes(include=np.number).columns

for col in numeric_cols:
    std = noise_rows[col].std()
    if std == 0 or np.isnan(std):
        continue
    noise = np.random.normal(0, 0.1 * std, size=len(noise_rows))
    noise_rows[col] = noise_rows[col] + noise

# 删除原始 50 条
df_remaining = df.drop(index=sample_indices)

# 拼接 noisy 版本
df_final = pd.concat([df_remaining, noise_rows], ignore_index=True)

print("最终数据:", df_final.shape)

os.makedirs(output_dir, exist_ok=True)
df_final.to_csv(output_path, index=False)

print("已保存到:", output_path)