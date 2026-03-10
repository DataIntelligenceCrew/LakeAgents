
import pandas as pd
import json

import pandas as pd
import os

input_path = "original_query_table/Public_safety-NYC/rows.csv"
output_path = "query_table/Public_safety-NYC/rows.csv"

# 读取原始数据
df = pd.read_csv(input_path)

print("原始数据:", df.shape)
print("原始列数:", len(df.columns))

# 正确列名写法
keep_cols = [
    "Police Precinct/Geographic Location",
    "Overall DV Rate",
    "Shootings"
]

# 检查列是否存在
print("缺失列:", [c for c in keep_cols if c not in df.columns])

# 选列
df_subset = df[keep_cols]

print("新数据形状:", df_subset.shape)

# 保存到新文件（不覆盖原文件）
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # ← 添加这一行
df_subset.to_csv(output_path, index=False)

print("已保存到:", output_path)


# 2. 更新 metadata.json
with open('original_query_table/Public_safety-NYC/metadata.json', 'r') as f:
    metadata = json.load(f)

original_cols = metadata['resource']['columns_name']
keep_indices = [original_cols.index(col) for col in keep_cols]

print(f"\n原始 metadata 列数: {len(original_cols)}")
print(f"保留的列索引: {keep_indices}")

metadata['resource']['columns_name'] = [original_cols[i] for i in keep_indices]
metadata['resource']['columns_field_name'] = [metadata['resource']['columns_field_name'][i] for i in keep_indices]
metadata['resource']['columns_datatype'] = [metadata['resource']['columns_datatype'][i] for i in keep_indices]
metadata['resource']['columns_description'] = [metadata['resource']['columns_description'][i] for i in keep_indices]
metadata['resource']['columns_format'] = [metadata['resource']['columns_format'][i] for i in keep_indices]

print(f"\n过滤后 metadata 列数: {len(metadata['resource']['columns_name'])}")
print(f"\n列描述:")
for col, desc in zip(metadata['resource']['columns_name'], metadata['resource']['columns_description']):
    print(f"  - {col}: {desc}")

output_metadata_path = 'query_table/Public_safety-NYC/metadata.json'
os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)  
with open(output_metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n✓ 已更新 metadata.json")
  