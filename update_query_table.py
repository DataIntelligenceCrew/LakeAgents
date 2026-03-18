
import pandas as pd
import json

import pandas as pd
import os

input_path = "original_query_table/Food Inspections-Chicago/rows.csv"
output_path = "query_table/Food Inspections-Chicago/rows.csv"

df = pd.read_csv(input_path)

print("Original data:", df.shape)
print("Original columns:", len(df.columns)) 

keep_cols = [
    "Risk",
    "Inspection ID",
    "Zip"
]

print("Missing columns:", [c for c in keep_cols if c not in df.columns])

df_subset = df[keep_cols]

print("New data shape:", df_subset.shape)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_subset.to_csv(output_path, index=False)

print("Saved to:", output_path)


with open('original_query_table/Food Inspections-Chicago/metadata.json', 'r') as f:
    metadata = json.load(f)

original_cols = metadata['resource']['columns_name']
keep_indices = [original_cols.index(col) for col in keep_cols]

print(f"\nOriginal metadata columns: {len(original_cols)}")
print(f"Kept column indices: {keep_indices}")

metadata['resource']['columns_name'] = [original_cols[i] for i in keep_indices]
metadata['resource']['columns_field_name'] = [metadata['resource']['columns_field_name'][i] for i in keep_indices]
metadata['resource']['columns_datatype'] = [metadata['resource']['columns_datatype'][i] for i in keep_indices]
metadata['resource']['columns_description'] = [metadata['resource']['columns_description'][i] for i in keep_indices]
metadata['resource']['columns_format'] = [metadata['resource']['columns_format'][i] for i in keep_indices]

print(f"\nFiltered metadata columns: {len(metadata['resource']['columns_name'])}")
print(f"\nColumn descriptions:")
for col, desc in zip(metadata['resource']['columns_name'], metadata['resource']['columns_description']):
    print(f"  - {col}: {desc}")

output_metadata_path = 'query_table/Food Inspections-Chicago/metadata.json'
os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)  
with open(output_metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n✓ Updated metadata.json")
  