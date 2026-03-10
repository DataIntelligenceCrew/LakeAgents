#!/usr/bin/env python3
"""
脚本功能：从 datasets_agent 目录中筛选包含 BORO 列的表，
并将它们复制到桌面的 dataset 文件夹，以数据集 ID 命名
"""

import os
import shutil
from pathlib import Path

# 源目录和目标目录
source_dir = Path("/localdisk3/ytang49/opendata/datasets_agent")
desktop_dir = Path("/localdisk3/ytang49/opendata/datasets_agent_1")

# 创建目标目录（如果不存在）
desktop_dir.mkdir(parents=True, exist_ok=True)

# 统计信息
found_count = 0
copied_count = 0

print(f"开始扫描 {source_dir} 目录...")
print(f"目标目录: {desktop_dir}\n")

# 遍历所有数据集文件夹
for dataset_folder in source_dir.iterdir():
    if not dataset_folder.is_dir():
        continue
    
    dataset_id = dataset_folder.name
    rows_csv = dataset_folder / "rows.csv"
    
    # 检查 rows.csv 是否存在
    if not rows_csv.exists():
        continue
    
    # 读取第一行（表头）检查是否包含 BORO
    try:
        with open(rows_csv, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            
        # 检查表头中是否包含 BORO（不区分大小写）
        if 'BORO' in header.upper():
            found_count += 1
            print(f"找到包含 BORO 列的数据集: {dataset_id}")
            
            # 复制文件到目标目录，以数据集 ID 命名
            dest_file = desktop_dir / f"{dataset_id}.csv"
            shutil.copy2(rows_csv, dest_file)
            copied_count += 1
            print(f"  ✓ 已复制到: {dest_file}")
            
    except Exception as e:
        print(f"  ✗ 处理 {dataset_id} 时出错: {e}")

print(f"\n完成！")
print(f"找到 {found_count} 个包含 BORO 列的数据集")
print(f"成功复制 {copied_count} 个文件到 {desktop_dir}")