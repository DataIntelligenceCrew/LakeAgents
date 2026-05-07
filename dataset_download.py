#!/usr/bin/env python3
"""
"""

import os
import shutil
from pathlib import Path

source_dir = Path("/localdisk3/username/opendata/datasets_agent")
desktop_dir = Path("/localdisk3/username/opendata/datasets_agent_1")

desktop_dir.mkdir(parents=True, exist_ok=True)

found_count = 0
copied_count = 0

print(f"Scanning {source_dir}...")
print(f"Target dir: {desktop_dir}\n")

for dataset_folder in source_dir.iterdir():
    if not dataset_folder.is_dir():
        continue
    
    dataset_id = dataset_folder.name
    rows_csv = dataset_folder / "rows.csv"
    
    if not rows_csv.exists():
        continue
    
    try:
        with open(rows_csv, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            
        if 'BORO' in header.upper():
            found_count += 1
            print(f"Found dataset with BORO column: {dataset_id}")
            
            dest_file = desktop_dir / f"{dataset_id}.csv"
            shutil.copy2(rows_csv, dest_file)
            copied_count += 1
            print(f"  ✓ Copied to: {dest_file}")
            
    except Exception as e:
        print(f"  ✗ Error processing {dataset_id}: {e}")

print(f"\nDone.")
print(f"Found {found_count} datasets with BORO column")
print(f"Copied {copied_count} files to {desktop_dir}")
