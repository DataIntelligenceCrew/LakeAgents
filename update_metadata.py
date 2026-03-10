#!/usr/bin/env python3
"""
Update metadata.json to only include columns that exist in rows.csv
"""

import csv
import json
from pathlib import Path


def update_metadata():
    """Update metadata to match columns in rows.csv"""
    table_path = Path("datasets_omnimatch2/join table")
    
    # Read actual columns from rows.csv
    print("Reading columns from rows.csv...")
    with open(table_path / "rows.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        actual_columns = next(reader)
    
    print(f"Actual columns in rows.csv: {actual_columns}")
    
    # Load metadata
    print("Loading metadata.json...")
    with open(table_path / "metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if "resource" not in metadata:
        print("No 'resource' found in metadata")
        return
    
    resource = metadata["resource"]
    column_names = resource.get("columns_name", [])
    
    # Find indices of actual columns in metadata
    selected_indices = []
    for col in actual_columns:
        if col in column_names:
            idx = column_names.index(col)
            selected_indices.append(idx)
            print(f"Found column '{col}' at index {idx}")
        else:
            print(f"Warning: Column '{col}' not found in metadata")
    
    # Filter all column-related arrays
    resource["columns_name"] = [column_names[i] for i in selected_indices]
    resource["columns_field_name"] = [resource.get("columns_field_name", [])[i] 
                                      if i < len(resource.get("columns_field_name", [])) else "" 
                                      for i in selected_indices]
    resource["columns_datatype"] = [resource.get("columns_datatype", [])[i] 
                                    if i < len(resource.get("columns_datatype", [])) else "Unknown" 
                                    for i in selected_indices]
    resource["columns_description"] = [resource.get("columns_description", [])[i] 
                                       if i < len(resource.get("columns_description", [])) else "" 
                                       for i in selected_indices]
    resource["columns_format"] = [resource.get("columns_format", [])[i] 
                                  if i < len(resource.get("columns_format", [])) else {} 
                                  for i in selected_indices]
    
    # Save updated metadata
    output_file = table_path / "metadata.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nUpdated metadata saved to {output_file}")
    print(f"Kept {len(selected_indices)} columns out of {len(column_names)} original columns")


if __name__ == "__main__":
    update_metadata()

