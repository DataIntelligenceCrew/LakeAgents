#!/usr/bin/env python3
"""
Simple script to join subtables back together using join columns.
Reads analysis_results_optimized.json and joins subtables for each dataset.
"""

import json
import pandas as pd
import os

def main():
    """Main function to join all subtables."""
    
    # Load analysis results
    with open("analysis_results_optimized.json", "r") as f:
        analysis_results = json.load(f)
    
    print(f"Found {len(analysis_results)} datasets to process")
    
    # Process each dataset
    for dataset_info in analysis_results:
        dataset_id = dataset_info['dataset']
        if 'result' not in dataset_info:
            print(f"Skipping {dataset_id}: {dataset_info.get('error', 'No result available')}")
            continue
        result = dataset_info['result']
        
        # Skip if analysis failed
        if 'status' not in result or result.get('status') != 'success':
            print(f"Skipping {dataset_id}: {result.get('status')}")
            continue
        
        print(f"Processing dataset: {dataset_id}")
        
        # Get join columns
        join_columns = result.get('join_columns', [])
        print(f"  Join columns: {join_columns}")
        
        # Load subtables
        subtables = {}
        subtable_dir = f"subtables/{dataset_id}"
        
        if not os.path.exists(subtable_dir):
            print(f"  No subtables directory found for {dataset_id}")
            continue
        
        # Load each subtable
        for i in range(1, 4):
            subtable_key = f'subtable_{i}'
            if subtable_key in result:
                subtable_config = result[subtable_key]
                subtable_name = subtable_config.get('name', f'subtable_{i}')
                csv_file = f"{subtable_dir}/{subtable_name}.csv"
                
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    
                    # Check if join columns exist
                    missing_join_cols = [col for col in join_columns if col not in df.columns]
                    if missing_join_cols:
                        print(f"  Warning: {subtable_name} missing join columns: {missing_join_cols}")
                        print(f"  Skipping {subtable_name}")
                        continue
                    
                    subtables[subtable_name] = df
                    print(f"  Loaded {subtable_name}: {len(df)} rows, {len(df.columns)} columns")
                else:
                    print(f"  Warning: {csv_file} not found")
                    
        # Join subtables
        if len(subtables) >= 2:
            # Start with the first subtable
            subtable_names = list(subtables.keys())
            joined_df = subtables[subtable_names[0]]
            
            # Join with remaining subtables
            for subtable_name in subtable_names[1:]:
                joined_df = joined_df.merge(
                    subtables[subtable_name], 
                    on=join_columns, 
                    how='inner'
                )
                print(f"  Joined with {subtable_name}: {len(joined_df)} rows")
            
            original_df = pd.read_csv(f"datasets/{dataset_id}/rows.csv", nrows=len(joined_df))
            for col in joined_df.columns:
                if col in original_df.columns:
                    original_dtype = original_df[col].dtype
                    try:
                        if original_dtype in ['int64', 'int32', 'int16']:
                            joined_df[col] = joined_df[col].fillna(0).astype(original_dtype)
                        else:
                            joined_df[col] = joined_df[col].astype(original_dtype)
                    except (ValueError, TypeError) as e:
                        print(f"  Warning: Could not convert column '{col}' to {original_dtype}: {e}")
                        # Keep original data type, do not convert
                        continue                        
            
            # Save joined result
            output_dir = f"joined_tables"
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/{dataset_id}_joined.csv"
            joined_df.to_csv(output_file, index=False)
            print(f"  Saved joined table: {output_file}")
        else:
            print(f"  Not enough subtables to join (found {len(subtables)})")
    
    print("Processing completed!")


if __name__ == "__main__":
    main()