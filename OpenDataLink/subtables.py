#!/usr/bin/env python3
"""
Simple script to split original tables into subtables based on analysis results.
Reads analysis_results_optimized.json and creates separate CSV files for each subtable.
"""

import json
import pandas as pd
import os

def main():
    """Main function to process all datasets and create subtables."""
    
    # Load analysis results
    with open("analysis_results_optimized.json", "r") as f:
        analysis_results = json.load(f)
    
    print(f"Found {len(analysis_results)} datasets to process")
    
    # Process each dataset
    for dataset_info in analysis_results:
        dataset_id = dataset_info['dataset']
       
        print(f"Processing dataset: {dataset_id}")

        if 'result' not in dataset_info:
            print(f"  Skipping {dataset_id}: {dataset_info.get('error', 'No result available')}")
            continue

        result = dataset_info['result']   
             
        # Skip if analysis failed
        if 'status' not in result or result.get('status') != 'success':
            print(f"Skipping {dataset_id}: {result.get('status')}")
            continue
        
        print(f"Processing dataset: {dataset_id}")
        
        # Try to find the CSV file
        csv_path = f"datasets/{dataset_id}/rows.csv"

    
        if not os.path.exists(csv_path):  
            print(f"  No CSV file found for {dataset_id}")
            continue
        
        # Check file size and determine reading strategy
        file_size = os.path.getsize(csv_path) / (1024*1024)  # Size in MB
        print(f"  File size: {file_size:.1f}MB")
        
        # Determine number of rows to read based on file size
        if file_size < 50:
            nrows = None  # Read all rows
            print(f"  Reading all rows")
        elif file_size < 100:
            nrows = 10000  # Read first 10,000 rows
            print(f"  Large file detected, reading first {nrows:,} rows only")
        else:
            nrows = 5000   # Read first 5,000 rows
            print(f"  Very large file detected, reading first {nrows:,} rows only")
        
        # Load the original data
        df = pd.read_csv(csv_path, nrows=nrows)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Create output directory (at same level as datasets folder)
        output_dir = f"subtables/{dataset_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create each subtable
        for i in range(1, 4):
            subtable_key = f'subtable_{i}'
            if subtable_key in result:
                subtable_config = result[subtable_key]
                subtable_name = subtable_config.get('name', f'subtable_{i}')
                columns = subtable_config.get('columns', [])
                
                # Check if columns exist in the dataframe
                existing_columns = [col for col in columns if col in df.columns]
                if not existing_columns:
                    print(f"  Warning: No valid columns found for {subtable_name}")
                    continue
                
                # Create subtable
                subtable_df = df[existing_columns].copy()
                
                # Remove duplicates based on join columns
                join_columns = result.get('join_columns', [])
                existing_join_cols = [col for col in join_columns if col in subtable_df.columns]
                if existing_join_cols:
                    subtable_df = subtable_df.drop_duplicates(subset=existing_join_cols)
                
                # Save subtable
                output_file = f"{output_dir}/{subtable_name}.csv"
                subtable_df.to_csv(output_file, index=False)
                print(f"  Created {subtable_name}: {len(subtable_df)} rows, {len(existing_columns)} columns")
    
    print("Processing completed!")


if __name__ == "__main__":
    main()
