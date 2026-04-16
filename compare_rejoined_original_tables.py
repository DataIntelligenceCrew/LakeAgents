#!/usr/bin/env python3
"""
Simple script to compare rejoined tables with original tables.
Compares joined_tables/ with datasets/ to verify they are identical.
"""

import pandas as pd
import os
import glob
import numpy as np
import json
import sys

def dataframes_equal_with_tolerance(df1, df2, tolerance=1e-5):
    """Compare two DataFrames whether they are equal, with tolerance for floating point numbers"""
    if len(df1) != len(df2):
        return False
    
    if len(df1.columns) != len(df2.columns):
        return False
    
    # Check if the column names are the same
    if not df1.columns.equals(df2.columns):
        return False
    
    # Compare row by row
    for i in range(len(df1)):
        for col in df1.columns:
            val1 = df1.iloc[i][col]
            val2 = df2.iloc[i][col]
            
            # Handle NaN values
            if pd.isna(val1) and pd.isna(val2):
                continue
            elif pd.isna(val1) or pd.isna(val2):
                return False
            
            # Check if the value is a numerical type
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Compare the values with tolerance
                if abs(val1 - val2) > tolerance:
                    return False
            else:
                # Compare the values directly
                if val1 != val2:
                    return False
    
    return True

def values_equal_with_tolerance(val1, val2, tolerance=1e-5):
    """Compare two values whether they are equal, with tolerance for floating point numbers"""
    # Handle NaN values
    if pd.isna(val1) and pd.isna(val2):
        return True
    elif pd.isna(val1) or pd.isna(val2):
        return False
    
    # Check if the value is a numerical type
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return abs(val1 - val2) <= tolerance
    else:
        return val1 == val2

def rows_equal_with_tolerance(row1, row2, tolerance=1e-5):
    """Compare two rows whether they are equal, with tolerance for floating point numbers"""
    for col in row1.index:
        if not values_equal_with_tolerance(row1[col], row2[col], tolerance):
            return False
    return True

def main(datasets_dir="datasets"):
    """Main function to compare rejoined and original tables.
    
    Args:
        datasets_dir: Directory containing the datasets (default: "datasets")
    """
    
    print("=== COMPARING REJOINED vs ORIGINAL TABLES ===")
    print(f"Using datasets directory: {datasets_dir}")
    
    # Load successful join dataset IDs from join step
    try:
        with open("successful_joins.json", "r") as f:
            successful_join_datasets = set(json.load(f))
        print(f"Loaded {len(successful_join_datasets)} successful join dataset IDs from join step")
    except FileNotFoundError:
        print("Warning: successful_joins.json not found, will compare all joined files")
        successful_join_datasets = None
    
    # Find joined tables directory
    joined_dir = "joined_tables"
    if not os.path.exists(joined_dir):
        print(f"Error: {joined_dir} directory not found!")
        return []
    
    # Only process datasets that successfully passed the join step
    if successful_join_datasets:
        # Only verify datasets that passed join step
        joined_files = []
        for dataset_id in successful_join_datasets:
            joined_file = f"{joined_dir}/{dataset_id}_joined.csv"
            if os.path.exists(joined_file):
                joined_files.append(joined_file)
        
        print(f"Found {len(joined_files)} joined tables (filtered to match successful_joins.json)")
    else:
        # Fallback: scan all joined files if successful_joins.json not found
        joined_files = glob.glob(f"{joined_dir}/*_joined.csv")
        print(f"Found {len(joined_files)} joined tables to compare (no successful_joins filter)")
    
    total_tables = len(joined_files)
    successful_joins = 0
    failed_joins = 0
    successful_table_names = []
    
    # Compare each joined table with original
    for joined_file in joined_files:
        # Extract dataset_id from filename (e.g., "22u3-xenr_joined.csv" -> "22u3-xenr")
        dataset_id = os.path.basename(joined_file).replace("_joined.csv", "")
        print(f"\n--- Comparing {dataset_id} ---")
        
        # Load joined table
        try:
            joined_df = pd.read_csv(joined_file)
            print(f"  Joined table: {len(joined_df)} rows, {len(joined_df.columns)} columns")
        except Exception as e:
            print(f"  Error loading joined table: {e}")
            continue
        
        # Load original table
        original_file = f"{datasets_dir}/{dataset_id}/rows.csv"
        try:
            # Determine nrows based on file size (same logic as subtables.py)
            file_size = os.path.getsize(original_file) / (1024*1024)  # Size in MB
            if file_size < 50:
                nrows = None  # Read all rows
            elif file_size < 100:
                nrows = 10000  # Read first 10,000 rows
            else:
                nrows = 5000   # Read first 5,000 rows
            
            original_df = pd.read_csv(original_file, nrows=nrows)
            print(f"  Original table: {len(original_df)} rows, {len(original_df.columns)} columns (file size: {file_size:.1f}MB, nrows={nrows})")
        except Exception as e:
            print(f"  Error loading original table: {e}")
            failed_joins += 1
            continue
        
        # Basic comparison
        # print(f"  Row count match: {len(joined_df) == len(original_df)}")
        # print(f"  Column count match: {len(joined_df.columns) == len(original_df.columns)}")
        
        # Check if column names match
        # joined_cols = set(joined_df.columns)
        # original_cols = set(original_df.columns)
        # if joined_cols == original_cols:
        #     print(f"  Column names match: ✓")

                # Check if column names match
        joined_cols = set(joined_df.columns)
        original_cols = set(original_df.columns)
        
        if joined_cols != original_cols:
            print(f"\n--- JOIN FAILED: {dataset_id} ---")
            print(f"  Column names differ: ✗")
            missing_in_joined = original_cols - joined_cols
            missing_in_original = joined_cols - original_cols
            if missing_in_joined:
                print(f"    Missing in joined: {missing_in_joined}")
            if missing_in_original:
                print(f"    Missing in original: {missing_in_original}")
            failed_joins += 1
            continue
        print(f"  Column names match: ✓")

            
        # If column names match, compare entire content
        if len(joined_df) > 0 and len(original_df) > 0:
            print(f"  Comparing entire tables...")
                
            # Find primary key column (a column with unique values)
            primary_key = None
            for col in joined_df.columns:
                if joined_df[col].nunique() == len(joined_df):
                    primary_key = col
                    print(f"  Using '{primary_key}' as primary key for sorting")
                    break
            
            # Sort both dataframes using primary key or all columns
            if primary_key:
                joined_sorted = joined_df.sort_values(primary_key).reset_index(drop=True)
                original_sorted = original_df.sort_values(primary_key).reset_index(drop=True)
            else:
                print(f"  No unique primary key found, sorting by all columns")
                sort_columns = sorted(set(joined_df.columns) & set(original_df.columns))
                joined_sorted = joined_df.sort_values(sort_columns, ascending=True).reset_index(drop=True)
                original_sorted = original_df.sort_values(sort_columns, ascending=True).reset_index(drop=True)

            # Reorder columns alphabetically for consistent comparison
            joined_sorted = joined_sorted.reindex(sorted(joined_sorted.columns), axis=1)
            original_sorted = original_sorted.reindex(sorted(original_sorted.columns), axis=1)
                   
            # Compare entire tables with tolerance for floating point numbers
            if not dataframes_equal_with_tolerance(joined_sorted, original_sorted):
                print(f"\n--- JOIN FAILED: {dataset_id} ---")
                print(f"  Tables are DIFFERENT")
                
                # Find specific differences
                print(f"  Analyzing differences...")
                
                # Check row by row for first few differences
                diff_count = 0
                max_diffs_to_show = 5
                
                for i in range(min(len(joined_sorted), len(original_sorted))):
                    if not rows_equal_with_tolerance(joined_sorted.iloc[i], original_sorted.iloc[i]):
                        diff_count += 1
                        if diff_count <= max_diffs_to_show:
                            print(f"    Row {i} differs:")
                            joined_row = joined_sorted.iloc[i]
                            original_row = original_sorted.iloc[i]
                            
                            for col in joined_sorted.columns:
                                # Use tolerance comparison for floating point numbers
                                if not values_equal_with_tolerance(joined_row[col], original_row[col]):
                                    print(f"      {col}: joined='{joined_row[col]}' vs original='{original_row[col]}'")
                
                if diff_count > max_diffs_to_show:
                    print(f"    ... and {diff_count - max_diffs_to_show} more differences")
                
                print(f"  Total differences found: {diff_count}")
                failed_joins += 1
            else:
                print(f"  Entire tables match: ✓")
                print(f"  Tables are IDENTICAL")
                successful_joins += 1
                successful_table_names.append(dataset_id)
            # print(f"  Cannot compare - one table is empty")
        else:
            print(f"\n--- JOIN FAILED: {dataset_id} ---")
            print(f"  Cannot compare - one table is empty")
            failed_joins += 1
    # Print summary
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"Total tables processed: {total_tables}")
    print(f"Successful joins: {successful_joins}")
    print(f"Failed joins: {failed_joins}")
    print(f"Success rate: {(successful_joins/total_tables)*100:.1f}%" if total_tables > 0 else "N/A")
    
    print(f"\n=== COMPARISON COMPLETED ===")
    return successful_table_names

def get_successful_rejoined_tables(datasets_dir="datasets"):
    """Wrapper function to get successful rejoined table names.
    
    Args:
        datasets_dir: Directory containing the datasets (default: "datasets")
    
    Returns:
        list: List of dataset_ids that successfully passed the comparison
    """
    return main(datasets_dir)
    
if __name__ == "__main__":
    datasets_dir = "datasets"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--datasets-dir" and len(sys.argv) > 2:
            datasets_dir = sys.argv[2]
    main(datasets_dir)