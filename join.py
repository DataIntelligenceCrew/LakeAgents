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
        all_results = json.load(f)
    
    print(f"Found {len(all_results)} datasets in analysis_results_optimized.json")
    
    # Filter to only include successful datasets
    analysis_results = []
    skipped_count = 0
    for result in all_results:
        # Skip if top-level status is failed
        if result.get('status') != 'success':
            skipped_count += 1
            continue
        
        # Skip if result.status is failed
        if 'result' not in result or result['result'].get('status') != 'success':
            skipped_count += 1
            continue
        
        analysis_results.append(result)
    
    if skipped_count > 0:
        print(f"Filtered out {skipped_count} failed datasets")
    
    print(f"Processing {len(analysis_results)} successful datasets\n")
    
    # Track results
    total_entered = len(analysis_results)
    successful_count = 0
    failed_details = []
    
    # Process each dataset
    for dataset_info in analysis_results:
        dataset_id = dataset_info['dataset']
        result = dataset_info['result']
        
        # Get join columns
        join_columns = result.get('join_columns', [])
        
        # Load subtables
        subtables = {}
        subtable_dir = f"subtables/{dataset_id}"
        
        if not os.path.exists(subtable_dir):
            print(f"{dataset_id}: FAILURE - No subtables directory")
            failed_details.append({'dataset': dataset_id, 'reason': 'No subtables directory found'})
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
                        print(f"{dataset_id}: FAILURE - {subtable_name} missing join columns: {missing_join_cols}")
                        failed_details.append({'dataset': dataset_id, 'reason': f'{subtable_name} missing join columns: {missing_join_cols}'})
                        break
                    
                    subtables[subtable_name] = df
                    
        # Check if we have enough subtables
        if len(subtables) < 2:
            print(f"{dataset_id}: FAILURE - Not enough subtables (found {len(subtables)})")
            failed_details.append({'dataset': dataset_id, 'reason': f'Not enough subtables (found {len(subtables)})'})
            continue
                    
        # Join subtables
        if len(subtables) >= 2:
            subtable_names = list(subtables.keys())
            joined_df = subtables[subtable_names[0]]
            
            # Join with remaining subtables
            initial_rows = len(subtables[subtable_names[0]])
            for subtable_name in subtable_names[1:]:
                joined_df = joined_df.merge(
                    subtables[subtable_name], 
                    on=join_columns, 
                    how='inner'
                )
            
            # Check for join explosion
            explosion_threshold = 10
            if len(joined_df) > initial_rows * explosion_threshold:
                print(f"{dataset_id}: FAILURE - Join explosion ({initial_rows} → {len(joined_df)} rows, {len(joined_df)/initial_rows:.1f}x)")
                failed_details.append({'dataset': dataset_id, 'reason': f'Join explosion ({initial_rows} → {len(joined_df)} rows)'})
                continue
            
            # Check absolute size limit
            max_allowed_rows = 10000000
            if len(joined_df) > max_allowed_rows:
                print(f"{dataset_id}: FAILURE - Result too large ({len(joined_df)} rows)")
                failed_details.append({'dataset': dataset_id, 'reason': f'Result too large ({len(joined_df)} rows)'})
                continue
            
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
                        continue                        
            
            # Save joined result
            output_dir = f"joined_tables"
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/{dataset_id}_joined.csv"
            joined_df.to_csv(output_file, index=False)
            print(f"{dataset_id}: SUCCESS")
            successful_count += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"JOIN SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets entered: {total_entered}")
    print(f"Successfully joined: {successful_count}")
    print(f"Failed: {len(failed_details)}")
    
    if failed_details:
        print(f"\n{'='*80}")
        print(f"FAILED JOINS DETAILS")
        print(f"{'='*80}")
        for item in failed_details:
            print(f"  {item['dataset']}: {item['reason']}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()