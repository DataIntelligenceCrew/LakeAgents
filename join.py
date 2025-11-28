#!/usr/bin/env python3
"""
Simple script to join subtables back together using join columns.
Reads analysis_results_optimized.json and joins subtables for each dataset.
"""

import json
import pandas as pd
import os
import sys

def main(datasets_dir="datasets"):
    """Main function to join all subtables.

    Args:
        datasets_dir: Directory containing the datasets (default: "datasets")
    """
    
    # Load analysis results
    with open("analysis_results_optimized.json", "r") as f:
        all_results = json.load(f)
    
    print(f"Found {len(all_results)} datasets in analysis_results_optimized.json")
    print(f"Using datasets directory: {datasets_dir}")
    
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
    successful_dataset_ids = []  # Track successful joins
    
    # Process each dataset
    for idx, dataset_info in enumerate(analysis_results, 1):
        dataset_id = dataset_info['dataset']
        result = dataset_info['result']
        
        print(f"\n[{idx}/{len(analysis_results)}] Processing {dataset_id}...")
        
        # Get join columns
        join_columns = result.get('join_columns', [])
        
        # Load subtables
        subtables = {}
        subtable_dir = f"subtables/{dataset_id}"
        
        if not os.path.exists(subtable_dir):
            print(f"{dataset_id}: FAILURE - No subtables directory")
            failed_details.append({'dataset': dataset_id, 'reason': 'No subtables directory found'})
            continue
        
        # Prefer new candidate/non-candidate tables
        if 'candidate_table' in result and 'non_candidate_table' in result:
            ct_conf = result['candidate_table']
            nct_conf = result['non_candidate_table']
            ct_name = ct_conf.get('name', 'Candidate_Features')
            nct_name = nct_conf.get('name', 'NonCandidate_With_Target')
            for name in [ct_name, nct_name]:
                csv_file = f"{subtable_dir}/{name}.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    missing_join_cols = [col for col in join_columns if col not in df.columns]
                    if missing_join_cols:
                        print(f"{dataset_id}: FAILURE - {name} missing join columns: {missing_join_cols}")
                        failed_details.append({'dataset': dataset_id, 'reason': f'{name} missing join columns: {missing_join_cols}'})
                        subtables = {}
                        break
                    subtables[name] = df
        else:
            # Legacy: load subtable_1..3
            for i in range(1, 4):
                subtable_key = f'subtable_{i}'
                if subtable_key in result:
                    subtable_config = result[subtable_key]
                    subtable_name = subtable_config.get('name', f'subtable_{i}')
                    csv_file = f"{subtable_dir}/{subtable_name}.csv"
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                        missing_join_cols = [col for col in join_columns if col not in df.columns]
                        if missing_join_cols:
                            print(f"{dataset_id}: FAILURE - {subtable_name} missing join columns: {missing_join_cols}")
                            failed_details.append({'dataset': dataset_id, 'reason': f'{subtable_name} missing join columns: {missing_join_cols}'})
                            break
                        subtables[subtable_name] = df
                    
        # Check if we have enough tables
        if len(subtables) < 2:
            print(f"{dataset_id}: FAILURE - Not enough tables (found {len(subtables)})")
            failed_details.append({'dataset': dataset_id, 'reason': f'Not enough tables (found {len(subtables)})'})
            continue
                    
        # Join subtables
        if len(subtables) >= 2:
            subtable_names = list(subtables.keys())
            print(f"  {dataset_id}: Starting join with {len(subtables)} tables on {join_columns}...")
            joined_df = subtables[subtable_names[0]]
            print(f"  {dataset_id}: Base table has {len(joined_df)} rows")
            
            # Join with remaining subtables
            initial_rows = len(subtables[subtable_names[0]])
            for subtable_name in subtable_names[1:]:
                subtable_df = subtables[subtable_name].copy()
                
                # Remove columns that already exist in joined_df (except join columns)
                cols_to_drop = [col for col in subtable_df.columns 
                                if col in joined_df.columns and col not in join_columns]
                if cols_to_drop:
                    print(f"  {dataset_id}: Removing duplicate columns from {subtable_name}: {cols_to_drop}")
                    subtable_df = subtable_df.drop(columns=cols_to_drop)
                
                print(f"  {dataset_id}: Joining with {subtable_name} ({len(subtable_df)} rows)...")
                joined_df = joined_df.merge(
                    subtable_df, 
                    on=join_columns, 
                    how='inner'
                )
                print(f"  {dataset_id}: After join: {len(joined_df)} rows")
            
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
            
            # Read original data with progress indication
            print(f"  {dataset_id}: Reading original data for dtype alignment...")
            try:
                original_df = pd.read_csv(f"{datasets_dir}/{dataset_id}/rows.csv", nrows=len(joined_df))
                print(f"  {dataset_id}: Aligning data types...")
            except Exception as e:
                print(f"{dataset_id}: FAILURE - Could not read original data: {e}")
                failed_details.append({'dataset': dataset_id, 'reason': f'Could not read original data: {e}'})
                continue

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
            print(f"  {dataset_id}: Saving joined table ({len(joined_df)} rows, {len(joined_df.columns)} columns)...")
            joined_df.to_csv(output_file, index=False)
            print(f"{dataset_id}: SUCCESS ✓")
            successful_count += 1
            successful_dataset_ids.append(dataset_id)  # Track successful join
    
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
    
    # Save successful join dataset IDs for compare step
    if successful_dataset_ids:
        output_file = "successful_joins.json"
        with open(output_file, 'w') as f:
            json.dump(successful_dataset_ids, f, indent=2)
        print(f"\nSaved {len(successful_dataset_ids)} successful join dataset IDs to {output_file}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    datasets_dir = "datasets"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--datasets-dir" and len(sys.argv) > 2:
            datasets_dir = sys.argv[2]
    main(datasets_dir)