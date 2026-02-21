import pandas as pd
from typing import List, Optional, Tuple

def classify_column_type(series: pd.Series) -> str:
    """
    classify column type: 'numerical', 'categorical', 'text', 'ignore'
    
    Rules:
    - Numerical: numeric dtype or (object with distinct > 20 after conversion)
    - Bool: distinct == 2 (treated as numerical)
    - Categorical: 2 < distinct <= 20
    - Text: object with distinct > 20
    - Ignore: other types
    """
    # 1. Numeric dtype → numerical
    if pd.api.types.is_numeric_dtype(series):
        return 'numerical'
    
    # 2. Object: try convert to numeric first
    if series.dtype == 'object':
        # Try numeric conversion
        converted = pd.to_numeric(series, errors='coerce')
        if converted.notna().sum() / len(series) >= 0.5:  # at least 50% can be converted
            n_unique = converted.nunique()
            if n_unique == 2:
                return 'numerical'  # Bool as numerical
            elif 2 < n_unique <= 20:
                return 'categorical'
            else:
                return 'numerical'
        
        # Cannot convert to numeric → check as categorical/text
        n_unique = series.nunique()
        if n_unique == 2:
            return 'categorical'  # Bool-like text
        elif 2 < n_unique <= 20:
            return 'categorical'
        else:
            return 'text'
    
    # 3. Other types (datetime, etc.) → ignore
    return 'ignore'

def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns that contain numeric values to numeric dtype.
    Uses pd.to_numeric with errors='coerce' (invalid values become NaN).
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with numeric columns converted.
    """
    df_converted = df.copy()
    for col in df_converted.columns:
        if df_converted[col].dtype == 'object':
            # Try to convert to numeric
            converted = pd.to_numeric(df_converted[col], errors='coerce')
            # If conversion succeeded for most values (e.g., >50% non-null), use it
            if converted.notna().sum() > 0:
                # Check if at least 50% of non-null original values converted successfully
                original_non_null = df_converted[col].notna().sum()
                if original_non_null == 0 or converted.notna().sum() / original_non_null >= 0.5:
                    df_converted[col] = converted
    return df_converted

def aggregate_categorical_column(
    df: pd.DataFrame,
    join_columns: List[str],
    categorical_column: str,
    method: str = "count",  # "count" or "proportion"
    return_as_vector: bool = True,  # True: 单列 vector, False: 多列 one-hot
) -> pd.DataFrame:
    """
    Aggregate a categorical column by join key.
    
    Args:
        return_as_vector: If True, return single column with vector values.
                         If False, return one-hot columns (current implementation).
    
    Returns:
        If return_as_vector=True:
            DataFrame with join_columns + one vector column (e.g., grade_level_vector)
        If return_as_vector=False:
            DataFrame with join_columns + one-hot columns (e.g., grade_level_K-5, ...)
    """
    # Step 1: Count by (join_key, category)
    grouped = df.groupby(join_columns + [categorical_column]).size().reset_index(name='count')
    
    # Step 2: Get all categories (global order)
    all_categories = sorted(df[categorical_column].dropna().unique())
    
    if return_as_vector:
        # Step 3a: Convert to vector format
        vectors = {}
        for join_key_vals in df[join_columns].drop_duplicates().values:
            # Make join_key_vals a tuple for dict key (handle numpy array)
            if len(join_columns) == 1:
                key = join_key_vals[0]
            else:
                key = tuple(join_key_vals.tolist() if hasattr(join_key_vals, 'tolist') else join_key_vals)
            
            # Get distribution for this join key
            mask = True
            for i, jc in enumerate(join_columns):
                mask = mask & (grouped[jc] == join_key_vals[i])
            subset = grouped[mask]
            
            # Build count vector
            count_dict = dict(zip(subset[categorical_column], subset['count']))
            vector = [count_dict.get(cat, 0) for cat in all_categories]
            
            # Convert to proportion if needed
            if method == "proportion":
                total = sum(vector)
                vector = [v / total if total > 0 else 0 for v in vector]
            
            vectors[key] = vector
        
        # Convert to DataFrame
        if len(join_columns) == 1:
            result_df = pd.DataFrame({join_columns[0]: list(vectors.keys())})
        else:
            result_df = pd.DataFrame(list(vectors.keys()), columns=join_columns)
        result_df[f'{categorical_column}_vector'] = list(vectors.values())
        result_df[f'{categorical_column}_categories'] = [all_categories] * len(result_df)
        return result_df
    
    else:
        # Step 3b: Pivot to one-hot format (current implementation)
        pivot = grouped.pivot_table(
            index=join_columns,
            columns=categorical_column,
            values='count',
            fill_value=0
        ).reset_index()
        
        # Step 3: Rename columns (add prefix)
        new_cols = {c: f"{categorical_column}_{c}" for c in pivot.columns if c not in join_columns}
        pivot = pivot.rename(columns=new_cols)
        
        # Step 4: Convert to proportion if needed
        if method == "proportion":
            feature_cols = [c for c in pivot.columns if c not in join_columns]
            row_sums = pivot[feature_cols].sum(axis=1)
            for col in feature_cols:
                pivot[col] = pivot[col] / row_sums.replace(0, 1)  
        
        return pivot

def aggregate_candidate_by_join_key(
    cand_df: pd.DataFrame,
    join_columns: List[str],
    numerical_agg: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate candidate table by join key.
    
    Args:
        cand_df: Candidate table DataFrame
        join_columns: Selected join column(s) from join column selection agent
        numerical_agg: Aggregation for numerical columns (default: "mean")
    
    Returns:
        Aggregated DataFrame with one row per join key.
    """
    cand_df = convert_numeric_columns(cand_df)

    # Validate join columns
    missing = [c for c in join_columns if c not in cand_df.columns]
    if missing:
        raise ValueError(f"Join columns not in DataFrame: {missing}")
    
    # Columns to aggregate (exclude join columns)
    agg_cols = [c for c in cand_df.columns if c not in join_columns]
    
    # Classify columns
    numerical_cols = []
    categorical_cols = []
    for col in agg_cols:
        col_type = classify_column_type(cand_df[col])
        if col_type == 'numerical':
            numerical_cols.append(col)
        elif col_type == 'categorical':
            categorical_cols.append(col)

    results_to_merge = []

    # 1. Numerical
    if numerical_cols:
        num_agg = cand_df.groupby(join_columns)[numerical_cols].agg(numerical_agg).reset_index()
        results_to_merge.append(num_agg)

    # 2. Categorical
    for cat_col in categorical_cols:
        cat_agg = aggregate_categorical_column(cand_df, join_columns, cat_col, method="proportion")
        results_to_merge.append(cat_agg)

    # 3. Merge all results
    if not results_to_merge:
        return cand_df[join_columns].drop_duplicates().reset_index(drop=True)

    result = results_to_merge[0]
    for df_to_merge in results_to_merge[1:]:
        result = result.merge(df_to_merge, on=join_columns, how='outer')

    return result


def aggregate_selected_tables(
    selected_tables: List[dict],
    base_dir: str = None,
    opendata_domain: str = None,
) -> List[dict]:
    """
    Aggregate all tables from Phase 2 join column selection.
    
    Args:
        selected_tables: Output from Phase 2 (final_selected_tables)
        base_dir: Base directory for local tables
        opendata_domain: Domain for fetching from API
    
    Returns:
        List of dicts with candidate_table, selected_columns, and aggregated_df.
    """
    from utils.sketch import get_candidate_table
    
    results = []
    for tbl in selected_tables:
        cand_name = tbl.get("candidate_table")
        selected_cols = tbl.get("selected_columns", [])
        
        if not cand_name or not selected_cols:
            continue
        
        # Fetch table (might be cached from Phase 2)
        cand_df, status = get_candidate_table(cand_name, opendata_domain)
        if not status["success"]:
            print(f"   ⚠️  Failed to fetch {cand_name}: {status['reason']}")
            continue
        
        # Aggregate
        try:
            agg_df = aggregate_candidate_by_join_key(cand_df, selected_cols)
            results.append({
                "candidate_table": cand_name,
                "selected_columns": selected_cols,
                "aggregated_df": agg_df,
                "original_rows": len(cand_df),
                "aggregated_rows": len(agg_df),
            })
            print(f"   ✅ {cand_name}: {len(cand_df)} → {len(agg_df)} rows")
        except Exception as e:
            print(f"   ❌ Aggregation failed for {cand_name}: {e}")
    
    return results