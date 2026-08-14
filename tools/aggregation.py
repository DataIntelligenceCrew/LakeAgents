import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

import numpy.typing as npt

def classify_column_type(
    series: pd.Series,
    column_name: str,
    table_id: Optional[str] = None,
    column_datatypes: Optional[dict[str, str]] = None,
) -> str:
    """
    Classify column type: 'numerical', 'categorical', 'text', 'ignore'.
    Uses metadata from opendata_table_index.json when available, with distinct value rules:
    - 1 distinct → ignore
    - 2 distinct → numerical (binary)
    - 3-20 distinct → categorical
    - >20 distinct → numerical if metadata is number/checkbox, text if metadata is text
    """
    sample = series.dropna().head(1)
    if len(sample) > 0 and isinstance(sample.iloc[0], (dict, list)):
        return "ignore"

    n_unique = series.nunique()
    
    # 1 distinct value → skip
    if n_unique <= 1:
        return "ignore"

    # Get metadata type from index
    meta_type = None
    if column_datatypes is not None and column_name in column_datatypes:
        meta_type = (column_datatypes.get(column_name) or "").strip().lower()
    elif table_id:
        from tools.column_descriptions import get_column_datatypes_from_index
        dtypes = get_column_datatypes_from_index(table_id)
        meta_type = (dtypes.get(column_name) or "").strip().lower()

    # Metadata types we care about: number, checkbox → number; text → text
    is_number_meta = meta_type in ("number", "checkbox")
    is_text_meta = meta_type == "text"

    # 2 distinct → binary, treat as numerical
    if n_unique == 2:
        return "numerical"
    
    n_rows = max(len(series), 1)
    ratio = n_unique / n_rows

    CATEGORICAL_RATIO_THRESHOLD = 0.03  
    if ratio < CATEGORICAL_RATIO_THRESHOLD:
        return "categorical"

    if is_number_meta:
        return "numerical"
    if is_text_meta:
        return "text"
    return "ignore"

    # No metadata or unknown: fallback to heuristic for backward compatibility
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    if series.dtype == "object":
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() / max(len(series), 1) >= 0.5:
            return "numerical"
        return "text"
    return "ignore"

def convert_numeric_columns(
    df: pd.DataFrame,
    exclude_columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Convert object columns that contain numeric values to numeric dtype.
    Uses pd.to_numeric with errors='coerce' (invalid values become NaN).

    Args:
        df: Input DataFrame
        exclude_columns: Columns to skip (e.g. join/target ID columns that cause segfault on large data).

    Returns:
        DataFrame with numeric columns converted.
    """
    df_converted = df.copy()
    exclude = set(exclude_columns or [])

    # #region agent log
    try:
        import json as _json; _ts = __import__('time').time_ns() // 1000000
        with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_D0","timestamp":_ts,"location":"aggregation.py:82","message":"dtypes before convert_numeric_columns","data":{"dtypes":{col:str(dtype) for col,dtype in df.dtypes.items()}},"hypothesisId":"D"}) + '\n')
    except OSError:
        pass
    # #endregion

    for col in df_converted.columns:
        if col in exclude:
            continue
        if df_converted[col].dtype == 'object':
            # Try to convert to numeric
            converted = pd.to_numeric(df_converted[col], errors='coerce')
            # If conversion succeeded for most values (e.g., >50% non-null), use it
            if converted.notna().sum() > 0:
                # Check if at least 50% of non-null original values converted successfully
                original_non_null = df_converted[col].notna().sum()
                if original_non_null == 0 or converted.notna().sum() / original_non_null >= 0.5:
                    df_converted[col] = converted
    
    # #region agent log
    try:
        import json as _json; _ts = __import__('time').time_ns() // 1000000
        with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_D0b","timestamp":_ts,"location":"aggregation.py:99","message":"dtypes after convert_numeric_columns","data":{"dtypes":{col:str(dtype) for col,dtype in df_converted.dtypes.items()}},"hypothesisId":"D"}) + '\n')
    except OSError:
        pass
    # #endregion
    
    return df_converted

def aggregate_categorical_column(
    df: pd.DataFrame,
    join_columns: List[str],
    categorical_column: str,
    method: str = "count",  # "count" or "proportion"
    return_as_vector: bool = True,
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
    table_id: Optional[str] = None,
    base_dir: Optional[str] = None,
    join_table_folder: Optional[str] = None,
    fasttext_model_path: Optional[str] = "fasttext.bin",
    text_k: int = 5,
    text_concat_sep: str = " | ",
    text_verbose: bool = False,
    llm_summarize: bool = True,
    llm_provider: str = "openai",
    llm_join_keys: Optional[List] = None,
    target_agg: Optional[pd.DataFrame] = None,
    target_column: Optional[str] = None,
    target_type: Optional[str] = None,
    task_type: Optional[str] = None,
    target_description: Optional[str] = None,
    user_intent: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate candidate table by join key.

    Args:
        cand_df: Candidate table DataFrame
        join_columns: Selected join column(s) from join column selection agent
        numerical_agg: Aggregation for numerical columns (default: "mean")
        table_id: Opendata table ID for metadata lookup
        base_dir: Base directory for local dataset (e.g. "datasets_agent")
        join_table_folder: Dataset folder name for local metadata (e.g. "5uac-w243")
        fasttext_model_path: Path to FastText .bin model for text embedding
                  (default: "fasttext.bin"). Set to None to skip text aggregation.
        text_k: Max texts per (join_key, text_col) for greedy subset (default: 10)
        text_concat_sep: Separator when concatenating selected texts (default: " | ")
        text_verbose: Whether to print text subset selection progress (default: False)
        llm_summarize: If True, call LLM to summarize each *_text column into {base}_summary
            (e.g. OFNS_DESC_text -> OFNS_DESC_summary). Column names are derived automatically.
        llm_provider: LLM provider for summarization ("openai" or "gemini").

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
    text_cols: List[str] = []

    column_datatypes = None
    if table_id:
        from tools.column_descriptions import get_column_datatypes_from_index
        column_datatypes = get_column_datatypes_from_index(table_id)
    elif base_dir and join_table_folder:
        from tools.column_descriptions import get_column_datatypes_from_local_metadata
        column_datatypes = get_column_datatypes_from_local_metadata(
            base_dir, join_table_folder
        )

    for col in agg_cols:
        col_type = classify_column_type(
            cand_df[col], col, table_id=table_id, column_datatypes=column_datatypes
        )
        if col_type == "numerical":
            numerical_cols.append(col)
        elif col_type == "categorical":
            categorical_cols.append(col)
        elif col_type == "text":
            text_cols.append(col)

    # Only aggregate columns that are actually numeric dtype (convert_numeric may leave some as object)
    numerical_cols = [
        c for c in numerical_cols
        if pd.api.types.is_numeric_dtype(cand_df[c])
    ]

    results_to_merge = []

    # 1. Numerical
    if numerical_cols:
        num_agg = cand_df.groupby(join_columns)[numerical_cols].agg(
            numerical_agg
        ).reset_index()
        
        # #region agent log
        import json as _json; _ts = __import__('time').time_ns() // 1000000
        with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_D2","timestamp":_ts,"location":"aggregation.py:264","message":"num_agg dtypes after groupby","data":{"table_id":table_id,"join_columns":join_columns,"dtypes":{col:str(dtype) for col,dtype in num_agg.dtypes.items()}},"hypothesisId":"D,E"}) + '\n')
        # #endregion
        
        results_to_merge.append(num_agg)

    # 2. Categorical
    for cat_col in categorical_cols:
        cat_agg = aggregate_categorical_column(
            cand_df, join_columns, cat_col, method="proportion"
        )
        
        # #region agent log
        import json as _json; _ts = __import__('time').time_ns() // 1000000
        with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_D3","timestamp":_ts,"location":"aggregation.py:275","message":"cat_agg dtypes after categorical aggregation","data":{"table_id":table_id,"cat_col":cat_col,"join_columns":join_columns,"dtypes":{col:str(dtype) for col,dtype in cat_agg.dtypes.items()}},"hypothesisId":"D,E"}) + '\n')
        # #endregion
        
        results_to_merge.append(cat_agg)

    # 3. Text (greedy subset + concat per join key, FastText used internally)
    if text_cols and fasttext_model_path:
        try:
            import time as _time
            _t0 = _time.time()
            from tools.text_integration import (
                select_text_subset_per_join_key,
                subset_map_to_dataframe,
            )
            subset_map = select_text_subset_per_join_key(
                cand_df,
                join_columns,
                text_cols,
                k=text_k,
                model_path=fasttext_model_path,
                verbose=text_verbose,
            )
            print(f"   [TIMING] Greedy text selection: {_time.time() - _t0:.1f}s")
            text_agg = subset_map_to_dataframe(
                subset_map,
                join_columns,
                text_cols,
                sep=text_concat_sep,
                suffix="_text",
            )
            results_to_merge.append(text_agg)
        except (FileNotFoundError, ImportError) as e:
            import warnings
            warnings.warn(
                f"Skipping text aggregation: could not load FastText ({e})",
                UserWarning,
            )

    # 4. Merge all results
    if not results_to_merge:
        return cand_df[join_columns].drop_duplicates().reset_index(drop=True)

    result = results_to_merge[0]
    
    # #region agent log
    import json as _json; _ts = __import__('time').time_ns() // 1000000
    with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_D1","timestamp":_ts,"location":"aggregation.py:307","message":"result dtypes after first merge","data":{"table_id":table_id,"join_columns":join_columns,"dtypes":{col:str(dtype) for col,dtype in result.dtypes.items()}},"hypothesisId":"D,E"}) + '\n')
    # #endregion
    
    for df_to_merge in results_to_merge[1:]:
        result = result.merge(df_to_merge, on=join_columns, how="outer")
    
    # #region agent log
    import json as _json; _ts = __import__('time').time_ns() // 1000000
    with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_D4","timestamp":_ts,"location":"aggregation.py:328","message":"result dtypes after all merges","data":{"table_id":table_id,"join_columns":join_columns,"dtypes":{col:str(dtype) for col,dtype in result.dtypes.items()}},"hypothesisId":"D,E"}) + '\n')
    # #endregion

    # 5. LLM summarization (bottom-5 by correlation -> select 2 -> summarize)
    if llm_summarize and any(c.endswith("_text") for c in result.columns) and target_agg is not None:
        try:
            import time as _time
            _t1 = _time.time()

            result_for_corr = result
            if llm_join_keys is not None and len(join_columns) == 1:
                result_for_corr = result[result[join_columns[0]].isin(llm_join_keys)].copy()

            merged_for_corr = result_for_corr.merge(
                target_agg[[c for c in target_agg.columns if c in join_columns + [target_column, f"{target_column}_vector"]]][:],
                on=join_columns, how="inner"
            )

            from tools.correlation import compute_feature_correlations
            corr_df = compute_feature_correlations(
                merged_for_corr, join_columns, target_column, target_type,
                fasttext_model_path=fasttext_model_path,
            )

            text_corr = corr_df[corr_df["feature_type"] == "text"].copy()
            if len(text_corr) == 0:
                pass  
            else:
              
                text_corr = text_corr.sort_values("value", ascending=True).head(5)
                bottom_5 = [
                    (row["feature"] + "_text", float(row["value"]) if pd.notna(row["value"]) else 0, row.get("description", ""))
                    for _, row in text_corr.iterrows()
                ]
               
                if len(bottom_5) < 2:
                    pass
                else:
                   
                    cand_descs = {}
                    if table_id:
                        from tools.column_descriptions import get_column_descriptions_from_index
                        cand_descs = get_column_descriptions_from_index(table_id)
                    elif base_dir and join_table_folder:
                        from tools.column_descriptions import get_column_descriptions_from_local_metadata
                        cand_descs = get_column_descriptions_from_local_metadata(base_dir, join_table_folder)
                 
                    bottom_5_with_desc = [
                        (c, v, cand_descs.get(c.replace("_text", ""), d))
                        for c, v, d in bottom_5
                    ]

                    from tools.text_integration import _llm_select_from_bottom_for_summary
                    _task_type = task_type if task_type else ("regression" if target_type == "numerical" else "classification")
                    selected_text_cols = _llm_select_from_bottom_for_summary(
                        bottom_5_with_desc,
                        target_column=target_column,
                        task_type=_task_type,
                        target_description=target_description or "",
                        user_intent=user_intent or "",
                        max_cols=2,
                        provider=llm_provider,
                    )

                    from tools.text_integration import summarize_text_per_join_key_with_llm
                    llm_df = summarize_text_per_join_key_with_llm(
                        result_for_corr, 
                        join_columns,
                        text_suffix="_text",
                        provider=llm_provider,
                        text_cols_to_summarize=selected_text_cols,
                    )
                    result = result.merge(llm_df, on=join_columns, how="left")
            print(f"   [TIMING] LLM summarization: {_time.time() - _t1:.1f}s")

        except Exception as e:
            import warnings
            warnings.warn(f"LLM summarization skipped: {e}", UserWarning)
    
    # #region agent log
    import json as _json; _ts = __import__('time').time_ns() // 1000000
    with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_D5","timestamp":_ts,"location":"aggregation.py:340","message":"final result dtypes before return","data":{"table_id":table_id,"join_columns":join_columns,"dtypes":{col:str(dtype) for col,dtype in result.dtypes.items()}},"hypothesisId":"D,E"}) + '\n')
    # #endregion

    return result


def aggregate_selected_tables(
    selected_tables: List[dict],
    base_dir: str = None,
    opendata_domain: str = None,
    join_key_filter: Optional[list] = None,
    query_join_columns: Optional[list] = None,
    llm_join_keys: Optional[list] = None,
    target_agg=None,
    target_column=None,
    target_type=None,
    task_type: Optional[str] = None,
    target_description: Optional[str] = None,
    user_intent: Optional[str] = None,
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
    from tools.sketch import get_candidate_table
    
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

        if join_key_filter is not None and len(selected_cols) == 1:
            from tools.sketch import _normalize_for_hash
            key_set = {_normalize_for_hash(v) for v in join_key_filter}  
            col = selected_cols[0]
            if col in cand_df.columns:
                cand_df = cand_df[
                    cand_df[col].apply(_normalize_for_hash).isin(key_set)  
                ].copy()
                cand_df[col] = cand_df[col].apply(_normalize_for_hash)

        # Aggregate
        try:
            # #region agent log
            import json as _json; _ts = __import__('time').time_ns() // 1000000
            with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_E1","timestamp":_ts,"location":"aggregation.py:375","message":"cand_df dtypes before aggregation","data":{"candidate_table":cand_name,"selected_cols":selected_cols,"dtypes":{col:str(dtype) for col,dtype in cand_df[selected_cols].dtypes.items() if col in cand_df.columns}},"hypothesisId":"D,E"}) + '\n')
            # #endregion
            
            agg_df = aggregate_candidate_by_join_key(
                cand_df,
                selected_cols,
                table_id=cand_name,
                llm_join_keys=llm_join_keys,
                target_agg=target_agg,
                target_column=target_column,
                target_type=target_type,
                task_type=task_type,
                target_description=target_description,
                user_intent=user_intent,
            )
            
            # #region agent log
            import json as _json; _ts = __import__('time').time_ns() // 1000000
            with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_E2","timestamp":_ts,"location":"aggregation.py:383","message":"agg_df dtypes after aggregation, before rename","data":{"candidate_table":cand_name,"selected_cols":selected_cols,"query_join_columns":query_join_columns,"dtypes":{col:str(dtype) for col,dtype in agg_df.dtypes.items()}},"hypothesisId":"D,E"}) + '\n')
            # #endregion
            
            if query_join_columns and len(selected_cols) == len(query_join_columns):
                rename_map = {selected_cols[i]: query_join_columns[i] for i in range(len(selected_cols))}
                agg_df = agg_df.rename(columns=rename_map)
                
                # #region agent log
                import json as _json; _ts = __import__('time').time_ns() // 1000000
                with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_E3","timestamp":_ts,"location":"aggregation.py:393","message":"agg_df dtypes after rename","data":{"candidate_table":cand_name,"rename_map":rename_map,"dtypes":{col:str(dtype) for col,dtype in agg_df.dtypes.items()}},"hypothesisId":"E"}) + '\n')
                # #endregion
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

def aggregate_target_by_join_key(
    query_df: pd.DataFrame,
    join_columns: List[str],
    target_column: str,
    numerical_agg: str = "mean",
    base_dir: Optional[str] = None,
    join_table_folder: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Aggregate query table's target column by join key (same logic as candidate columns).

    Args:
        query_df: Query table DataFrame (with join_columns and target_column)
        join_columns: Join column(s) used for join with candidates
        target_column: Target column to predict/explain
        numerical_agg: Aggregation for numerical target (default: "mean")
        base_dir: Base directory for local dataset (e.g. "dataset_agent3")
        join_table_folder: Dataset folder name for join/query table (e.g. "shooting count")

    Returns:
        (agg_df, target_type): agg_df has one row per join key with target value
        (scalar for numerical, _vector + _categories for categorical).
        target_type is 'numerical' or 'categorical' for downstream correlation computation.
    """
    query_df = convert_numeric_columns(query_df.copy())

    missing = [c for c in join_columns + [target_column] if c not in query_df.columns]
    if missing:
        raise ValueError(f"Join columns or target column not in DataFrame: {missing}")

    column_datatypes = None
    if base_dir and join_table_folder:
        from tools.column_descriptions import get_column_datatypes_from_local_metadata
        column_datatypes = get_column_datatypes_from_local_metadata(base_dir, join_table_folder)
    
    # #region agent log
    import json as _json; _ts = __import__('time').time_ns() // 1000000
    with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_M2","timestamp":_ts,"location":"aggregation.py:428","message":"before classify target column","data":{"target_column":target_column,"base_dir":base_dir,"join_table_folder":join_table_folder,"column_datatypes":column_datatypes,"target_in_datatypes":target_column in (column_datatypes or {})},"hypothesisId":"M"}) + '\n')
    # #endregion

    target_type = classify_column_type(
        query_df[target_column], target_column, column_datatypes=column_datatypes
    )
    
    # #region agent log
    import json as _json; _ts = __import__('time').time_ns() // 1000000
    with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_M3","timestamp":_ts,"location":"aggregation.py:437","message":"after classify target column","data":{"target_column":target_column,"target_type":target_type},"hypothesisId":"M"}) + '\n')
    # #endregion

    if target_type == "numerical":
        col = query_df[target_column]

        if col.dtype == "object" or col.dtype.name == "string":
            uniq = col.dropna().unique()
            if len(uniq) == 2:
              
                sorted_vals = sorted(str(v) for v in uniq)
                mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
                query_df = query_df.copy()
                query_df[target_column] = col.map(lambda x: mapping.get(str(x), np.nan))
                query_df[target_column] = pd.to_numeric(query_df[target_column], errors="coerce")

        agg_df = (
            query_df.groupby(join_columns)[target_column]
            .agg(numerical_agg)
            .reset_index()
        )
        return agg_df, "numerical"

    if target_type == "categorical":
        agg_df = aggregate_categorical_column(
            query_df, join_columns, target_column, method="proportion", return_as_vector=True
        )
        return agg_df, "categorical"

    raise ValueError(
        f"Target column '{target_column}' has unsupported type '{target_type}' "
        "(only 'numerical' or 'categorical' supported)"
    )