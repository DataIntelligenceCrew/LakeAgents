import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union
from scipy.spatial.distance import cosine
from fasttext import FastText
import re
from typing import List, Dict, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def find_dataset_dir(dataset_name: str, base_dir: str = "datasets_omnimatch2") -> str:
    """
    Find the real directory name based on the cleaned dataset name (ignoring trailing spaces, case, etc.).
    
    Args:
        dataset_name: The dataset name returned by LLM (possibly without trailing spaces)
        base_dir: The base directory path
        
    Returns:
        The real directory name (with spaces, etc.)
        
    Raises:
        FileNotFoundError: If the base directory does not exist
    """
    base_path = Path(base_dir).resolve()
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_path}")
    
    # Clean the input name: convert to lowercase, remove trailing spaces, remove trailing numbers

    clean_input = dataset_name.lower().strip()
    clean_input = re.sub(r'\s+\d+$', '', clean_input).strip()
    
    # Create a mapping: cleaned name -> real directory name
    name_map = {}
    for d in base_path.iterdir():
        if not d.is_dir():
            continue
        clean_dir_name = d.name.lower().strip()
        clean_dir_name = re.sub(r'\s+\d+$', '', clean_dir_name).strip()
        name_map[clean_dir_name] = d.name
    
    # Find a match
    if clean_input in name_map:
        return name_map[clean_input]
    
    # If no match, try to match directly (possibly the user has given an exact name)
    direct_path = base_path / dataset_name
    if direct_path.exists() and direct_path.is_dir():
        return dataset_name
    
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found in {base_path}. "
        f"Available datasets: {sorted(name_map.values())}"
    )


def read_metadata(dataset_name: str = None, base_dir: str = "datasets_omnimatch2") -> Dict[str, Any]:
    if "datasets_omnimatch2" not in str(base_dir):
        base_dir = "datasets_omnimatch2"

    base_path = Path(base_dir).resolve()
    
    if not base_path.exists():
        raise FileNotFoundError(f"Path does not exist: {base_path}")
    
    valid_dirs = {d.name.lower().strip(): d.name for d in base_path.iterdir() if d.is_dir()}

    if dataset_name:
        import re
        clean_input = dataset_name.lower().strip()
        clean_input = re.sub(r'\s+\d+$', '', clean_input).strip()
        
    
        if clean_input in valid_dirs:
            dataset_name = valid_dirs[clean_input]
        elif dataset_name.lower().strip() in valid_dirs:
            dataset_name = valid_dirs[dataset_name.lower().strip()]
    
    out: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    
    for ds in base_path.iterdir():
        if not ds.is_dir(): continue
    
        if dataset_name and ds.name != dataset_name: continue
        
        mf = ds / "metadata.json"
        if not mf.exists(): continue
        try:
            with mf.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            resource = meta.get("resource", {})
            out[ds.name] = {
                "table_description": resource.get("description", "")
            }
        except Exception as e:
            errors[ds.name] = str(e)
            
    return {"metadata_by_dataset": out, "errors": errors}
model = FastText.load_model("fasttext.bin")

def get_fasttext_sim(model, text1, text2):
    
    v1 = model.get_sentence_vector(str(text1))
    v2 = model.get_sentence_vector(str(text2))
    return 1 - cosine(v1, v2)

def compute_statistics(
    dataset_name: str,
    join_table_name: str,
    join_column: List[str],
    base_dir: str = "datasets_omnimatch2",
    data_filename: str = "rows.csv",
    max_rows: int = 1000
) -> List[Dict[str, Any]]:
    """
    Compute statistics for each column in candidate_df compared with a known join column.

    Args:
        candidate_df: Candidate table DataFrame
        join_column: Name of the known join column
        join_column_table: DataFrame containing the join column (e.g., non-candidate table)
        dataset_name: Dataset folder name to read metadata from
        base_dir: Base directory containing datasets
        max_rows: Maximum number of rows to use (default 100)

    Returns:
        List of dicts, each containing statistics for a candidate column compared with join_column:
        - candidate_column: Column name from candidate_df
        - join_column: The known join column name
        - jaccard_similarity: Jaccard similarity between value sets
        - containment1: How much candidate column is contained in join column
        - containment2: How much join column is contained in candidate column
        - embedding_similarity: Embedding similarity (placeholder)
        - uniqueness_ratio_candidate: Uniqueness ratio of candidate column
        - uniqueness_ratio_join_column: Uniqueness ratio of join column
        - missing_rate_candidate: Missing rate of candidate column
        - missing_rate_join_column: Missing rate of join column
        - candidate_column_name: Column name from metadata
        - candidate_column_description: Column description from metadata
        - join_column_name: Join column name from metadata
        - join_column_description: Join column description from metadata
    """
    base_dir = "datasets_omnimatch2"

    real_candidate_name = find_dataset_dir(dataset_name, base_dir)
    real_join_table_name = find_dataset_dir(join_table_name, base_dir)
    
    candidate_path = Path(base_dir) / real_candidate_name / data_filename
    join_path = Path(base_dir) / real_join_table_name / data_filename

    candidate_df = pd.read_csv(candidate_path, low_memory=False)
    join_column_table = pd.read_csv(join_path, low_memory=False)


    # Sort by all columns and take first max_rows
    candidate_df_sample = candidate_df.sort_values(by=list(candidate_df.columns)).head(max_rows).copy()
    join_column_table_sample = join_column_table.sort_values(by=list(join_column_table.columns)).head(max_rows).copy()
    
    # Verify join column(s) exist
    if isinstance(join_column, list):
        missing = [c for c in join_column if c not in join_column_table_sample.columns]
        if missing:
            raise ValueError(f"Join column(s) {missing} not found in join table")
    else:
        if join_column not in join_column_table_sample.columns:
            raise ValueError(f"Join column '{join_column}' not found in join table")

    # Read metadata to get column descriptions
    dataset_path = Path(base_dir) / real_candidate_name
    metadata_file = dataset_path / "metadata.json"
    
    column_metadata = {}
    
    if metadata_file.exists():
        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        if "resource" in metadata:
            resource = metadata["resource"]
            column_names = resource.get("columns_name", [])
            descriptions = resource.get("columns_description", [])
            
            for i, col_name in enumerate(column_names):
                column_metadata[col_name] = {
                    "name": col_name,
                    "description": descriptions[i] if i < len(descriptions) else ""
                }
    
    # Get join column metadata
    if isinstance(join_column, list):
        join_col_info = [
            column_metadata.get(c, {"name": c, "description": ""})
            for c in join_column
        ]
    else:
        join_col_info = column_metadata.get(join_column, {"name": join_column, "description": ""})

    # Extract value set for join column
    if isinstance(join_column, list):
        join_names = []
        join_descs = []
        for c in join_column:
            info = column_metadata.get(c, {"name": c, "description": ""})
            join_names.append(info.get("name", c))
            d = info.get("description", "")
            if d:
                join_descs.append(f"{c}: {d}")
        join_col_info = {
            "name": "||".join(join_names),
            "description": " | ".join(join_descs)
        }
    else:
        join_col_info = column_metadata.get(join_column, {"name": join_column, "description": ""})
    # Extract value set for join column
    if isinstance(join_column, list):
        join_col_values = (
            join_column_table_sample[join_column]
            .dropna()
            .astype(str)
            .agg("||".join, axis=1)
            .str.lower()
            .str.strip()
        )
    else:
        join_col_values = (
            join_column_table_sample[join_column]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
        )

    join_col_set = set(join_col_values)

    main_keys = join_column if isinstance(join_column, list) else [join_column]
    candidate_all_cols = list(candidate_df.columns)
    
    mapping_results = {}
    total_sim = 0
    for m_key in main_keys:
        best_match = None
        max_sim = -1.0
        for c_col in candidate_all_cols:
            sim = get_fasttext_sim(model, m_key, c_col)
            if sim > max_sim:
                max_sim = sim
                best_match = c_col
        mapping_results[m_key] = {"match": best_match, "score": max_sim}
        total_sim += max_sim
    
    avg_emb_score = total_sim / len(main_keys)
    embedding_similarity = float(round(avg_emb_score, 4))
    
    # Uniqueness ratio and missing rate for join column
    # total_rows_join = len(join_column_table_sample)
    # unique_count_join = join_column_table_sample[join_column].nunique()
    # uniqueness_ratio_join = unique_count_join / total_rows_join if total_rows_join > 0 else 0.0
    # missing_count_join = join_column_table_sample[join_column].isna().sum()
    # missing_rate_join = missing_count_join / total_rows_join if total_rows_join > 0 else 0.0
    
    # Compute statistics for each column in candidate_df
    final_results = [] 
    total_rows_candidate = len(candidate_df_sample)
    
    for candidate_col in candidate_df_sample.columns:
        # Get candidate column metadata
        candidate_col_info = column_metadata.get(candidate_col, {"name": candidate_col, "description": ""})
        
        # Extract value set for candidate column
        candidate_col_values = candidate_df_sample[candidate_col].dropna().astype(str).str.lower().str.strip()
        candidate_col_set = set(candidate_col_values)
        
        # Jaccard similarity
        intersection = candidate_col_set & join_col_set
        union = candidate_col_set | join_col_set
        jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0
        
        # Set containment
        containment1 = len(intersection) / len(join_col_set) if len(join_col_set) > 0 else 0.0
        containment2 = len(intersection) / len(candidate_col_set) if len(candidate_col_set) > 0 else 0.0
        
        # Uniqueness and missing
        unique_count = candidate_df_sample[candidate_col].nunique()
        uniqueness_ratio = unique_count / total_rows_candidate if total_rows_candidate > 0 else 0.0
        missing_rate = candidate_df_sample[candidate_col].isna().sum() / total_rows_candidate if total_rows_candidate > 0 else 0.0
        
        
        col_stats = {
            "candidate_column": candidate_col,
            "join_column": join_column,
            "jaccard_similarity": jaccard_similarity,
            "containment1": containment1,
            "containment2": containment2,
            "uniqueness_ratio_candidate": uniqueness_ratio,
            "missing_rate_candidate": missing_rate,
            "candidate_column_name": candidate_col_info["name"],
            "candidate_column_description": candidate_col_info["description"],
            "join_column_name": join_col_info["name"],
            "join_column_description": join_col_info["description"],
            "embedding_similarity": embedding_similarity
        }
        
        final_results.append(col_stats)
    
    
    sorted_results = sorted(
        final_results,
        key=lambda r: max(r["containment1"], r["containment2"], r["jaccard_similarity"]),
        reverse=True
    )[:5]
    for r in sorted_results:
        r.pop("jaccard_similarity", None)
        r.pop("containment1", None)
        r.pop("containment2", None)
    
    return sorted_results

def compute_integration_quality(
    base_table_name: str,
    candidate_table_name: str,
    base_join_columns: List[str],
    candidate_join_columns: List[str] = None,
    base_dir: str = "datasets_omnimatch2",
    data_filename: str = "rows.csv"
) -> float:
    """
    Compute Integration Quality (IQ): proportion of instances in the base table 
    that can be successfully augmented by the candidate table.
    
    Args:
        base_table_name: Name of the base/join table
        candidate_table_name: Name of the candidate table to join
        base_join_columns: Join columns in the base table
        candidate_join_columns: Join columns in the candidate table (if None, same as base_join_columns)
        base_dir: Base directory containing datasets
        data_filename: CSV filename (default: "rows.csv")
    
    Returns:
        IQ value (float between 0.0 and 1.0): proportion of base table rows successfully joined
    """
    # Find real directory names
    real_base_name = find_dataset_dir(base_table_name, base_dir)
    real_candidate_name = find_dataset_dir(candidate_table_name, base_dir)
    
    # Load tables
    base_path = Path(base_dir) / real_base_name / data_filename
    candidate_path = Path(base_dir) / real_candidate_name / data_filename
    
    base_df = pd.read_csv(base_path, low_memory=False)
    candidate_df = pd.read_csv(candidate_path, low_memory=False)
    
    # Use same join columns if candidate_join_columns not specified
    if candidate_join_columns is None:
        candidate_join_columns = base_join_columns
    
    # Verify columns exist
    missing_base = [col for col in base_join_columns if col not in base_df.columns]
    if missing_base:
        raise ValueError(f"Join columns {missing_base} not found in base table")
    
    missing_candidate = [col for col in candidate_join_columns if col not in candidate_df.columns]
    if missing_candidate:
        raise ValueError(f"Join columns {missing_candidate} not found in candidate table")
    
    # Perform join
    merged = pd.merge(
        base_df,
        candidate_df,
        left_on=base_join_columns,
        right_on=candidate_join_columns,
        how='inner'
    )
    
    # Calculate IQ: proportion of base table rows successfully augmented
    total_base_rows = len(base_df)
    if total_base_rows == 0:
        return 0.0
    
    successfully_augmented_rows = len(merged)
    iq = successfully_augmented_rows / total_base_rows
    
    return float(iq)

def compute_feature_importance(
    base_table_name: str,
    candidate_table_name: str,
    base_join_columns: List[str],
    candidate_column: str,
    target_column: str,
    task_type: str,
    candidate_join_columns: Optional[List[str]] = None,
    base_dir: str = "datasets_omnimatch2",
    data_filename: str = "rows.csv",
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Compute Feature Importance (FI): the improvement in prediction metrics 
    when adding a candidate column to the base table.
    
    FI = metrics_with_candidate_column - metrics_without_candidate_column
    
    Args:
        base_table_name: Name of the base/join table
        candidate_table_name: Name of the candidate table
        base_join_columns: Join columns in the base table
        candidate_column: The column from candidate table to evaluate
        target_column: Target column to predict (in base table)
        task_type: "regression" or "classification"
        candidate_join_columns: Join columns in candidate table (if None, same as base_join_columns)
        base_dir: Base directory containing datasets
        data_filename: CSV filename
        sample_size: Number of rows to sample for training (default: 1000)
    
    Returns:
        Dictionary with FI value and metadata
    """
    try:
        # Find real directory names
        real_base_name = find_dataset_dir(base_table_name, base_dir)
        real_candidate_name = find_dataset_dir(candidate_table_name, base_dir)
        
        # Load tables
        base_path = Path(base_dir) / real_base_name / data_filename
        candidate_path = Path(base_dir) / real_candidate_name / data_filename
        
        base_df = pd.read_csv(base_path, low_memory=False)
        candidate_df = pd.read_csv(candidate_path, low_memory=False)
        
        # Verify target column exists
        if target_column not in base_df.columns:
            return {
                "error": f"Target column '{target_column}' not found in base table",
                "feature_importance": 0.0
            }
        
        # Verify candidate column exists
        if candidate_column not in candidate_df.columns:
            return {
                "error": f"Candidate column '{candidate_column}' not found in candidate table",
                "feature_importance": 0.0
            }
        
        # Use same join columns if candidate_join_columns not specified
        if candidate_join_columns is None:
            candidate_join_columns = base_join_columns
        
        # Verify join columns exist
        missing_base = [col for col in base_join_columns if col not in base_df.columns]
        if missing_base:
            return {
                "error": f"Join columns {missing_base} not found in base table",
                "feature_importance": 0.0
            }
        
        missing_candidate = [col for col in candidate_join_columns if col not in candidate_df.columns]
        if missing_candidate:
            return {
                "error": f"Join columns {missing_candidate} not found in candidate table",
                "feature_importance": 0.0
            }
        
        # Perform join (inner join)
        merged = pd.merge(
            base_df,
            candidate_df[[*candidate_join_columns, candidate_column]],
            left_on=base_join_columns,
            right_on=candidate_join_columns,
            how='inner'
        )
        
        if len(merged) == 0:
            return {
                "error": "No rows matched after join",
                "feature_importance": 0.0
            }
        
        # Sample rows (take first sample_size rows after join)
        if len(merged) > sample_size:
            merged_sampled = merged.head(sample_size).copy()
        else:
            merged_sampled = merged.copy()
        
        # Prepare baseline features (base table columns, excluding target and join columns)
        baseline_features = [col for col in base_df.columns 
                           if col != target_column and col not in base_join_columns]
        
        if len(baseline_features) == 0:
            return {
                "error": "No baseline features available (base table only has target and join columns)",
                "feature_importance": 0.0
            }
        
        # Prepare datasets
        # Baseline: base table features only
        baseline_df = merged_sampled[baseline_features + [target_column]].copy()
        
        # Augmented: base table features + candidate column
        augmented_df = merged_sampled[baseline_features + [candidate_column] + [target_column]].copy()
        
        # Remove rows with missing target
        baseline_df = baseline_df.dropna(subset=[target_column])
        augmented_df = augmented_df.dropna(subset=[target_column])
        
        if len(baseline_df) < 10 or len(augmented_df) < 10:
            return {
                "error": f"Insufficient data after removing missing values (baseline: {len(baseline_df)}, augmented: {len(augmented_df)})",
                "feature_importance": 0.0
            }
        
        # Evaluate baseline and augmented models
        baseline_metric = _train_and_evaluate(
            baseline_df, target_column, task_type
        )
        augmented_metric = _train_and_evaluate(
            augmented_df, target_column, task_type
        )
        
        # Feature Importance = improvement in metric
        fi = augmented_metric - baseline_metric
        
        return float(fi)    
        
    except Exception as e:
        return {
            "error": str(e),
            "feature_importance": 0.0
        }


def _train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    task_type: str
) -> float:
    """
    Train model and return evaluation metric.
    - Regression: Linear Regression, metric = R2 score
    - Classification: XGBoost, metric = F1 score (weighted)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle missing values in features
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna('Unknown')
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Convert to numpy array
    X_array = X.values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y.values, test_size=0.2, random_state=42
    )
    
    # Train and evaluate
    if task_type == "regression":
        # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric = r2_score(y_test, y_pred)
    elif task_type == "classification":
        # XGBoost
        n_classes = len(np.unique(y_train))
        xgb_params = {
            'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic',
            'random_state': 42,
            'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss',
            'verbosity': 0
        }
        if n_classes > 2:
            xgb_params['num_class'] = n_classes
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'regression' or 'classification'")
    
    return float(metric)


def compute_utility_gain_from_params(
    base_table_name: str,
    candidate_table_name: str,
    base_join_columns: List[str],
    candidate_column: str,
    target_column: str,
    task_type: str,
    candidate_join_columns: Optional[List[str]] = None,
    base_dir: str = "datasets_omnimatch2",
    data_filename: str = "rows.csv",
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Compute Utility Gain by first calculating IQ and FI, then multiplying them.
    
    Returns a dictionary with utility_gain, iq, fi, and other metadata.
    """
    # Compute IQ
    try:
        iq_result = compute_integration_quality(
            base_table_name=base_table_name,
            candidate_table_name=candidate_table_name,
            base_join_columns=base_join_columns,
            candidate_join_columns=candidate_join_columns,
            base_dir=base_dir,
            data_filename=data_filename
        )
        
        # compute_integration_quality returns float, not dict
        if isinstance(iq_result, dict) and "error" in iq_result:
            return {
                "error": f"IQ computation failed: {iq_result['error']}",
                "utility_gain": 0.0,
                "iq": 0.0,
                "fi": 0.0
            }
        
        # iq_result is a float
        iq = float(iq_result) if not isinstance(iq_result, dict) else iq_result.get("iq", 0.0)
        
    except Exception as e:
        return {
            "error": f"IQ computation failed: {str(e)}",
            "utility_gain": 0.0,
            "iq": 0.0,
            "fi": 0.0
        }
    
    # Compute FI
    fi_result = compute_feature_importance(
        base_table_name=base_table_name,
        candidate_table_name=candidate_table_name,
        base_join_columns=base_join_columns,
        candidate_column=candidate_column,
        target_column=target_column,
        task_type=task_type,
        candidate_join_columns=candidate_join_columns,
        base_dir=base_dir,
        data_filename=data_filename,
        sample_size=sample_size
    )
    
    # Check if fi_result is dict and has error
    if isinstance(fi_result, dict) and "error" in fi_result:
        return {
            "error": f"FI computation failed: {fi_result['error']}",
            "utility_gain": 0.0,
            "iq": iq,
            "fi": 0.0
        }
    
    # fi_result should be dict with "feature_importance" key, but handle float case too
    if isinstance(fi_result, dict):
        fi = fi_result.get("feature_importance", 0.0)
    else:
        fi = float(fi_result) if isinstance(fi_result, (int, float)) else 0.0
    
    # Compute Utility Gain
    utility_gain = iq * fi
    
    return {
        "utility_gain": float(utility_gain),
        "iq": float(iq),
        "fi": float(fi),
        "candidate_column": candidate_column
    }