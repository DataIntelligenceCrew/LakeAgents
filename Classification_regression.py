#!/usr/bin/env python3
"""
Classification and Regression ML Pipeline
Incremental prediction with progressive subtable joining
"""

import sys
import os
import pandas as pd
import json
import numpy as np
import random
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from Data_preparation import load_verified_tables_from_file, get_verified_tables
from tools.text_integration import embed_texts_with_fasttext

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Load configuration
def load_config():
    """Load configuration from data_config.yaml"""
    config_path = Path("configs/data_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('layer3', {}).get('positive_threshold', 0.01)
    return 0.01  # Default threshold

# Get threshold from config
POSITIVE_THRESHOLD = load_config()


def load_analysis_results():
    """Load analysis results from JSON file and filter out failed datasets"""
    with open('analysis_results_optimized.json', 'r') as f:
        all_results = json.load(f)
    
    # Filter to only include successful datasets
    filtered_results = []
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
        
        filtered_results.append(result)
    
    if skipped_count > 0:
        print(f"  Filtered out {skipped_count} failed datasets")
    
    return filtered_results

def get_dataset_info(dataset_id, analysis_results):
    """Get complete dataset information including subtables and join columns"""
    for result in analysis_results:
        if result['dataset'] == dataset_id:
            return result['result']
    return None

def load_subtables(dataset_id, dataset_info):
    """Load specific subtables for a dataset based on analysis_results"""
    subtables = {}
    subtable_dir = f"subtables/{dataset_id}"
    
    if not os.path.exists(subtable_dir):
        print(f"  Subtables directory not found: {subtable_dir}")
        return None
    
    # New format: candidate_table and non_candidate_table
    if 'candidate_table' in dataset_info and 'non_candidate_table' in dataset_info:
        ct_conf = dataset_info['candidate_table']
        nct_conf = dataset_info['non_candidate_table']
        ct_name = ct_conf.get('name', 'Candidate_Features')
        nct_name = nct_conf.get('name', 'NonCandidate_With_Target')
        
        for name in [ct_name, nct_name]:
            file_path = f"{subtable_dir}/{name}.csv"
            if not os.path.exists(file_path):
                print(f"    Warning: Table file not found: {file_path}")
                continue
            try:
                df = pd.read_csv(file_path)
                subtables[name] = df
                print(f"    Loaded table '{name}': {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"    Error loading {file_path}: {e}")
    else:
        # Legacy format: subtable_1, subtable_2, etc.
        subtable_keys = sorted([k for k in dataset_info.keys() if k.startswith('subtable_')])
        
        for subtable_key in subtable_keys:
            subtable_info = dataset_info[subtable_key]
            subtable_name = subtable_info['name']  # Get the name from analysis_results
            
            # Construct the file path
            file_path = f"{subtable_dir}/{subtable_name}.csv"
            
            if not os.path.exists(file_path):
                print(f"    Warning: Subtable file not found: {file_path}")
                continue
            
            try:
                df = pd.read_csv(file_path)
                subtables[subtable_name] = df
                print(f"    Loaded subtable '{subtable_name}': {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"    Error loading {file_path}: {e}")
    
    return subtables

def preprocess_data(df, target_column,task_type):
    """Preprocess the data for ML tasks"""
    # Separate features and target
    if target_column not in df.columns:
        print(f"    Warning: Target column '{target_column}' not found in data")
        return None, None, None, None
    
    # #region agent log
    import json as _json; _ts = __import__('time').time_ns() // 1000000
    with open('/localdisk3/username49/opendata/.cursor/debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_J2","timestamp":_ts,"location":"Classification_regression.py:135","message":"df at function entry","data":{"shape":[int(x) for x in df.shape],"columns":list(df.columns),"dtypes":{col:str(dtype) for col,dtype in df.dtypes.items()},"target_column":target_column,"task_type":task_type},"hypothesisId":"J,M"}) + '\n')
    # #endregion
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # #region agent log
    import json as _json; _ts = __import__('time').time_ns() // 1000000
    _y_sample = [str(v) if not pd.api.types.is_numeric_dtype(y) else (float(v) if hasattr(v,'item') else v) for v in (y[:5] if len(y)>0 else [])]
    with open('/localdisk3/username49/opendata/.cursor/debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_L1","timestamp":_ts,"location":"Classification_regression.py:145","message":"X,y after separation","data":{"X_shape":[int(x) for x in X.shape],"X_dtypes":{col:str(dtype) for col,dtype in X.dtypes.items()},"y_dtype":str(y.dtype),"y_sample":_y_sample,"y_stats":{"mean":float(y.mean()) if hasattr(y,'mean') and pd.api.types.is_numeric_dtype(y) else None,"min":float(y.min()) if hasattr(y,'min') and pd.api.types.is_numeric_dtype(y) else None,"max":float(y.max()) if hasattr(y,'max') and pd.api.types.is_numeric_dtype(y) else None}},"hypothesisId":"L"}) + '\n')
    # #endregion

    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"    Removing {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)

    # Handle missing values in features
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna('Unknown')
    
    # Handle missing values in target
    y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 'Unknown')

    nan_threshold = 0.5 
    nan_ratio = X.isnull().sum(axis=1) / len(X.columns)
    valid_mask = nan_ratio <= nan_threshold
    removed_count = (~valid_mask).sum()
    if removed_count > 0:
        print(f"    Removing {removed_count} rows with >{nan_threshold*100}% NaN values")
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    # Remove rows where features still have NaN (in case mean was NaN)
    valid_mask = ~X.isnull().any(axis=1)
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    
    if len(X) == 0:
        print(f"    Error: No valid samples after removing NaN")
        return None, None, None, None
    
    # Encode categorical variables in features.
    # For date-like string columns, use FastText embeddings (same style as manual experiment).
    object_columns = X.select_dtypes(include=['object']).columns
    for col in object_columns:
        col_text = X[col].fillna('Unknown').astype(str)
        parsed_dates = pd.to_datetime(col_text, errors='coerce')
        date_like_ratio = float(parsed_dates.notna().mean()) if len(col_text) > 0 else 0.0
        if date_like_ratio >= 0.5:
            try:
                model_path = str(Path(__file__).resolve().parent / "fasttext.bin")
                emb = embed_texts_with_fasttext(col_text.tolist(), model_path=model_path)
                emb_cols = [f"{col}__ft_{i}" for i in range(emb.shape[1])]
                emb_df = pd.DataFrame(emb, columns=emb_cols, index=X.index)
                X = X.drop(columns=[col])
                X = pd.concat([X, emb_df], axis=1)
                continue
            except Exception:
                # Fallback to previous behavior if FastText model is unavailable.
                pass

        numeric_vals = pd.to_numeric(col_text, errors='coerce')
        if numeric_vals.notna().mean() > 0.5:
            X[col] = numeric_vals.fillna(numeric_vals.mean())
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(col_text)
    
    # #region agent log
    import json as _json; _ts = __import__('time').time_ns() // 1000000
    with open('/localdisk3/username49/opendata/.cursor/debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_K2","timestamp":_ts,"location":"Classification_regression.py:177","message":"after encoding object columns","data":{"X_dtypes":{col:str(dtype) for col,dtype in X.dtypes.items()}},"hypothesisId":"K,N"}) + '\n')

    
    # Encode target variable
    # For classification: always encode to ensure 0-based consecutive integers
    # For regression: convert to float
    if task_type == 'classification':
        target_encoder = LabelEncoder()
        # Convert to string first to handle any data type
        y_encoded = target_encoder.fit_transform(y.astype(str))
        
        # Verify encoding
        unique_classes = np.unique(y_encoded)
        print(f"    Target classes: {len(unique_classes)} unique values, range [{unique_classes.min()}, {unique_classes.max()}]")
        
        # Double check: should be consecutive from 0
        expected_classes = np.arange(len(unique_classes))
        if not np.array_equal(unique_classes, expected_classes):
            print(f"    WARNING: Classes are not consecutive! Re-encoding...")
            class_map = {old: new for new, old in enumerate(unique_classes)}
            y_encoded = np.array([class_map[val] for val in y_encoded])
        
        # Filter out classes with too few samples (< 5)
        min_samples_per_class = 5
        class_counts = pd.Series(y_encoded).value_counts()
        valid_classes = class_counts[class_counts >= min_samples_per_class].index
        
        if len(valid_classes) < len(class_counts):
            removed_classes = len(class_counts) - len(valid_classes)
            removed_samples = sum(class_counts[class_counts < min_samples_per_class])
            print(f"    Filtering {removed_classes} classes with <{min_samples_per_class} samples ({removed_samples} total samples)")
            
            mask = np.isin(y_encoded, valid_classes)
            y_encoded = y_encoded[mask]
            X = X[mask].reset_index(drop=True)
            
            # Remap class labels to consecutive 0, 1, 2, ...
            unique_classes = np.unique(y_encoded)
            class_map = {old: new for new, old in enumerate(unique_classes)}
            y_encoded = np.array([class_map[val] for val in y_encoded])
            
            print(f"    After filtering: {len(unique_classes)} classes, {len(X)} samples")
    else:  # regression
        target_encoder = None
        # For regression, try to convert to float, handle non-numeric values
        y_encoded = pd.to_numeric(y, errors='coerce')  # convert to float, handle non-numeric values
        
        # #region agent log
        import json as _json; _ts = __import__('time').time_ns() // 1000000
        _y_enc_sample = [float(v) if hasattr(v,'item') else v for v in (y_encoded[:5] if len(y_encoded)>0 else [])]
        with open('/localdisk3/username49/opendata/.cursor/debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_L2","timestamp":_ts,"location":"Classification_regression.py:235","message":"y_encoded for regression","data":{"y_encoded_dtype":str(y_encoded.dtype),"y_encoded_sample":_y_enc_sample,"y_encoded_stats":{"mean":float(y_encoded.mean()),"min":float(y_encoded.min()),"max":float(y_encoded.max()),"nan_count":int(pd.isna(y_encoded).sum())}},"hypothesisId":"L"}) + '\n')
        # #endregion
        
        # Remove rows where target is NaN
        valid_mask = ~pd.isna(y_encoded)
        y_encoded = y_encoded[valid_mask]
        X = X[valid_mask].reset_index(drop=True)
        
        if len(X) == 0:
            print(f"    Error: No valid samples after removing invalid target values")
            return None, None, None, None
        
        y_encoded = y_encoded.astype(float)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = X.select_dtypes(include=[np.number]).columns
    if len(numerical_columns) > 0:
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    # #region agent log
    import json as _json; _ts = __import__('time').time_ns() // 1000000
    _y_final_sample = [float(v) if hasattr(v,'item') else (int(v) if isinstance(v,(np.integer,np.int64)) else v) for v in (y_encoded[:5] if len(y_encoded)>0 else [])]
    with open('/localdisk3/username49/opendata/.cursor/debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_K3","timestamp":_ts,"location":"Classification_regression.py:259","message":"final X,y before return","data":{"X_shape":[int(x) for x in X.shape],"X_columns":list(X.columns),"X_dtypes":{col:str(dtype) for col,dtype in X.dtypes.items()},"y_dtype":str(y_encoded.dtype),"y_shape":[len(y_encoded)],"y_sample":_y_final_sample,"numerical_columns":list(numerical_columns)},"hypothesisId":"K"}) + '\n')
    # #endregion
    
    return X, y_encoded, target_encoder, scaler

def run_classification_task(X, y, target_encoder):
    """Run classification task using a shallow Decision Tree."""
    # Determine number of classes
    n_classes = len(np.unique(y))
    
    # Split data with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    
    # Use shallow trees to keep greedy feature evaluation fast.
    max_depth = 5 if n_classes > 2 else 3
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def run_classification_task_xgboost(X, y, target_encoder):
    """Run classification task using XGBoost (for final full evaluation)."""
    n_classes = len(np.unique(y))

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    xgb_params = {
        'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic',
        'random_state': 42,
        'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss',
        'verbosity': 0,
    }
    if n_classes > 2:
        xgb_params['num_class'] = n_classes

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }

def run_regression_task(X, y):
    """Run regression task using Linear Regression"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2_score': r2
    }

def verify_final_join(dataset_id, final_df):
    """Verify that the final joined table matches the verified joined table"""
    # Load the verified joined table
    joined_file = f"joined_tables/{dataset_id}_joined.csv"
    
    if not os.path.exists(joined_file):
        print(f"    Warning: Verified joined table not found: {joined_file}")
        return False
    
    try:
        verified_df = pd.read_csv(joined_file)
        
        # Compare dimensions
        print(f"    Final joined table: {len(final_df)} rows, {len(final_df.columns)} columns")
        print(f"    Verified table:     {len(verified_df)} rows, {len(verified_df.columns)} columns")
        
        if len(final_df) != len(verified_df):
            print(f"    ❌ Row count mismatch!")
            return False
        
        if len(final_df.columns) != len(verified_df.columns):
            print(f"    ❌ Column count mismatch!")
            return False
        
        # Compare column names (order might be different)
        final_cols = set(final_df.columns)
        verified_cols = set(verified_df.columns)
        
        if final_cols != verified_cols:
            print(f"    ❌ Column names mismatch!")
            missing_in_final = verified_cols - final_cols
            extra_in_final = final_cols - verified_cols
            if missing_in_final:
                print(f"       Missing in final: {missing_in_final}")
            if extra_in_final:
                print(f"       Extra in final: {extra_in_final}")
            return False
        
        print(f"    ✓ Dimensions match!")
        print(f"    ✓ Column names match!")
        print(f"    ✓ Final joined table is losslessly reconstructed!")
        return True
        
    except Exception as e:
        print(f"    Error during verification: {e}")
        return False
        
def run_incremental_ml_tasks(verified_tables):
    """Run ML tasks with incremental subtable joining"""
    print(f"\n=== RUNNING INCREMENTAL ML TASKS ===")

    analysis_results = load_analysis_results()
    all_results = {}
    failed_tables = {}  # Track failed tables and reasons

    # Add counters for task types
    classification_count = 0
    regression_count = 0

    for table_name in verified_tables:
        print(f"\n{'='*60}")
        print(f"Processing {table_name}...")
        print(f"{'='*60}")

        # Get dataset info
        dataset_info = get_dataset_info(table_name, analysis_results)
        if not dataset_info:
            reason = "No dataset info found in analysis_results"
            print(f"  ❌ {reason}")
            failed_tables[table_name] = reason
            continue

        # check dataset_info status
        if dataset_info.get('status') == 'failed':
            reason = f"Dataset analysis failed: {dataset_info.get('error', 'No error message')[:200]}"
            print(f"  ❌ {reason}")
            failed_tables[table_name] = reason
            continue

        # try to get required fields
        try:
            target_column = dataset_info['target_column']['name']
            task_type = dataset_info['target_column']['task_type']
            join_columns = dataset_info.get('join_columns', [])
        except KeyError as e:
            reason = f"Missing required field: {e}"
            print(f"  ❌ {reason}")
            failed_tables[table_name] = reason
            continue
        
        print(f"  Target column: {target_column}")
        print(f"  Task type: {task_type}")
        print(f"  Join columns: {join_columns}")

        # Load all subtables
        print(f"\n  Loading subtables...")
        subtables = load_subtables(table_name, dataset_info)
        if not subtables:
            reason = "Failed to load subtables (directory or files not found)"
            print(f"  ❌ {reason}")
            failed_tables[table_name] = reason
            continue

        # New format: candidate_table and non_candidate_table
        if 'candidate_table' in dataset_info and 'non_candidate_table' in dataset_info:
            ct_conf = dataset_info['candidate_table']
            nct_conf = dataset_info['non_candidate_table']
            ct_name = ct_conf.get('name', 'Candidate_Features')
            nct_name = nct_conf.get('name', 'NonCandidate_With_Target')
            
            # non_candidate_table contains the target (label table)
            # candidate_table contains features (feature table)
            if nct_name not in subtables:
                reason = f"Non-candidate table '{nct_name}' not found in loaded tables"
                print(f"  ❌ {reason}")
                failed_tables[table_name] = reason
                continue
            
            if ct_name not in subtables:
                reason = f"Candidate table '{ct_name}' not found in loaded tables"
                print(f"  ❌ {reason}")
                failed_tables[table_name] = reason
                continue
            
            target_subtable_name = nct_name
            target_subtable_df = subtables[nct_name]
            
            # Verify target column is in non_candidate_table
            if target_column not in target_subtable_df.columns:
                reason = f"Target column '{target_column}' not found in non_candidate_table '{nct_name}'"
                print(f"  ❌ {reason}")
                failed_tables[table_name] = reason
                continue
            
            print(f"\n  Using new table structure:")
            print(f"    Target table (non_candidate): '{nct_name}' - contains target column")
            print(f"    Feature table (candidate): '{ct_name}' - contains augmentation columns")
            
            candidate_table_df = subtables[ct_name]
            available_columns = [col for col in candidate_table_df.columns if col not in join_columns]
            
        else:
            # Legacy format: subtable_1, subtable_2, etc.
            subtable_keys = sorted([k for k in dataset_info.keys() if k.startswith('subtable_')])
            
            # Find which subtable contains the target column
            target_subtable_key = None
            for subtable_key in subtable_keys:
                subtable_info = dataset_info[subtable_key]
                subtable_name = subtable_info['name']
                if subtable_name in subtables:
                    if target_column in subtables[subtable_name].columns:
                        target_subtable_key = subtable_key
                        print(f"\n  Target column found in subtable: '{subtable_name}' ({subtable_key})")
                        break
            
            if not target_subtable_key:
                reason = f"Target column '{target_column}' not found in any subtable"
                print(f"  ❌ {reason}")
                failed_tables[table_name] = reason
                continue
            
            print(f"\n  Found {len(subtable_keys)} subtables: {subtable_keys}")

            if len(subtable_keys) != 2:
                reason = f"Expected 2 subtables, but found {len(subtable_keys)}"
                print(f"  ❌ {reason}")
                failed_tables[table_name] = reason
                continue

            target_subtable_name = dataset_info[target_subtable_key]['name']
            target_subtable_df = subtables[target_subtable_name]

            non_target_subtable_key = [k for k in subtable_keys if k != target_subtable_key][0]
            non_target_subtable_name = dataset_info[non_target_subtable_key]['name']
            non_target_subtable_df = subtables[non_target_subtable_name]

            available_columns = [col for col in non_target_subtable_df.columns if col not in join_columns]

        # Incremental prediction
        table_results = {
            'target_column': target_column,
            'task_type': task_type,
            'incremental_results': [],
            'positive_pairs': [],  # increase >= 0.01
            'negative_pairs': [],  # metric_diff < 0.01 (no improvement, includes undefined)
        }

        # Step 0: Baseline - using only target subtable
        print(f"\n  --- Step 0 (Baseline): Using only '{target_subtable_name}' ---")
        baseline_df = target_subtable_df.copy()
        print(f"    Data shape: {len(baseline_df)} rows, {len(baseline_df.columns)} columns")

        X, y, target_encoder, scaler = preprocess_data(baseline_df, target_column, task_type)

        if X is not None and len(X) > 0:
            print(f"    Running {task_type} task...")
            
            # #region agent log
            import json as _json; _ts = __import__('time').time_ns() // 1000000
            with open('/localdisk3/username49/opendata/.cursor/debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_O1","timestamp":_ts,"location":"Classification_regression.py:536","message":"before running task","data":{"task_type":task_type,"X_shape":[int(x) for x in X.shape],"y_shape":[len(y)],"y_unique_count":int(len(np.unique(y)))},"hypothesisId":"O,P"}) + '\n')
            # #endregion
            
            if task_type == 'classification':
                try:
                    metrics = run_classification_task(X, y, target_encoder)
                    baseline_metric_value = metrics['f1_score']
                    metric_name = 'f1_score'
                    classification_count += 1
                    
                    # #region agent log
                    import json as _json; _ts = __import__('time').time_ns() // 1000000
                    _metrics_serializable = {k: (float(v) if hasattr(v,'item') else v) for k,v in metrics.items()}
                    with open('/localdisk3/username49/opendata/.cursor/debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_P1","timestamp":_ts,"location":"Classification_regression.py:548","message":"classification metrics computed","data":{"metrics":_metrics_serializable,"baseline_metric_value":float(baseline_metric_value)},"hypothesisId":"P,Q"}) + '\n')
                    # #endregion
                except Exception as e:
                    # #region agent log
                    import json as _json; _ts = __import__('time').time_ns() // 1000000
                    with open('/localdisk3/username49/opendata/.cursor/debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_P2","timestamp":_ts,"location":"Classification_regression.py:556","message":"classification task failed","data":{"error":str(e),"error_type":type(e).__name__},"hypothesisId":"P"}) + '\n')
                    # #endregion
                    raise
            elif task_type == 'regression':
                metrics = run_regression_task(X, y)
                baseline_metric_value = metrics['r2_score']
                metric_name = 'r2_score'
                regression_count += 1
            else:
                print(f"    Unknown task type: {task_type}")
                continue
            
            # save baseline metric
            table_results['baseline_metric'] = baseline_metric_value
            
            step_result = {
                'step': 0,
                'added_column': None,
                'num_features': X.shape[1],
                'num_samples': len(X),
                'metrics': metrics,
                'metric_diff': 0.0,  # baseline difference
                'category': 'baseline'
            }
            table_results['incremental_results'].append(step_result)
            
            # #region agent log
            import json as _json; _ts = __import__('time').time_ns() // 1000000
            _metrics_for_json = {k: (float(v) if hasattr(v,'item') else v) for k,v in metrics.items()}
            with open('/localdisk3/username49/opendata/.cursor/debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_Q1","timestamp":_ts,"location":"Classification_regression.py:564","message":"before saving to table_results","data":{"metrics":_metrics_for_json,"baseline_metric_value":float(baseline_metric_value) if baseline_metric_value is not None else None,"metric_name":metric_name},"hypothesisId":"Q"}) + '\n')
            # #endregion
            
            print(f"    Metrics:")
            for metric, value in metrics.items():
                print(f"      {metric}: {value:.4f}")
            print(f"    Baseline {metric_name}: {baseline_metric_value:.4f}")
        else:
            # #region agent log
            import json as _json; _ts = __import__('time').time_ns() // 1000000
            with open('/localdisk3/username49/opendata/.cursor/debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_O2","timestamp":_ts,"location":"Classification_regression.py:577","message":"preprocess returned None or empty","data":{"X_is_none":X is None,"X_len":len(X) if X is not None else 0,"baseline_df_shape":[int(x) for x in baseline_df.shape]},"hypothesisId":"O"}) + '\n')
            # #endregion
            
            reason = f"Failed to preprocess baseline data (target subtable: {target_subtable_name}, rows: {len(baseline_df)}, cols: {len(baseline_df.columns)})"
            print(f"    ❌ {reason}")
            failed_tables[table_name] = reason
            continue

        # Step 1-N: incremental prediction by adding single column
        for idx, column_name in enumerate(available_columns, start=1):
            print(f"\n  --- Step {idx}: Adding single column '{column_name}' ---")
            
            # always start from target_subtable_df, only join current column
            temp_df = target_subtable_df.copy()
            
            # select join_columns + current column from candidate/non-target table
            if 'candidate_table' in dataset_info and 'non_candidate_table' in dataset_info:
                # New format: join from candidate_table
                columns_to_join = join_columns + [column_name]
                column_data = candidate_table_df[columns_to_join]
            else:
                # Legacy format: join from non_target_subtable
                columns_to_join = join_columns + [column_name]
                column_data = non_target_subtable_df[columns_to_join]
            
            # Join
            if join_columns:
                temp_df = temp_df.merge(column_data, on=join_columns, how='inner')
                print(f"    After joining: {len(temp_df)} rows, {len(temp_df.columns)} columns")
            else:
                print(f"    No join columns specified, skipping")
                continue
            
            # Preprocess and run ML
            X, y, target_encoder, scaler = preprocess_data(temp_df, target_column, task_type)
            
            if X is None or len(X) == 0:
                print(f"    Failed to preprocess data")
                continue
            
            print(f"    Running {task_type} task...")
            if task_type == 'classification':
                metrics = run_classification_task(X, y, target_encoder)
                current_metric_value = metrics['f1_score']
            elif task_type == 'regression':
                metrics = run_regression_task(X, y)
                current_metric_value = metrics['r2_score']
            else:
                print(f"    Unknown task type: {task_type}")
                continue
            
            # calculate the difference between current and baseline
            metric_diff = current_metric_value - baseline_metric_value
            
            # classification
            if metric_diff >= POSITIVE_THRESHOLD:
                category = 'positive'
                pair_name = f"{target_subtable_name}+{column_name}"
                table_results['positive_pairs'].append(pair_name)
            else:  # metric_diff < POSITIVE_THRESHOLD (includes all undefined samples)
                category = 'negative'
                pair_name = f"{target_subtable_name}+{column_name}"
                table_results['negative_pairs'].append(pair_name)
            
            # Store results
            step_result = {
                'step': idx,
                'added_column': column_name,
                'pair_name': pair_name,
                'num_features': X.shape[1],
                'num_samples': len(X),
                'metrics': metrics,
                'metric_diff': metric_diff,
                'category': category
            }
            table_results['incremental_results'].append(step_result)
            
            print(f"    Metrics:")
            for metric, value in metrics.items():
                print(f"      {metric}: {value:.4f}")
            print(f"    {metric_name} diff: {metric_diff:+.4f} [{category}]")

        # print statistics for this table
        print(f"\n  --- Statistics for {table_name} ---")
        print(f"    Positive pairs (diff >= {POSITIVE_THRESHOLD}): {len(table_results['positive_pairs'])}")
        print(f"    Negative pairs (diff < {POSITIVE_THRESHOLD}, includes undefined): {len(table_results['negative_pairs'])}")

        all_results[table_name] = table_results

        # Calculate global statistics
    total_tables_tested = len(all_results)
    total_tables_failed = len(failed_tables)
    total_tables_verified = len(verified_tables)
    total_positive = sum(len(result['positive_pairs']) for result in all_results.values())
    total_negative = sum(len(result['negative_pairs']) for result in all_results.values())
    total_pairs = total_positive + total_negative
    
    # Print global statistics
    print(f"\n\n{'='*60}")
    print(f"=== GLOBAL STATISTICS ===")
    print(f"{'='*60}")
    print(f"Total verified tables: {total_tables_verified}")
    print(f"Successfully tested: {total_tables_tested} ({total_tables_tested/total_tables_verified*100:.1f}%)")
    print(f"Failed: {total_tables_failed} ({total_tables_failed/total_tables_verified*100:.1f}%)")
    print(f"")
    print(f"Among successful tests:")
    print(f"  Total positive pairs (diff >= {POSITIVE_THRESHOLD}): {total_positive} ({total_positive/total_pairs*100 if total_pairs > 0 else 0:.1f}%)")
    print(f"  Total negative pairs (diff < {POSITIVE_THRESHOLD}, includes undefined): {total_negative} ({total_negative/total_pairs*100 if total_pairs > 0 else 0:.1f}%)")
    print(f"  Total column additions tested: {total_pairs}")
    print(f"Classification tasks: {classification_count}")
    print(f"Regression tasks: {regression_count}")
    print(f"Total tasks: {classification_count + regression_count}")

    # Print failed tables details
    if failed_tables:
        print(f"\n{'='*60}")
        print(f"=== FAILED TABLES DETAILS ===")
        print(f"{'='*60}")
        for table_name, reason in failed_tables.items():
            print(f"{table_name}:")
            print(f"  Reason: {reason}")
    
    # Save results to JSON file
    print(f"\n{'='*60}")
    print(f"=== SAVING RESULTS TO JSON ===")
    print(f"{'='*60}")

    output_results = {
        'global_statistics': {
            'total_tables_verified': total_tables_verified,
            'total_tables_tested': total_tables_tested,
            'total_tables_failed': total_tables_failed,
            'success_rate': f"{total_tables_tested/total_tables_verified*100:.1f}%",
            'total_positive_pairs': total_positive,
            'total_negative_pairs': total_negative,
            'total_pairs': total_pairs
        },
        'failed_tables': failed_tables,
        'per_table_results': {}
    }
    
    for table_name, result_info in all_results.items():
        output_results['per_table_results'][table_name] = {
            'target_column': result_info['target_column'],
            'task_type': result_info['task_type'],
            'baseline_metric': result_info['baseline_metric'],
            'metric_name': 'f1_score' if result_info['task_type'] == 'classification' else 'r2_score',
            'positive_count': len(result_info['positive_pairs']),
            'negative_count': len(result_info['negative_pairs']),
            'positive_pairs': result_info['positive_pairs'],
            'negative_pairs': result_info['negative_pairs'],
            'detailed_results': result_info['incremental_results']
        }

    # Save to JSON file
    with open('incremental_column_results.json', 'w') as f:
        json.dump(output_results, f, indent=2)

    print(f"Results saved to 'incremental_column_results.json'")
    
    return all_results

def main():
    """Main function"""
    print("=== INCREMENTAL ML PIPELINE STARTED ===")

    # Load verified tables
    print("=== LOADING VERIFIED TABLES ===")
    verified_tables = load_verified_tables_from_file()

    if not verified_tables:
        print("No saved verified tables found. Running compare step...")
        verified_tables = get_verified_tables()
        if verified_tables:
            from Data_preparation import save_verified_tables_to_file
            save_verified_tables_to_file(verified_tables)
    
    if not verified_tables:
        print("No verified tables found.")
        sys.exit(1)
    
    print(f"Found {len(verified_tables)} verified tables")
    
    # Run incremental ML tasks
    results = run_incremental_ml_tasks(verified_tables)
    
    print(f"\n=== INCREMENTAL ML PIPELINE COMPLETED ===")

if __name__ == "__main__":
    main()
