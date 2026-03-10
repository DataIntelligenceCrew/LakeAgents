import pandas as pd
import json
import re
import ast
from pathlib import Path
from typing import Dict, Any
from llm_agent_tools import find_dataset_dir, _train_and_evaluate


def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from model output (shared utility function)."""
    try:
        # First try standard JSON parsing
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON-like structure (handle Python dict with single quotes)
        try:
            # Use ast.literal_eval for Python dict syntax
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # Try regex to find JSON object
            match = re.search(r'\{.*\}', text, flags=re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    # Try standard JSON first
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try replacing single quotes with double quotes
                    json_str_fixed = json_str.replace("'", '"')
                    try:
                        return json.loads(json_str_fixed)
                    except json.JSONDecodeError:
                        # Last resort: use ast.literal_eval
                        return ast.literal_eval(json_str)
            raise ValueError(f"Failed to extract JSON from: {text[:200]}")

class JoinValidatorCallback:
    def __init__(self, join_table_df, base_dir, target_threshold=None, max_explosion_factor=None, config=None):
        self.join_table_df = join_table_df
        self.base_dir = base_dir
        
        # Use config if provided, otherwise use defaults
        if config is not None:
            self.threshold = config.match_rate_threshold
            self.max_explosion_factor = config.max_data_explosion_factor
        else:
            self.threshold = target_threshold if target_threshold is not None else 0.1
            self.max_explosion_factor = max_explosion_factor if max_explosion_factor is not None else 2.0
        
            self.is_valid = False
        self.match_rate = 0.0
        self.reason = None  # Initialize reason attribute

    def on_event(self, event):
        """
        Google ADK runner will call this for every event.
        We look for the event where state_delta contains our result.
        """
        if hasattr(event, 'actions') and event.actions.state_delta:
            delta = event.actions.state_delta
            if "join_column_choice" in delta:
                # 1. Extract the choice made by the agent
                choice = extract_json(delta["join_column_choice"])
                if choice.get("join_type") == "no_join_found":
                    return

                # 2. Perform the actual physical join verification
                self.verify(choice)

    def verify(self, choice, global_join_col, opendata_domain=None):
            try:
                cand_name = choice["candidate_table_name"]
                selected_cols = choice["selected_columns"]
                
                # ensure join columns is a list
                if isinstance(global_join_col, str):
                    global_join_col = [global_join_col]
                if isinstance(selected_cols, str):
                    selected_cols = [selected_cols]
                
                # check if join columns length matches
                if len(global_join_col) != len(selected_cols):
                    self.is_valid = False
                    self.reason = f"Join column length mismatch: left_on has {len(global_join_col)} columns, right_on has {len(selected_cols)} columns"
                    print(f"--- [Callback] Verification Failed: {self.reason} ---")
                    return

                cand_df = None
                if opendata_domain:
                    from agent_config_loader import load_config
                    from datalake_client import SocrataDatalakeClient
                    cfg = load_config()
                    client = SocrataDatalakeClient(cfg.get("data", {}).get("datalake", {}))
                    rows = client.read_data(cand_name, opendata_domain, max_rows=10000)
                    cand_df = pd.DataFrame(rows) if rows else None
                if cand_df is None or cand_df.empty:
                    real_cand_name = find_dataset_dir(cand_name, self.base_dir)
                    cand_df = pd.read_csv(Path(self.base_dir) / real_cand_name / "rows.csv", low_memory=False)
                
                # Create copies for case-insensitive matching
                join_df_copy = self.join_table_df.copy()
                cand_df_copy = cand_df.copy()
                
                # Convert join columns to lowercase for case-insensitive matching
                for col in global_join_col:
                    if col in join_df_copy.columns:
                        join_df_copy[col] = join_df_copy[col].astype(str).str.upper().str.strip()
                for col in selected_cols:
                    if col in cand_df_copy.columns:
                        cand_df_copy[col] = cand_df_copy[col].astype(str).str.upper().str.strip()
                
                # Simplified merge test (inner join) with case-insensitive matching
                merged = pd.merge(join_df_copy, cand_df_copy, left_on=global_join_col, right_on=selected_cols)
                
                self.match_rate = len(merged) / len(self.join_table_df)
                if self.match_rate < 0.1:
                    self.is_valid = False
                    self.reason = "Match rate too low (sparse data)"
                elif len(merged) > len(self.join_table_df) * self.max_explosion_factor:
                    self.is_valid = False
                    self.reason = f"Data explosion detected (One-to-Many fan-out: {len(merged)} > {len(self.join_table_df)} * {self.max_explosion_factor})"
                else:
                    self.is_valid = True
                    self.reason = "Join successful"
                
                print(f"--- [Callback] Physical Verification: {self.match_rate:.2%} match rate ---")
            except Exception as e:
                print(f"--- [Callback] Verification Failed: {e} ---")
                self.is_valid = False
                self.reason = str(e)

class AugmentValidatorCallback:
    def __init__(self, base_table_df, target_column, task_type, join_columns, base_dir, sample_size=None, config=None):
        self.base_table_df = base_table_df
        self.target_column = target_column
        self.task_type = task_type
        self.join_columns = join_columns if isinstance(join_columns, list) else [join_columns]
        self.base_dir = base_dir
        
        # Use config if provided, otherwise use defaults
        if config is not None:
            self.sample_size = config.sample_size
        else:
            self.sample_size = sample_size if sample_size is not None else 1000
        
        self.baseline_metric = None
    
    def _compute_baseline(self):
        """Compute baseline metric using only base table features (without augmentation)."""
        try:
            # Prepare baseline features (exclude target and join columns)
            baseline_features = [
                col for col in self.base_table_df.columns 
                if col != self.target_column and col not in self.join_columns
            ]
            
            if len(baseline_features) == 0:
                return None
            
            baseline_df = self.base_table_df[baseline_features + [self.target_column]].copy()
            baseline_df = baseline_df.dropna(subset=[self.target_column])
            
            if len(baseline_df) < 10:
                return None
            
            # Sample if needed
            if len(baseline_df) > self.sample_size:
                baseline_df = baseline_df.head(self.sample_size)
            
            # Run task
            self.baseline_metric = _train_and_evaluate(
                baseline_df, self.target_column, self.task_type
            )
            return self.baseline_metric
        except Exception as e:
            print(f"   ⚠️  Baseline computation failed: {e}")
            return None
    
    def verify(self, candidate_table_name, selected_columns, candidate_join_columns, opendata_domain=None):
        """
        Merge selected columns to base table and run ML task.
        Compare with baseline to show improvement.
        
        Args:
            candidate_table_name: Name of candidate table
            selected_columns: List of columns to join (augment columns)
            candidate_join_columns: Join columns in candidate table
            opendata_domain: If provided and local load fails, fetch from API
        
        Returns:
            Dictionary with metric result and improvement
        """
        try:
            # Compute baseline if not already computed
            if self.baseline_metric is None:
                print(f"   Computing baseline metric...")
                self._compute_baseline()
                if self.baseline_metric is not None:
                    print(f"   Baseline metric: {self.baseline_metric:.4f}")
            
            # Load candidate table (API if opendata_domain, else local)
            cand_df = None
            if opendata_domain:
                try:
                    from agent_config_loader import load_config
                    from datalake_client import SocrataDatalakeClient
                    cfg = load_config()
                    api_client = SocrataDatalakeClient(cfg.get("data", {}).get("datalake", {}))
                    rows = api_client.read_data(candidate_table_name, opendata_domain, max_rows=500000)
                    cand_df = pd.DataFrame(rows) if rows else None
                except Exception:
                    pass
            if cand_df is None or cand_df.empty:
                real_cand_name = find_dataset_dir(candidate_table_name, self.base_dir)
                cand_df = pd.read_csv(Path(self.base_dir) / real_cand_name / "rows.csv", low_memory=False)
            
            # Prepare columns to join
            if isinstance(candidate_join_columns, str):
                candidate_join_columns = [candidate_join_columns]
            
            columns_to_join = candidate_join_columns + selected_columns
            
            # Merge
            merged_df = pd.merge(
                self.base_table_df,
                cand_df[columns_to_join],
                left_on=self.join_columns,
                right_on=candidate_join_columns,
                how='inner'
            )
            
            # Sample if needed
            if len(merged_df) > self.sample_size:
                merged_df = merged_df.head(self.sample_size)
            
            # Remove rows with missing target
            merged_df = merged_df.dropna(subset=[self.target_column])
            
            # Run task on augmented data
            augmented_metric = _train_and_evaluate(
                merged_df, self.target_column, self.task_type
            )
            
            # Calculate improvement
            improvement = None
            improvement_percent = None
            is_valid = False  
            
            if self.baseline_metric is not None:
                improvement = augmented_metric - self.baseline_metric
                if self.baseline_metric != 0:
                    improvement_percent = (improvement / abs(self.baseline_metric)) * 100
                else:
                    improvement_percent = improvement * 100 if improvement != 0 else 0
                      
                is_valid = improvement > 0
            
            else:
                is_valid = False
            
            # Get feature counts
            base_features = [
                col for col in self.base_table_df.columns 
                if col != self.target_column and col not in self.join_columns
            ]
            total_features = len(base_features) + len(selected_columns)
            
            return {
                "baseline_metric": self.baseline_metric,
                "augmented_metric": augmented_metric,
                "improvement": improvement,
                "improvement_percent": improvement_percent,
                "metric": augmented_metric,
                "task_type": self.task_type,
                "rows_used": len(merged_df),
                "base_features_count": len(base_features),
                "augment_features_count": len(selected_columns),
                "total_features_count": total_features,
                "is_valid": is_valid  
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "metric": None,
                "baseline_metric": self.baseline_metric,
                "is_valid": False  
            }