from typing import Any, Dict, List, Optional, Set, Tuple

import json
import numpy as np
import pandas as pd


def _to_hashable_value(v: Any) -> Any:
    """Convert nested/unhashable values (dict/list/set/tuple) to stable strings."""
    if isinstance(v, dict):
        try:
            return json.dumps(v, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(v)
    if isinstance(v, (list, tuple, set)):
        try:
            return json.dumps(list(v), ensure_ascii=False)
        except Exception:
            return str(v)
    return v


def _safe_nunique_non_null(series: pd.Series) -> int:
    """nunique that works with unhashable Python objects."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0
    safe = non_null.map(_to_hashable_value)
    return int(safe.nunique())


def _top1_ratio(series: pd.Series) -> float:
    """Compute dominant-value ratio on non-null values."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    freq = non_null.astype(str).value_counts(normalize=True, dropna=True)
    if len(freq) == 0:
        return 0.0
    return float(freq.iloc[0])


def apply_hard_quality_rules(
    df: pd.DataFrame,
    protected_columns: Optional[List[str]] = None,
    missing_threshold: float = 0.4,
    top1_ratio_threshold: float = 0.8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply deterministic hard-quality rules on non-protected columns:
    1) missing_ratio > missing_threshold -> drop
    2) unique_non_null <= 1 -> drop
    3) top1_ratio > top1_ratio_threshold -> drop
    """
    protected: Set[str] = set(protected_columns or [])
    to_drop: Set[str] = set()
    details: List[Dict[str, Any]] = []

    for col in df.columns:
        if col in protected:
            continue

        s = df[col]
        n_rows = len(s)
        missing_ratio = float(s.isna().mean()) if n_rows > 0 else 0.0
        unique_non_null = _safe_nunique_non_null(s)
        top1 = _top1_ratio(s)

        reasons: List[str] = []
        if missing_ratio > missing_threshold:
            reasons.append("missing_ratio")
        if unique_non_null <= 1:
            reasons.append("constant_or_singleton")
        if top1 > top1_ratio_threshold:
            reasons.append("near_constant_top1")

        if reasons:
            to_drop.add(col)
            details.append(
                {
                    "column": col,
                    "reasons": reasons,
                    "missing_ratio": round(missing_ratio, 4),
                    "unique_non_null": unique_non_null,
                    "top1_ratio": round(top1, 4),
                }
            )

    cleaned_df = df.drop(columns=sorted(to_drop), errors="ignore")
    report = {
        "missing_threshold": missing_threshold,
        "top1_ratio_threshold": top1_ratio_threshold,
        "protected_columns": sorted(protected),
        "dropped_columns": sorted(to_drop),
        "dropped_count": len(to_drop),
        "details": details,
        "columns_before": int(len(df.columns)),
        "columns_after": int(len(cleaned_df.columns)),
    }
    return cleaned_df, report


def _infer_dtype(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical_or_text"


def _outlier_rate_iqr(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 8:
        return 0.0
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return 0.0
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = ((s < low) | (s > high)).sum()
    return float(outliers / len(s))


def column_quality_metrics(values: List[Any]) -> Dict[str, Any]:
    """Compute quality metrics for one column from raw values."""
    s = pd.Series(values, dtype="object")
    n = len(s)
    missing_rate = float(s.isna().mean()) if n > 0 else 0.0
    non_null = s.dropna()
    unique_non_null = _safe_nunique_non_null(s)
    rel_cardinality = float(unique_non_null / len(non_null)) if len(non_null) > 0 else 0.0
    top1 = _top1_ratio(s)

    numeric = pd.to_numeric(s, errors="coerce")
    n_numeric = int(numeric.notna().sum())
    variance = float(numeric.var()) if n_numeric >= 2 else 0.0
    outlier_rate = _outlier_rate_iqr(s)

    return {
        "missing_rate": round(missing_rate, 4),
        "outlier_rate": round(outlier_rate, 4),
        "variance": round(variance, 6),
        "relative_cardinality": round(rel_cardinality, 4),
        "unique_non_null": unique_non_null,
        "top1_ratio": round(top1, 4),
        "dtype_guess": "numeric" if n_numeric >= max(3, int(0.6 * max(len(non_null), 1))) else "categorical_or_text",
        "sample_values": [None if pd.isna(v) else str(v) for v in non_null.head(5).tolist()],
    }


def compute_table_quality_summary(
    df: pd.DataFrame,
    column_descriptions: Optional[Dict[str, str]] = None,
    exclude_columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Build per-column quality summary for LLM decision."""
    exclude = set(exclude_columns or [])
    descs = column_descriptions or {}
    summary: List[Dict[str, Any]] = []
    for col in df.columns:
        if col in exclude:
            continue
        metrics = column_quality_metrics(df[col].tolist())
        summary.append(
            {
                "column_name": col,
                "column_description": descs.get(col, ""),
                "dtype": _infer_dtype(df[col]),
                **metrics,
            }
        )
    return summary


def winsorize_values(values: List[Any], lower_q: float = 0.05, upper_q: float = 0.95) -> Dict[str, Any]:
    """Preview winsorization for a numeric column."""
    s = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce")
    valid = s.dropna()
    if len(valid) < 5:
        return {"applied": False, "reason": "not enough numeric values"}
    low = float(valid.quantile(lower_q))
    high = float(valid.quantile(upper_q))
    clipped = s.clip(lower=low, upper=high)
    changed = int(((s.notna()) & (clipped != s)).sum())
    return {
        "applied": True,
        "lower_q": lower_q,
        "upper_q": upper_q,
        "lower_value": round(low, 6),
        "upper_value": round(high, 6),
        "changed_count": changed,
    }


def impute_median(values: List[Any]) -> Dict[str, Any]:
    s = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce")
    med = float(s.dropna().median()) if s.dropna().size > 0 else 0.0
    return {"method": "median", "fill_value": round(med, 6)}


def impute_mode(values: List[Any]) -> Dict[str, Any]:
    s = pd.Series(values, dtype="object")
    mode = s.dropna().astype(str).mode()
    fill = mode.iloc[0] if len(mode) > 0 else ""
    return {"method": "mode", "fill_value": str(fill)}


def bayesian_ridge_impute_preview(target_values: List[Any], feature_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Preview whether BayesianRidge imputation is feasible."""
    try:
        from sklearn.linear_model import BayesianRidge
        from sklearn.impute import SimpleImputer
    except ImportError:
        return {"feasible": False, "reason": "scikit-learn not installed"}
    if not feature_rows:
        return {"feasible": False, "reason": "no feature rows"}

    X = pd.DataFrame(feature_rows)
    y = pd.to_numeric(pd.Series(target_values, dtype="object"), errors="coerce")
    if len(y) != len(X):
        return {"feasible": False, "reason": "length mismatch"}
    train_mask = y.notna()
    if int(train_mask.sum()) < 20:
        return {"feasible": False, "reason": "too few observed targets"}
    X_num = X.apply(pd.to_numeric, errors="coerce")
    X_imp = SimpleImputer(strategy="median").fit_transform(X_num)
    model = BayesianRidge()
    model.fit(X_imp[train_mask.values], y[train_mask].values)
    pred_mask = ~train_mask
    n_pred = int(pred_mask.sum())
    return {"feasible": n_pred > 0, "train_rows": int(train_mask.sum()), "predict_rows": n_pred}


def random_forest_impute_preview(target_values: List[Any], feature_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Preview whether RandomForestRegressor imputation is feasible."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
    except ImportError:
        return {"feasible": False, "reason": "scikit-learn not installed"}
    if not feature_rows:
        return {"feasible": False, "reason": "no feature rows"}

    X = pd.DataFrame(feature_rows)
    y = pd.to_numeric(pd.Series(target_values, dtype="object"), errors="coerce")
    if len(y) != len(X):
        return {"feasible": False, "reason": "length mismatch"}
    train_mask = y.notna()
    if int(train_mask.sum()) < 30:
        return {"feasible": False, "reason": "too few observed targets"}
    X_num = X.apply(pd.to_numeric, errors="coerce")
    X_imp = SimpleImputer(strategy="median").fit_transform(X_num)
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    model.fit(X_imp[train_mask.values], y[train_mask].values)
    pred_mask = ~train_mask
    n_pred = int(pred_mask.sum())
    return {"feasible": n_pred > 0, "train_rows": int(train_mask.sum()), "predict_rows": n_pred}


def apply_quality_actions(
    df: pd.DataFrame,
    actions: List[Dict[str, Any]],
    protected_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply LLM-selected quality actions to a table deterministically."""
    protected = set(protected_columns or [])
    out = df.copy()
    logs: List[Dict[str, Any]] = []
    dropped: List[str] = []

    def _feature_df(target_col: str) -> pd.DataFrame:
        feat_cols = [c for c in out.columns if c != target_col and c not in protected]
        if not feat_cols:
            return pd.DataFrame(index=out.index)
        return out[feat_cols].apply(pd.to_numeric, errors="coerce")

    for act in actions or []:
        col = act.get("column")
        if not col or col not in out.columns or col in protected:
            continue
        action = str(act.get("action", "keep")).lower()
        method = str(act.get("method", "none")).lower()
        params = act.get("params") or {}

        entry: Dict[str, Any] = {"column": col, "action": action, "method": method, "status": "skipped"}

        if action == "drop":
            out = out.drop(columns=[col], errors="ignore")
            dropped.append(col)
            entry["status"] = "dropped"
            logs.append(entry)
            continue

        if method == "winsorize":
            s = pd.to_numeric(out[col], errors="coerce")
            valid = s.dropna()
            if len(valid) >= 5:
                lq = float(params.get("lower_q", 0.05))
                uq = float(params.get("upper_q", 0.95))
                low = float(valid.quantile(lq))
                high = float(valid.quantile(uq))
                out[col] = s.clip(lower=low, upper=high)
                entry["status"] = "applied"
                entry["lower_value"] = round(low, 6)
                entry["upper_value"] = round(high, 6)
            logs.append(entry)
            continue

        if method == "median":
            s = pd.to_numeric(out[col], errors="coerce")
            med = float(s.dropna().median()) if s.dropna().size > 0 else 0.0
            out[col] = s.fillna(med)
            entry["status"] = "applied"
            entry["fill_value"] = round(med, 6)
            logs.append(entry)
            continue

        if method == "mode":
            s = out[col]
            mode = s.dropna().mode()
            fill = mode.iloc[0] if len(mode) > 0 else ""
            out[col] = s.fillna(fill)
            entry["status"] = "applied"
            entry["fill_value"] = str(fill)
            logs.append(entry)
            continue

        if method in ("bayesianridge", "bayesian_ridge", "rf", "random_forest"):
            try:
                from sklearn.impute import SimpleImputer
                if method in ("rf", "random_forest"):
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=1)
                else:
                    from sklearn.linear_model import BayesianRidge
                    model = BayesianRidge()

                y = pd.to_numeric(out[col], errors="coerce")
                X = _feature_df(col)
                if X.shape[1] == 0:
                    raise ValueError("no numeric features for model imputation")
                X_imp = SimpleImputer(strategy="median").fit_transform(X)
                train_mask = y.notna()
                pred_mask = ~train_mask
                if int(train_mask.sum()) < 20 or int(pred_mask.sum()) == 0:
                    raise ValueError("insufficient rows for model imputation")
                model.fit(X_imp[train_mask.values], y[train_mask].values)
                pred = model.predict(X_imp[pred_mask.values])
                y2 = y.copy()
                y2[pred_mask] = pred
                out[col] = y2
                entry["status"] = "applied"
                entry["predicted_count"] = int(pred_mask.sum())
            except Exception as e:
                # Fallback to median
                s = pd.to_numeric(out[col], errors="coerce")
                med = float(s.dropna().median()) if s.dropna().size > 0 else 0.0
                out[col] = s.fillna(med)
                entry["status"] = "fallback_median"
                entry["fallback_reason"] = str(e)
                entry["fill_value"] = round(med, 6)
            logs.append(entry)
            continue

        entry["status"] = "kept_no_change"
        logs.append(entry)

    report = {
        "applied_actions": logs,
        "dropped_columns": dropped,
        "columns_before": int(len(df.columns)),
        "columns_after": int(len(out.columns)),
    }
    return out, report
