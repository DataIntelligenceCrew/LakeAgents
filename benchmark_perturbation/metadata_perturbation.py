import re
import json
from pathlib import Path
from typing import Union, List, Optional
import random
import shutil 
import csv

def tokenize(text: str) -> set:
    """
    Convert text to token set.
    1. lowercase
    2. camel case split (IncidentDate → incident date)
    3. split by non-alphanumeric characters
    4. remove tokens with length 1
    """
    if not text or not isinstance(text, str):
        return set()
    
    # 1. lowercase
    s = text.lower()
    
    # 2. camel case split: insert space at case boundary (helloWorld -> hello world)
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    
    # 3. split by non-alphanumeric characters
    tokens = re.split(r'[^a-z0-9]+', s)
    
    # 4. remove empty tokens and tokens with length 1
    return {t for t in tokens if len(t) > 1}

def _col_to_desc(columns_name: list, columns_description: list, col_name: str) -> str:
    """Get description from columns_name/columns_description by column name."""
    cols = columns_name or []
    descs = columns_description or []
    while len(descs) < len(cols):
        descs.append("")
    try:
        idx = cols.index(col_name)
        return descs[idx] if idx < len(descs) else ""
    except (ValueError, TypeError):
        return ""


def extract_query_table_jaccard_fields(
    metadata_path: Union[str, Path],
    join_column: Union[str, List[str]],
    target_column: str,
) -> dict:
    """
    Extract 6 fields for Jaccard calculation from query table's metadata.json.

    Returns:
        {
            "query_table_name": str,
            "query_table_description": str,
            "target_column": str,
            "target_column_description": str,
            "join_column": list[str],           # unified to list
            "join_column_description": list[str],
        }
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    res = meta.get("resource") or {}
    columns_name = res.get("columns_name") or []
    columns_description = res.get("columns_description") or []

    query_table_name = res.get("name", "")
    query_table_description = res.get("description", "")

    join_cols = [join_column] if isinstance(join_column, str) else list(join_column)
    join_descs = [_col_to_desc(columns_name, columns_description, c) for c in join_cols]

    return {
        "query_table_name": query_table_name,
        "query_table_description": query_table_description,
        "target_column": target_column,
        "target_column_description": _col_to_desc(columns_name, columns_description, target_column),
        "join_column": join_cols,
        "join_column_description": join_descs,
    }

def save_jaccard_fields(fields: dict, output_path: Union[str, Path]) -> None:
    """Save extracted fields to JSON for subsequent Jaccard calculation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, ensure_ascii=False)


def load_jaccard_fields(path: Union[str, Path]) -> dict:
    """Load saved Jaccard fields."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity = |A ∩ B| / |A ∪ B|. Returns 0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _collect_tokens_from_field(field_val) -> set:
    """Collect tokens from a field: either {raw, tokens} or list of {raw, tokens}."""
    if isinstance(field_val, dict) and "tokens" in field_val:
        return set(field_val["tokens"])
    if isinstance(field_val, list):
        out = set()
        for item in field_val:
            if isinstance(item, dict) and "tokens" in item:
                out.update(item["tokens"])
        return out
    return set()


def collect_all_tokens(tokenized_data: dict) -> set:
    """Collect union of all tokens from a tokenized JSON (all 6 field types)."""
    all_tokens = set()
    for key in (
        "query_table_name",
        "query_table_description",
        "target_column",
        "target_column_description",
        "join_column",
        "join_column_description",
    ):
        if key in tokenized_data:
            all_tokens.update(_collect_tokens_from_field(tokenized_data[key]))
    return all_tokens



def perturb_tokens_greedy(
    original_tokens: set,
    synonym_dict: dict,
    threshold: float = 0.85,
    random_state: int = None,
) -> tuple:
    """
    Greedily replace tokens (in random order) until Jaccard <= threshold.

    random_state: If set, makes the process reproducible.
    """
    if random_state is not None:
        random.seed(random_state)

    perturbed = set(original_tokens)
    replacements = {}

    while jaccard_similarity(original_tokens, perturbed) > threshold:
        replaceable = [t for t in perturbed if t in synonym_dict and t not in replacements]
        if not replaceable:
            break

        # Randomly pick one token to replace
        t = random.choice(replaceable)
        synonyms = synonym_dict[t]

        # Prefer synonyms NOT in original (better Jaccard drop); shuffle for randomness
        candidates_not_in_orig = [s for s in synonyms if s != t and s not in original_tokens]
        candidates_any = [s for s in synonyms if s != t]
        pool = candidates_not_in_orig if candidates_not_in_orig else candidates_any
        if not pool:
            pool = synonyms  # fallback

        s = random.choice(pool)

        perturbed.discard(t)
        perturbed.add(s)
        replacements[t] = s

    return perturbed, replacements

def process_tokenized_file(
    json_path: Union[str, Path],
    synonym_path: Union[str, Path],
    threshold: float = 0.85,
    output_path: Union[str, Path] = None,
) -> dict:
    """
    For one tokenized JSON file: collect tokens, greedy perturb, compute Jaccard.
    """
    json_path = Path(json_path)
    synonym_path = Path(synonym_path)

    with open(json_path, "r", encoding="utf-8") as f:
        tokenized_data = json.load(f)
    with open(synonym_path, "r", encoding="utf-8") as f:
        synonym_dict = json.load(f)

    original = collect_all_tokens(tokenized_data)
    perturbed, replacements = perturb_tokens_greedy(
        original, synonym_dict, threshold=threshold
    )
    jaccard_after = jaccard_similarity(original, perturbed)

    result = {
        "file": str(json_path.name),
        "original_tokens": sorted(original),
        "perturbed_tokens": sorted(perturbed),
        "replacements": replacements,
        "jaccard_before": 1.0,
        "jaccard_after": round(jaccard_after, 4),
        "threshold": threshold,
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def run_all_tokenized_files(
    jaccard_tokenized_dir: Union[str, Path],
    synonym_path: Union[str, Path],
    threshold: float = 0.85,
    output_dir: Union[str, Path] = None,
) -> list:
    """Process each *_tokenized.json; compute Jaccard per file."""
    jaccard_tokenized_dir = Path(jaccard_tokenized_dir)
    synonym_path = Path(synonym_path)
    if output_dir is None:
        output_dir = jaccard_tokenized_dir.parent / "jaccard_perturbed"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(jaccard_tokenized_dir.glob("*_tokenized.json"))
    results = []
    for jf in json_files:
        out = output_dir / jf.name.replace("_tokenized.json", "_perturbed.json")
        res = process_tokenized_file(jf, synonym_path, threshold=threshold, output_path=out)
        results.append(res)
        print(
            f"{res['file']}: Jaccard {res['jaccard_before']} -> {res['jaccard_after']} "
            f"(threshold={threshold}), {len(res['replacements'])} replacements"
        )
    return results


def _apply_replacements_to_text(text: str, replacements: dict) -> str:
    """Replace words in text (case-insensitive, word boundary)."""
    if not text or not isinstance(text, str) or not replacements:
        return text
    for old_word, new_word in replacements.items():
        pattern = r'\b' + re.escape(old_word) + r'\b'
        text = re.sub(pattern, new_word, text, flags=re.IGNORECASE)
    return text

def apply_replacements_to_rows_csv(
    rows_path: Union[str, Path],
    output_path: Union[str, Path],
    replacements: dict,
    data_filename: str = "rows.csv",
) -> None:
    """
    Copy rows.csv and apply the same replacements to the header (column names),
    so column names stay in sync with perturbed metadata columns_name.
    """
    if not replacements or not rows_path or not Path(rows_path).exists():
        if rows_path and Path(rows_path).exists():
            shutil.copy2(rows_path, output_path)
        return
    with open(rows_path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        rows = list(reader)
    new_header = [_apply_replacements_to_text(h, replacements) for h in header]
    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(new_header)
        writer.writerows(rows)

def apply_replacements_to_metadata(meta: dict, replacements: dict) -> dict:
    """
    Apply token replacements to metadata fields used for LLM (resource.name,
    resource.description, columns_name, columns_description). Other fields unchanged.
    """
    if not replacements:
        return meta
    meta = json.loads(json.dumps(meta))  # deep copy
    res = meta.get("resource") or {}
    if res.get("name"):
        res["name"] = _apply_replacements_to_text(res["name"], replacements)
    if res.get("description"):
        res["description"] = _apply_replacements_to_text(res["description"], replacements)
    if res.get("columns_name"):
        res["columns_name"] = [
            _apply_replacements_to_text(c, replacements) for c in res["columns_name"]
        ]
    if res.get("columns_description"):
        res["columns_description"] = [
            _apply_replacements_to_text(c, replacements) for c in res["columns_description"]
        ]
    meta["resource"] = res
    return meta


def map_perturbed_to_query_table(
    jaccard_perturbed_dir: Union[str, Path],
    query_table_dir: Union[str, Path],
    output_base: Union[str, Path],
    data_filename: str = "rows.csv",
    beta: Optional[float] = None,
) -> None:
    """
    For each *_perturbed.json: load replacements, apply to metadata, save to
    output_base/metadata.json and copy rows.csv. Output dir: {output_base}/{table}/.
    """
    jaccard_perturbed_dir = Path(jaccard_perturbed_dir)
    query_table_dir = Path(query_table_dir)
    output_base = Path(output_base)

    for p in sorted(jaccard_perturbed_dir.glob("*_perturbed.json")):
        with open(p, "r", encoding="utf-8") as f:
            perturbed = json.load(f)
        table_name = p.stem.replace("_perturbed", "")
        replacements = perturbed.get("replacements", {})
        threshold = perturbed.get("threshold", 0.85)
        if beta is not None:
            subdir = f"perturbed_{threshold}_{beta}"
        else:
            subdir = f"perturbed_{threshold}"
        out_dir = output_base / subdir / table_name
        out_dir.mkdir(parents=True, exist_ok=True)

        meta_path = query_table_dir / table_name / "metadata.json"
        rows_path = query_table_dir / table_name / data_filename
        if not meta_path.exists():
            print(f"Skip {table_name}: metadata.json not found")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        perturbed_meta = apply_replacements_to_metadata(meta, replacements)
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(perturbed_meta, f, indent=2, ensure_ascii=False)

        if rows_path.exists():
            apply_replacements_to_rows_csv(
                rows_path,
                out_dir / data_filename,
                replacements,
                data_filename=data_filename,
            )
        else:
            print(f"Warning: {table_name} rows not found, skip copy")

        print(f"Saved: {out_dir}/metadata.json, {data_filename}")
    return output_base / subdir 