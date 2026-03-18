#!/usr/bin/env python3
"""
Simple LLM-based Table Analysis
Reads metadata and sample rows, asks LLM to select target column and decompose table
"""

from openai import OpenAI
import google.generativeai as genai
import json
import os
import csv
import sys
import time
from pathlib import Path

csv.field_size_limit(min(1000000, sys.maxsize))

class OpenAIClient:
    """Client for OpenAI API"""
    
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.provider = "openai"
    
    def ask(self, prompt):
        """Send prompt to LLM and get response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None


class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self, api_key=None, model="gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Please set GOOGLE_API_KEY environment variable")
        
        # Configure the client with API key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.provider = "gemini"
    
    def ask(self, prompt):
        """Send prompt to LLM and get response"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None


def create_llm_client(provider="openai", model=None):
    """
    Factory function to create LLM client
    
    Args:
        provider: "openai" or "gemini"
        model: Model name (optional, uses defaults if not specified)
    
    Returns:
        LLM client instance
    """
    if provider.lower() == "openai":
        default_model = model or "gpt-4o"
        return OpenAIClient(model=default_model)
    elif provider.lower() == "gemini":
        default_model = model or "gemini-2.5-flash"
        return GeminiClient(model=default_model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose 'openai' or 'gemini'")


def load_dataset(dataset_path, num_rows=30):
    """
    Load metadata and sample rows from a dataset
    
    Returns: (metadata_dict, sample_rows_list)
    """
    dataset_path = Path(dataset_path)
    
    # Read metadata.json
    with open(dataset_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Extract key info
    resource = metadata['resource']
    table_info = {
        "table_name": resource['name'],
        "description": resource.get('description', ''),
        "columns": {}
    }
    
    # Build column info from parallel arrays
    # Note: CSV files use columns_name (display names), not columns_field_name (API names)
    column_names = resource.get('columns_name', [])
    field_names = resource.get('columns_field_name', [])
    datatypes = resource['columns_datatype']
    descriptions = resource.get('columns_description', [])

    # extract the actual column names from the CSV file
    csv_columns = []
    with open(dataset_path / "rows.csv", 'r') as f:
        reader = csv.reader(f)
        csv_columns = next(reader)  # get the first row as the column names

    # compare the column names in the metadata and the CSV file
    metadata_columns = set(column_names)
    csv_columns_set = set(csv_columns)

    missing_in_csv = metadata_columns - csv_columns_set
    missing_in_metadata = csv_columns_set - metadata_columns

    table_info['column_comparison'] = {
        'metadata_columns': list(metadata_columns),
        'csv_columns': csv_columns,
        'missing_in_csv': list(missing_in_csv),
        'missing_in_metadata': list(missing_in_metadata),
        'columns_match': len(missing_in_csv) == 0 and len(missing_in_metadata) == 0
    }

    actual_columns = csv_columns

    for col_name in actual_columns:
        if col_name in column_names:
            idx = column_names.index(col_name)
            table_info['columns'][col_name] = {
                "field_name": field_names[idx] if idx < len(field_names) else col_name,
                "type": datatypes[idx] if idx < len(datatypes) else 'unknown',
                "description": descriptions[idx] if idx < len(descriptions) else ''
            }
        else:
            # the column exists in the CSV file but not in the metadata
            table_info['columns'][col_name] = {
                "field_name": col_name,
                "type": 'unknown',
                "description": ''
            }


    # Use columns_name as the primary key (matches CSV headers)
    # for i, col_name in enumerate(column_names):
    #     table_info['columns'][col_name] = {
    #         "field_name": field_names[i] if i < len(field_names) else col_name,  # API field name
    #         "type": datatypes[i] if i < len(datatypes) else 'unknown',
    #         "description": descriptions[i] if i < len(descriptions) else ''
    #     }
    
    # Read sample rows from CSV
    rows = []
    with open(dataset_path / "rows.csv", 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_rows:
                break
            rows.append(row)
    
    uniqueness_ratios = {}
    for col_name in table_info['columns'].keys():
        if rows and col_name in rows[0]:  # ensure the column exists in the sample data
            values = [row[col_name] for row in rows if col_name in row]
            unique_values = len(set(values))
            total_values = len(values)
            uniqueness_ratio = (unique_values / total_values) * 100 if total_values > 0 else 0
            uniqueness_ratios[col_name] = {
                'unique_count': unique_values,
                'total_count': total_values,
                'uniqueness_ratio': round(uniqueness_ratio, 2)
            }
    
    # add the uniqueness ratios to the table_info
    table_info['uniqueness_ratios'] = uniqueness_ratios
    
    # detect date columns
    def detect_date_columns(table_info, rows):
        """detect date/time columns"""
        date_columns = []
        date_keywords = ['date', 'time', 'datetime', 'created', 'updated', 'published', 'issue', 'start', 'end', 'pickup', 'dropoff']
        
        for col_name in table_info['columns'].keys():
            col_lower = col_name.lower()
            # check if the column name contains date related keywords
            if any(keyword in col_lower for keyword in date_keywords):
                date_columns.append(col_name)
            # check if the column data type is date, datetime or timestamp
            elif table_info['columns'][col_name].get('type', '').lower() in ['date', 'datetime', 'timestamp']:
                date_columns.append(col_name)
            # check if the column has sample data and the sample data is a date
            elif rows and col_name in rows[0]:
                sample_values = [row[col_name] for row in rows[:5] if col_name in row and row[col_name]]
                if sample_values:
                    # check if the sample data is a date
                    date_patterns = ['/', '-', 'T', 'Z']
                    if any(pattern in str(sample_values[0]) for pattern in date_patterns):
                        date_columns.append(col_name)

    date_columns = detect_date_columns(table_info, rows)
    table_info['date_columns'] = date_columns

    return table_info, rows

def create_prompt(table_info, sample_rows):
    """Create prompt for LLM"""
    # date information
    date_columns_info = ""
    if 'date_columns' in table_info and table_info['date_columns']:
        date_columns_info = f"""
    ## CRITICAL: Date/Time Columns Detected
    **The following columns have been identified as date/time columns: {table_info['date_columns']}**

    **MANDATORY REQUIREMENT: You MUST include at least one of these date columns in your join key selection, unless there is a compelling technical reason not to.**

    **This is a hard requirement - failure to include date columns when they are available will result in incorrect analysis.**
    """

    # add the uniqueness information to the prompt
    uniqueness_info = ""
    if 'uniqueness_ratios' in table_info:
        uniqueness_info = "\n## Column Uniqueness Analysis\n"
        uniqueness_info += "**Pre-calculated uniqueness ratios for each column:**\n"
        for col_name, stats in table_info['uniqueness_ratios'].items():
            uniqueness_info += f"- **{col_name}**: {stats['unique_count']}/{stats['total_count']} = {stats['uniqueness_ratio']}%\n"
        uniqueness_info += "\n**Only select columns with uniqueness ≥ 90% as join columns.**\n"

    # if the column names in the metadata and the CSV file are not the same, print a warning
    mismatch_warning = ""
    if 'column_comparison' in table_info and not table_info['column_comparison']['columns_match']:
        missing_in_csv = table_info['column_comparison']['missing_in_csv']
        csv_columns = table_info['column_comparison']['csv_columns']
        
        mismatch_warning = f"""
**IMPORTANT: The following columns from metadata are missing in the actual CSV file: {missing_in_csv}**
**Use only the column names that exist in the CSV file: {csv_columns}**
**Do not use column names that appear in the "Columns" section but are missing from the "Sample rows".**
"""

    return f"""
# Table Structure Understanding & ML Task Design Prompt

You are a data analysis assistant specialized in understanding table structures and designing ML prediction tasks.

Your task is to:

1. Select one target column for ML prediction  
   - If categorical → classification  
   - If continuous numeric → regression  

2. Select join columns and decompose the table into two subtables that can be losslessly rejoined via the join column(s).  

3. Always return a valid JSON following one of the formats below — no additional text.

---

## Step 0 — Column Name Verification 

### Goal
Verify that the column names in "Columns" section match the keys in "Sample rows" section.

### Rules
1. Extract column names from "Columns" section
2. Extract column names from "Sample rows" keys  
3. Compare them for consistency
4. If there are discrepancies, note them and use "Sample rows" keys as the authoritative source
**IMPORTANT: The keys in "Sample rows" are the actual column names from the CSV file. Use these exact column names in your output.**

## Step 1 — Target Column Selection

### Goal
Identify one column suitable as the prediction target.

### Rules
- Examine **sample_rows** to judge column suitability (do not rely only on column names).  
- Choose:
  - A categorical column with a few distinct values → classification  
  - A numeric column with continuous values → regression  
- Avoid:
  - ID-like columns (`id`, `uuid`, `code`, `identifier`, etc.)  
  - Constant, mostly-null, or descriptive text fields (`comments`, `notes`, etc.)  
- Prefer columns that can be explained by other features (dependent variables).  

### If no suitable target column exists
Return a JSON with `"status": "no_suitable_target_column"` and explain why (e.g., all columns are IDs or constants).

---

## Step 2 — Join Column(s) Selection

### Goal
Find **1 or 2 columns** (maximum 2 columns) that can serve as the join key(s) for table decomposition.

### Rules
1. **Try single column first**:
   - Use the pre-calculated uniqueness ratios provided below
   - A valid single join column must satisfy:
     a) Not the target column  
     b) Uniqueness ≥ 90%  
     c) Looks like a key (identifier-like, not aggregate or free text)  
   - Priority order:
     - Primary key–like columns: `id`, `objectid`, `globalid`, `uuid`, `guid`, `code`, `identifier`  
     - Foreign key–like columns: names ending with `_id` (e.g., `user_id`, `order_id`) — only if no primary key is available
    - **CRITICAL: If the dataset contains date/time columns, strongly consider using them as composite keys even if they don't meet the 90% uniqueness threshold, as they are often essential for temporal data integrity**
{uniqueness_info}

2. **If no single column qualifies, try composite key (MAXIMUM 2 COLUMNS)**:
   - **CRITICAL LIMIT: You can select AT MOST 2 columns for the join key**
   - Select 1-2 columns (maximum 2) that together form a unique identifier
   - Example: For aggregate reports, use `(store_id, date)` or `(id, date)` (2 columns maximum)
   - Verify: The combination of these columns should uniquely identify each row
   - Must NOT include the target column
   - **If you need more than 2 columns to form a unique key, return `"status": "no_suitable_join_column"`**

3. **Special consideration for date/time columns (MAXIMUM 2 COLUMNS)**:
- **CRITICAL LIMIT: When using date/time columns, you can select AT MOST 2 columns total (e.g., one date column + one id column, or two date columns)**
- **If using date/time columns as join columns, check if they need to be combined with other columns (but total must not exceed 2 columns)**
- **For temporal data (like pickup_start_date, report_date), consider composite keys like (date, id) - maximum 2 columns**
- **For time range data with start and end dates (e.g., pickup_date and dropoff_date), you can use BOTH dates as a composite key (2 columns maximum) to preserve temporal integrity**
- **If the dataset contains date range pairs (e.g., pickup_date and dropoff_date), you can use both dates as a composite key (2 columns maximum)**
- **Date columns often have low uniqueness in aggregated datasets and require composite keys, but remember the 2-column maximum limit**
- **IMPORTANT RULE: When a dataset contains date/time columns, strongly consider including them in composite join keys, but ensure the total number of join columns does not exceed 2**
- **VERY IMPORTANT RULE: Read {date_columns_info}, If a dataset contains date-related columns (Date, Time, Published Date, Issue Date, etc.), ALWAYS include at least one date column in the join key unless there is a compelling reason not to, but remember the maximum is 2 columns total**

4. **CRITICAL: Lossless Join Verification**
   - **MANDATORY CHECK**: Before finalizing your join column selection, you MUST verify that the selected join column(s) can perform a **lossless join**
   - **REMINDER: Maximum 2 columns allowed for join key**
   - A lossless join means:
     a) The join column(s) must have **NO duplicate values** in either subtable (each value appears exactly once)
     b) When joining the two subtables on the join column(s), the resulting table must have **exactly the same number of rows** as the original table
     c) All columns from the original table must be preserved in the joined result
   - **Verification steps**:
     1. Check the sample rows provided: count how many times each join column value appears
     2. If any join column value appears more than once in the sample, that column is NOT suitable for lossless join
     3. For composite keys (maximum 2 columns), verify that the combination of values is unique across all rows
     4. If you find duplicates, you MUST either:
        - Select different join column(s) that are truly unique (remember: maximum 2 columns)
        - Use a composite key (1-2 columns) that together forms a unique identifier
        - Return `"status": "no_suitable_join_column"` if no lossless join is possible with 1-2 columns
   - **Example**: If a column has values [1, 2, 2, 3] in the sample, it has duplicates and cannot guarantee a lossless join
   - **Example**: If composite key (id, date) has combinations [(1, '2024-01-01'), (1, '2024-01-02'), (2, '2024-01-01')], this is unique and suitable for lossless join (2 columns maximum)

### If no suitable join column(s) exist
Return `"status": "no_suitable_join_column"` with a brief explanation.

---

## Step 3 — Table Decomposition

### Requirements
- Split the table into two subtables
- All subtables must include **ALL** chosen join column(s)
- **All** columns in the original table must be included in the subtables
- Even if some columns are empty, they should be included in the subtables
- After decomposition, count the number of distinct columns in the subtables and the original table, and the number of distinct columns in the subtables must be same as the number of distinct columns in the original table
- Each subtable should contain a balanced, meaningful subset of columns
- **The original table must be reconstructable by joining the subtables on the join column(s) using an INNER JOIN, and the result must have exactly the same number of rows as the original table (lossless join)**


## Step 3A — Candidate Augmentation Feature Identification

### Goal
Create a minimal baseline table (non_candidate_table) with only essential features, and an augmentation table (candidate_table) with all other potentially useful features.

### Critical Strategy: Minimal Non-Candidate Table

The **non_candidate_table** MUST be kept as small as possible:

1. **Required components** (always include):
   - Target column (required)
   - All join column(s) (required for joining)
   - Core features (1-2 columns MAXIMUM):
        Only include if absolutely necessary for a reasonable baseline
        Choose the 1-2 features that are MOST directly and strongly related to the target
        These should be features that are essential context for the target, not optional augmentation
        Examples:
            *If target is "income", you might include "education_level" (highly correlated)
            *If target is "disease_status", you might include "age" (strong risk factor)
            *If target is "price", you might include a base price or category

2. **What NOT to include in non_candidate_table**:
   - Features that could provide augmentation value (put them in candidate_table)
   - Multiple redundant features (choose only the most essential)
   - Features that are better suited as augmentation candidates
   - More than 2 core features (strict limit)

### Candidate Table:
**CRITICAL**: The candidate_table MUST include:
  - **ALL join column(s)** (REQUIRED for joining - this is mandatory!)
  - ALL other columns EXCEPT:
    * Target column (never include)
    * The 1-2 core features placed in non_candidate_table (if any)

**Summary**: candidate_table = join_columns + all_other_features (excluding target and non_candidate core features)

- Even if a feature seems less useful, include it in candidate_table rather than non_candidate_table
- **REMINDER**: Join columns are NOT optional - they MUST be in candidate_table!

### Reasoning Required:
For the non_candidate_table, you MUST provide explicit reasoning:
- List the 1-2 core features (if any) beyond target+join
- Explain WHY each core feature is essential and cannot be left to candidate_table
- Justify why these specific features are chosen over alternatives
---

## Output Format (JSON Only)
**CRITICAL: Use column_name (display names) from the "Columns" section, NOT field_name (API names).**
**Output details only when the target column and join column(s) are found successfully, otherwise only output dataset_id.**

### Success Case:

{{
  "dataset": "dataset_id",
  "status": "success",
  "target_column": {{
    "name": "column_name",
    "task_type": "classification or regression",
    "reasoning": "brief reason"
  }},
  "candidate_table": {{
    "name": "Candidate_Features",
    "professional_term": "feature table",
    "columns": ["colA", "colB", "..."],
    "reasoning": ["why colA helps", "why colB helps"]
  }},
  "non_candidate_table": {{
    "name": "NonCandidate_With_Target",
    "professional_term": "label table",
    "columns": ["target_column_name", "colX", "colY", "..."]
  }},

  "join_columns": ["column_name"],
  "join_column_analysis": {{
    "candidate_columns": [
      {{
        "column": "column_name",
        "uniqueness_ratio": "use the pre-calculated ratio from the Column Uniqueness Analysis above",
        "reasoning": "High uniqueness, suitable as join key"
      }}
    ],
    "selected_columns": ["column_name"],
    "selection_reasoning": "Selected based on uniqueness ≥ 90% and key-like properties",
    "lossless_join_verification": {{
      "verified": true,
      "verification_method": "Checked sample rows - no duplicate values found in join column(s)",
      "row_count_check": "Original table has X rows, join will preserve all X rows"
    }}
  }}
}}


### Failed Case:

{{
  "dataset": "dataset_id",
  "status": "failed",
  "error": "brief error description"
}}


---

## Examples

**Example 1 **
Table: [id, user_name, age, city, income]
→ target_column: ["income"]
→ join_columns: ["id"]

**Example 2 **
Table: [Student_ID, Course_ID, Semester, Grade, Credits, Instructor_Name]
→ target_column: "Grade"
→ join_columns: ["Student_ID", "Course_ID"]
(Composite key needed: same student can take multiple courses, same course has multiple students. Note: Maximum 2 columns, so we use Student_ID + Course_ID instead of including Semester)

---

Now analyze this table:

{mismatch_warning}

Table: {table_info['table_name']}
Description: {table_info['description']}

Columns:
{json.dumps(table_info['columns'], indent=2)}

Sample rows (first 30):
{json.dumps(sample_rows[:30], indent=2)}

"""



def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from model output."""
    if not text or not isinstance(text, str):
        return {}
    text = text.strip()
    if not text:
        return {}

    json_str = None
    if "```json" in text:
        parts = text.split("```json", 1)[1].split("```", 1)
        json_str = parts[0].strip() if parts else None
    elif "```" in text:
        parts = text.split("```", 1)[1].split("```", 1)
        json_str = parts[0].strip() if parts else None

    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                import ast
                return ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                pass

    return {"relevant_tables": []}


def analyze_dataset(dataset_path, client):
    """Analyze one dataset"""
    dataset_id = Path(dataset_path).name
    print(f"\n=== Analyzing {dataset_id} ===")
    
    # Load data
    table_info, sample_rows = load_dataset(dataset_path, num_rows=20)
    print("Table: {}".format(table_info['table_name']))
    print("Columns: {}, Rows: {}".format(len(table_info['columns']), len(sample_rows)))
    
    # Create prompt and call LLM
    prompt = create_prompt(table_info, sample_rows)
    response = client.ask(prompt)
    
    # Check if response is None (API error)
    if response is None:
        print(f"✗ Error: LLM API returned None (likely quota exceeded or API error)")
        return {
            "dataset": dataset_id,
            "status": "failed",
            "error": "LLM API returned None (quota exceeded or API error)"
        }
    
    # Parse JSON
    try:
        json_str = extract_json(response)
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"✗ Error: Failed to parse JSON from LLM response: {e}")
        print(f"  Response preview: {response[:200] if response else 'None'}...")
        return {
            "dataset": dataset_id,
            "status": "failed",
            "error": f"JSON parsing error: {str(e)}"
        }
    
    # Print result based on status
    status = result.get('status', 'unknown')
    
    if status == 'no_suitable_target_column':
        print(f"⚠ No suitable target column found")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
    elif status == 'no_suitable_join_column':
        print(f"✓ Target: {result['target_column']['name']} ({result['target_column']['task_type']})")
        print(f"⚠ No suitable join column found")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
    elif status == 'success':
        print(f"✓ Target: {result['target_column']['name']} ({result['target_column']['task_type']})")
        print(f"✓ Join columns: {result.get('join_columns', [])}")
        # Prefer new candidate/non-candidate outputs; fall back to legacy subtables if present
        if 'candidate_table' in result:
            ct = result['candidate_table']
            nct = result.get('non_candidate_table', {'name': 'NonCandidate', 'columns': []})
            print(f"✓ Candidate table: {ct.get('name', 'Candidate_Features')} - {len(ct.get('columns', []))} cols")
            print(f"✓ Non-candidate table: {nct.get('name', 'NonCandidate_With_Target')} - {len(nct.get('columns', []))} cols")
        elif 'subtable_1' in result and 'subtable_2' in result:
            print(f"✓ Subtable 1: {result['subtable_1']['name']} - {len(result['subtable_1']['columns'])} cols")
            print(f"✓ Subtable 2: {result['subtable_2']['name']} - {len(result['subtable_2']['columns'])} cols")
            if 'subtable_3' in result:
                print(f"✓ Subtable 3: {result['subtable_3']['name']} - {len(result['subtable_3']['columns'])} cols")
    else:
        print(f"⚠ Unknown status: {status}")
    
    return result


def find_datasets(base_dir):
    """
    Find all dataset directories under base_dir
    (directories that contain both metadata.json and rows.csv)
    """
    base_path = Path(base_dir)
    datasets = []
    
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if it has both required files
            if (item / "metadata.json").exists() and (item / "rows.csv").exists():
                datasets.append(item)
    
    return sorted(datasets)


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_llm_prompt.py <datasets_directory> [--max N] [--provider openai|gemini] [--model MODEL_NAME]")
        print("Example: python test_llm_prompt.py datasets/")
        print("Example: python test_llm_prompt.py datasets/ --max 5")
        print("Example: python test_llm_prompt.py datasets/ --provider openai --model gpt-4o")
        print("Example: python test_llm_prompt.py datasets/ --provider gemini --model gemini-2.0-flash-exp")
        sys.exit(1)
    
    # Parse arguments
    datasets_dir = sys.argv[1]
    max_datasets = None
    provider = "openai"  # Default to OpenAI
    model = None  # Use default model for the provider
    
    if "--max" in sys.argv:
        max_idx = sys.argv.index("--max")
        if max_idx + 1 < len(sys.argv):
            max_datasets = int(sys.argv[max_idx + 1])
    
    if "--provider" in sys.argv:
        provider_idx = sys.argv.index("--provider")
        if provider_idx + 1 < len(sys.argv):
            provider = sys.argv[provider_idx + 1]
    
    if "--model" in sys.argv:
        model_idx = sys.argv.index("--model")
        if model_idx + 1 < len(sys.argv):
            model = sys.argv[model_idx + 1]
    
    # Find all datasets
    print(f"Scanning {datasets_dir} for datasets...")
    dataset_paths = find_datasets(datasets_dir)
    
    if not dataset_paths:
        print(f"No datasets found in {datasets_dir}")
        sys.exit(1)
    
    if max_datasets:
        dataset_paths = dataset_paths[:max_datasets]
    
    print(f"Found {len(dataset_paths)} datasets to process")
    print(f"Using LLM Provider: {provider.upper()}")
    if model:
        print(f"Using Model: {model}")
    print()
    
    # Create LLM client
    client = create_llm_client(provider=provider, model=model)
    results = []
    
    for i, dataset_path in enumerate(dataset_paths, 1):
        print(f"[{i}/{len(dataset_paths)}]", end=" ")
        try:
            result = analyze_dataset(dataset_path, client)
            results.append({
                "dataset": dataset_path.name,
                "status": "success",
                "result": result
            })
            # Add delay to avoid rate limits
            time.sleep(2)
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "dataset": dataset_path.name,
                "status": "failed",
                "error": str(e)
            })
    
    # Save results
    with open("analysis_results_optimized.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"Processed {len(results)} datasets ({success} success, {len(results)-success} failed)")
    print(f"Results saved to: analysis_results_optimized.json")


if __name__ == "__main__":
    main()
