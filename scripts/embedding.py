#!/usr/bin/env python3
"""
Batch embedding for multiple tables following PyTorch Frame's standard workflow.
Each table gets its own encoder instance (following RelBench pattern).
"""
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from torch import Tensor
from torch_frame import stype
from torch_frame.data import Dataset, DataLoader
from torch_frame.utils import infer_df_stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from src.models.layer1_embedding import TableEmbedding, build_roles_for_df

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

# Configuration
DATASETS_DIR = "/localdisk3/ytang49/opendata/datasets"  # Changed to datasets (9 folders)
ANALYSIS_JSON = "/localdisk3/ytang49/opendata/analysis_results_optimized.json"
OUTPUT_DIR = "/localdisk3/ytang49/opendata/processed_data/table_embeddings"
BATCH_SIZE = 512
MAX_ROWS_PER_TABLE = None  # None = use all rows
CHANNELS = 256
ROLE_DIM = 256

# Text embedding configuration
USE_TEXT_EMBEDDING = True  # Set to False to convert text columns to categorical
TEXT_EMBEDDER_TYPE = "sentence_transformer"  # Options: "sentence_transformer", "api_openai", "api_cohere", "transformers"
TEXT_EMBEDDER_MODEL = "all-distilroberta-v1"  # Model name (varies by type)
TEXT_EMBEDDER_BATCH_SIZE = 32

# API configuration (if using API)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", None)
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text embedder classes (PyTorch Frame standard patterns)
class SentenceTransformerEncoder:
    """Text embedder using sentence-transformers (local, free)
    
    Follows PyTorch Frame's PretrainedTextEncoder pattern from examples/mercari.py
    """
    def __init__(self, model_name: str = TEXT_EMBEDDER_MODEL, device: torch.device = None):
        try:
            from sentence_transformers import SentenceTransformer
            self.device = device or torch.device("cpu")
            self.model = SentenceTransformer(model_name, device=str(self.device))
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def __call__(self, sentences: list[str]) -> Tensor:
        # Inference on GPU (if available), then map back to CPU
        # This matches PyTorch Frame's PretrainedTextEncoder pattern
        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=False,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings.cpu()  # Map back to CPU (PyTorch Frame standard)


class OpenAIEncoder:
    """Text embedder using OpenAI API (requires API key, paid)"""
    dimension: int = 1536
    text_embedder_batch_size: int = 25
    
    def __init__(self, model: str = "text-embedding-ada-002", api_key: str = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")
        
        api_key = api_key or OPENAI_API_KEY
        if api_key is None:
            raise ValueError("OpenAI API key not specified. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def __call__(self, sentences: list[str]) -> Tensor:
        # OpenAI SDK v1.0+ returns EmbeddingResponse with .data attribute
        response = self.client.embeddings.create(
            input=sentences, model=self.model
        )
        items = response.data  # List of embedding objects
        assert len(items) == len(sentences)
        embeddings = [
            torch.FloatTensor(item.embedding).view(1, -1) for item in items
        ]
        return torch.cat(embeddings, dim=0)


class CohereEncoder:
    """Text embedder using Cohere API (requires API key, paid)"""
    dimension: int = 1024
    text_embedder_batch_size: int = 1000
    
    def __init__(self, model: str = "embed-english-v3.0", api_key: str = None):
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere not installed. Install with: pip install cohere")
        
        api_key = api_key or COHERE_API_KEY
        if api_key is None:
            raise ValueError("Cohere API key not specified. Set COHERE_API_KEY environment variable.")
        
        self.model = model
        self.co = cohere.Client(api_key)
    
    def __call__(self, sentences: list[str]) -> Tensor:
        from cohere import EmbedResponse
        response: EmbedResponse = self.co.embed(
            model=self.model,
            texts=sentences,
            input_type="classification"
        )
        assert len(response.embeddings) == len(sentences)
        return torch.tensor(response.embeddings)


class TransformersEncoder:
    """Text embedder using Hugging Face transformers (local, free, can use GPU)
    
    Note: For sentence-transformers models, prefer SentenceTransformerEncoder.
    This is for raw transformers models that need mean pooling.
    """
    # Default batch size for transformers (can be adjusted based on GPU memory)
    text_embedder_batch_size: int = 32
    
    def __init__(self, model_name: str = "distilbert-base-uncased", device: torch.device = None):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers")
        
        self.device = device or torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        with torch.no_grad():
            test_input = self.tokenizer("test", return_tensors="pt").to(self.device)
            test_output = self.model(**test_input)
            self.dimension = test_output.last_hidden_state.shape[-1]
    
    def __call__(self, sentences: list[str]) -> Tensor:
        # Tokenize
        inputs = self.tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling (standard approach for transformers)
            attention_mask = inputs["attention_mask"]
            embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
        
        return mean_pooled.cpu()  # Return on CPU (PyTorch Frame standard)

# Global text encoder (lazy initialization)
_text_encoder = None

def get_text_encoder():
    """Get or create the global text encoder based on configuration"""
    global _text_encoder
    if _text_encoder is None and USE_TEXT_EMBEDDING:
        print(f"Initializing text embedder: type={TEXT_EMBEDDER_TYPE}, model={TEXT_EMBEDDER_MODEL}...")
        
        if TEXT_EMBEDDER_TYPE == "sentence_transformer":
            _text_encoder = SentenceTransformerEncoder(TEXT_EMBEDDER_MODEL, device=device)
        elif TEXT_EMBEDDER_TYPE == "api_openai":
            _text_encoder = OpenAIEncoder(model=TEXT_EMBEDDER_MODEL, api_key=OPENAI_API_KEY)
        elif TEXT_EMBEDDER_TYPE == "api_cohere":
            _text_encoder = CohereEncoder(model=TEXT_EMBEDDER_MODEL, api_key=COHERE_API_KEY)
        elif TEXT_EMBEDDER_TYPE == "transformers":
            _text_encoder = TransformersEncoder(TEXT_EMBEDDER_MODEL, device=device)
        else:
            raise ValueError(f"Unknown TEXT_EMBEDDER_TYPE: {TEXT_EMBEDDER_TYPE}")
        
        print(f"  Text embedder dimension: {_text_encoder.dimension}")
        if hasattr(_text_encoder, 'text_embedder_batch_size'):
            print(f"  Recommended batch size: {_text_encoder.text_embedder_batch_size}")
    
    return _text_encoder

def get_table_list():
    """Get list of available tables from datasets directory"""
    tables = []
    for table_dir in Path(DATASETS_DIR).iterdir():
        if table_dir.is_dir():
            rows_file = table_dir / "rows.csv"
            if rows_file.exists():
                tables.append(table_dir.name)
    return sorted(tables)

def embed_single_table(table_id, max_rows=None):
    """
    Embed a single table following PyTorch Frame's standard workflow.
    Each table gets its own encoder instance (RelBench pattern).
    
    Args:
        table_id: Dataset ID (e.g., '2ji4-fd5z')
        max_rows: Maximum number of rows to process (None = all rows)
    
    Returns:
        dict: {
            'embeddings': torch.Tensor [num_cols, channels],
            'col_names': list of column names,
            'roles': torch.Tensor [num_cols],
            'table_id': str,
        }
    """
    csv_path = Path(DATASETS_DIR) / table_id / "rows.csv"
    
    try:
        # Step 1: Load data
        print(f"\n[{table_id}] Loading data...")
        df = pd.read_csv(csv_path, low_memory=False)  # Avoid dtype warnings
        if max_rows:
            df = df.head(max_rows)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Step 2: Infer semantic types (PyTorch Frame standard)
        col_to_stype = infer_df_stype(df)
        for col in df.columns:
            if col not in col_to_stype:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_to_stype[col] = stype.numerical
                else:
                    col_to_stype[col] = stype.categorical
        
        # Step 2.1: Handle multicategorical columns (PyTorch Frame standard approach)
        # Following PyTorch Frame's pattern: multicategorical columns need col_to_sep
        # - If data is list type: col_to_sep can be None
        # - If data is string type: col_to_sep must be provided (e.g., "|", ",")
        # - If separator cannot be determined: convert to categorical
        col_to_sep = {}  # Will be passed to Dataset if we have multicategorical columns
        POSSIBLE_SEPS = ["|", ","]  # PyTorch Frame's default separators (from infer_stype.py)
        
        for col, st in list(col_to_stype.items()):
            if st == stype.multicategorical:
                sample_values = df[col].dropna().head(500)
                if len(sample_values) == 0:
                    # Empty column, convert to categorical
                    col_to_stype[col] = stype.categorical
                    print(f"  Converted {col} from multicategorical to categorical (empty column)")
                    continue
                
                # Check if data is list type (col_to_sep can be None)
                first_val = sample_values.iloc[0]
                if isinstance(first_val, (list, np.ndarray)):
                    # List type: col_to_sep can be None (PyTorch Frame allows this)
                    col_to_sep[col] = None
                    print(f"  Multicategorical column '{col}' is list type, col_to_sep=None")
                    continue
                
                # String type: must detect separator (PyTorch Frame requirement)
                # Try PyTorch Frame's POSSIBLE_SEPS in order
                detected_sep = None
                for sep in POSSIBLE_SEPS:
                    try:
                        # Test if separator actually splits values into multiple parts
                        split_success = 0
                        for val in sample_values:
                            val_str = str(val)
                            if sep in val_str:
                                parts = [p.strip() for p in val_str.split(sep) if p.strip()]
                                if len(parts) > 1:  # Actually splits into multiple parts
                                    split_success += 1
                        
                        # If separator works for a reasonable portion of values, use it
                        if split_success > len(sample_values) * 0.1:  # At least 10% of values
                            detected_sep = sep
                            break
                    except Exception:
                        continue
                
                if detected_sep is not None:
                    col_to_sep[col] = detected_sep
                    print(f"  Multicategorical column '{col}': detected separator '{detected_sep}'")
                else:
                    # No valid separator found, convert to categorical
                    col_to_stype[col] = stype.categorical
                    print(f"  Converted {col} from multicategorical to categorical (no valid separator found)")
        
        # Step 2.5: Configure text embedding (PyTorch Frame standard)
        col_to_text_embedder_cfg = None
        if USE_TEXT_EMBEDDING:
            text_encoder = get_text_encoder()
            if text_encoder is not None:
                # Find text columns
                text_cols = [col for col, st in col_to_stype.items() 
                           if st == stype.text_embedded or st == stype.text_tokenized]
                
                if text_cols:
                    print(f"  Found {len(text_cols)} text columns: {text_cols[:3]}{'...' if len(text_cols) > 3 else ''}")
                    # Use recommended batch size if available (PyTorch Frame pattern)
                    # API classes have text_embedder_batch_size, local models use config value
                    batch_size = TEXT_EMBEDDER_BATCH_SIZE
                    if hasattr(text_encoder, 'text_embedder_batch_size'):
                        batch_size = text_encoder.text_embedder_batch_size
                    
                    # Create TextEmbedderConfig for each text column
                    # This matches PyTorch Frame's pattern: TextEmbedderConfig(text_embedder=..., batch_size=...)
                    col_to_text_embedder_cfg = {
                        col: TextEmbedderConfig(
                            text_embedder=text_encoder,
                            batch_size=batch_size,
                        )
                        for col in text_cols
                    }
                else:
                    # No text columns, convert text_embedded to categorical as fallback
                    for col, st in col_to_stype.items():
                        if st == stype.text_embedded or st == stype.text_tokenized:
                            col_to_stype[col] = stype.categorical
        else:
            # Convert text_embedded to categorical (fallback)
            for col, st in col_to_stype.items():
                if st == stype.text_embedded or st == stype.text_tokenized:
                    col_to_stype[col] = stype.categorical
        
        # Step 3: Create Dataset and materialize (PyTorch Frame standard)
        print(f"  Materializing...")
        # Build Dataset kwargs following PyTorch Frame's standard pattern
        dataset_kwargs = {
            'df': df,
            'col_to_stype': col_to_stype,
            'target_col': None,
        }
        # Add col_to_sep if we have multicategorical columns (PyTorch Frame requirement)
        if col_to_sep:
            dataset_kwargs['col_to_sep'] = col_to_sep
        # Add text embedder config if available
        if col_to_text_embedder_cfg is not None:
            dataset_kwargs['col_to_text_embedder_cfg'] = col_to_text_embedder_cfg
        
        dataset = Dataset(**dataset_kwargs)
        dataset.materialize()
        
        # Step 4: Create independent encoder for this table
        print(f"  Creating encoder...")
        encoder = TableEmbedding(channels=CHANNELS, num_roles=3, role_dim=ROLE_DIM)
        encoder._init_encoder(
            col_stats=dataset.col_stats,
            col_names_dict=dataset.tensor_frame.col_names_dict,
        )
        encoder.eval()
        encoder = encoder.to(device)
        
        # Step 5: Build roles
        roles = build_roles_for_df(ANALYSIS_JSON, table_id, df)
        roles = roles.unsqueeze(0).to(device)  # [1, num_cols]
        
        # Step 6: Encode using DataLoader
        print(f"  Encoding...")
        loader = DataLoader(dataset.tensor_frame, batch_size=BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            batch_embeddings = []
            for batch_tf in loader:
                batch_tf = batch_tf.to(device)
                batch_emb = encoder(batch_tf, roles)
                batch_embeddings.append(batch_emb[0])
            
            # Average across batches (column-level embeddings should be identical)
            emb = torch.stack(batch_embeddings).mean(dim=0)  # [num_cols, channels]
        
        print(f"  ✓ Embedding shape: {emb.shape}")
        
        return {
            'embeddings': emb.cpu(),
            'col_names': list(df.columns),
            'roles': roles[0].cpu(),
            'table_id': table_id,
            'num_rows': len(df),
        }
    
    except Exception as e:
        print(f"  ✗ Error processing {table_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Process all tables in datasets directory (9 tables)"""
    print("="*60)
    print("Batch Embedding for Multiple Tables")
    print("Following PyTorch Frame + RelBench pattern:")
    print("  - Each table gets its own encoder instance")
    print("  - Independent col_stats for each table")
    print(f"Source: {DATASETS_DIR}")
    print("="*60)
    
    # Get list of tables
    tables = get_table_list()
    print(f"\nFound {len(tables)} tables in {DATASETS_DIR}")
    print(f"Tables: {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}")
    
    # Process each table
    results = {}
    for table_id in tqdm(tables, desc="Processing tables"):
        result = embed_single_table(table_id, max_rows=MAX_ROWS_PER_TABLE)
        if result is not None:
            results[table_id] = result
            
            # Save individual table embedding
            out_npz = Path(OUTPUT_DIR) / f"{table_id}_embeddings.npz"
            out_json = Path(OUTPUT_DIR) / f"{table_id}_metadata.json"
            
            np.savez_compressed(
                out_npz,
                col_names=np.array(result['col_names'], dtype=object),
                column_embeddings=result['embeddings'].numpy(),
            )
            
            with open(out_json, 'w') as f:
                json.dump({
                    'table_id': result['table_id'],
                    'col_names': result['col_names'],
                    'roles': result['roles'].tolist(),
                    'num_rows': result['num_rows'],
                    'embedding_shape': list(result['embeddings'].shape),
                }, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Completed: {len(results)}/{len(tables)} tables processed successfully")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")
    
    # Save summary
    summary = {
        'total_tables': len(tables),
        'successful': len(results),
        'failed': len(tables) - len(results),
        'table_ids': list(results.keys()),
        'output_dir': str(OUTPUT_DIR),
    }
    
    with open(Path(OUTPUT_DIR) / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {Path(OUTPUT_DIR) / 'summary.json'}")

if __name__ == "__main__":
    main()

