# test_embedding.py - Following PyTorch Frame's standard workflow
import os
import sys

# add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
import numpy as np
import pandas as pd
import torch
from torch_frame import stype
from torch_frame.data import Dataset, DataLoader
from src.models.layer1_embedding import TableEmbedding, build_roles_for_df

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

DATA_ID = "2ji4-fd5z"
CSV_PATH = "/localdisk3/ytang49/opendata/datasets_opendata/2ji4-fd5z/rows.csv"
ANALYSIS_JSON = "/localdisk3/ytang49/opendata/analysis_results_optimized.json"
OUT_NPZ = "/localdisk3/ytang49/opendata/processed_data/embedding_test/2ji4-fd5z_inputs_head1k.npz"
OUT_ROLES = "/localdisk3/ytang49/opendata/processed_data/embedding_test/2ji4-fd5z_roles_head1k.json"

os.makedirs("/localdisk3/ytang49/opendata/processed_data/embedding_test", exist_ok=True)

MAX_ROWS = 1000
BATCH_SIZE = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if not torch.cuda.is_available():
    print("Warning: CUDA not available, using CPU")

print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH).head(MAX_ROWS)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Step 1: Create Dataset and materialize (PyTorch Frame standard workflow)
print("Creating Dataset and materializing (computing col_stats from full data)...")
from torch_frame.utils import infer_df_stype
col_to_stype = infer_df_stype(df)

# Handle columns that failed inference
for col in df.columns:
    if col not in col_to_stype:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_to_stype[col] = stype.numerical
        else:
            col_to_stype[col] = stype.categorical

dataset = Dataset(df=df, col_to_stype=col_to_stype, target_col=None)
dataset.materialize()  # Materialize once with full data
print(f"Materialized. col_stats computed for {len(dataset.col_stats)} columns")

# Step 2: Create encoder using full dataset's col_stats (PyTorch Frame standard)
print("Initializing encoder with full dataset's col_stats...")
encoder = TableEmbedding(channels=256, num_roles=3, role_dim=256)
encoder._init_encoder(
    col_stats=dataset.col_stats,
    col_names_dict=dataset.tensor_frame.col_names_dict,
)
encoder.eval()
encoder = encoder.to(device)

# Step 3: Build roles
roles = build_roles_for_df(ANALYSIS_JSON, DATA_ID, df)
roles = roles.unsqueeze(0).to(device)  # [1, num_cols]
print(f"Roles: {roles.shape}")

# Step 4: Use DataLoader to batch (PyTorch Frame standard)
print(f"Creating DataLoader with batch_size={BATCH_SIZE}...")
loader = DataLoader(dataset.tensor_frame, batch_size=BATCH_SIZE, shuffle=False)

# Step 5: Encode in batches
print("Encoding...")
with torch.no_grad():
    batch_embeddings = []
    for batch_tf in loader:
        batch_tf = batch_tf.to(device)
        batch_emb = encoder(batch_tf, roles)  # List[Tensor[num_cols, channels]]
        batch_embeddings.append(batch_emb[0])  # Get first (only) sample
    
    # Average embeddings across batches (though they should be identical for column-level)
    emb = torch.stack(batch_embeddings).mean(dim=0)  # [num_cols, channels]

# Step 6: Save
np.savez_compressed(OUT_NPZ, col_names=np.array(df.columns, dtype=object), 
                    column_embeddings=emb.detach().cpu().numpy())
with open(OUT_ROLES, "w") as f:
    json.dump({"col_names": list(df.columns), "roles": roles[0].tolist()}, f, indent=2)

print("Embedding shape:", emb.shape, "device:", emb.device)
print("Saved NPZ:", OUT_NPZ)
print("Saved roles:", OUT_ROLES)