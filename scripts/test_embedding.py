# test_embedding.py
import os, json, numpy as np, pandas as pd, torch
from src.models.layer1_embedding import TableEmbeddingWithTokenizer, build_roles_for_df

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

DATA_ID = "2ji4-fd5z"
CSV_PATH = "/localdisk3/ytang49/opendata/datasets_opendata/2ji4-fd5z/rows.csv"
ANALYSIS_JSON = "/localdisk3/ytang49/opendata/analysis_results_optimized.json"
OUT_NPZ = "/localdisk3/ytang49/opendata/processed_data/embedding_test/2ji4-fd5z_inputs_head1k.npz"
OUT_ROLES = "/localdisk3/ytang49/opendata/processed_data/embedding_test/2ji4-fd5z_roles_head1k.json"

os.makedirs("/localdisk3/ytang49/opendata/processed_data/embedding_test", exist_ok=True)

MAX_ROWS = 1000
BATCH_ROWS = 512

df = pd.read_csv(CSV_PATH).astype(str).head(MAX_ROWS)
encoder = TableEmbeddingWithTokenizer(hf_model_name="google/tapas-base", role_dim=768, pooling="mean", max_length=512)
encoder.eval()
encoder = encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

roles = build_roles_for_df(ANALYSIS_JSON, DATA_ID, df)  # torch.long [C]
# batch encoding and mean pooling
acc_sum = None
acc_cnt = 0
with torch.no_grad(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
    for start in range(0, len(df), BATCH_ROWS):
        # Ensure each chunk is a fresh, contiguous DataFrame to satisfy TAPAS tokenizer expectations
        part = df.iloc[start:start + BATCH_ROWS].reset_index(drop=True).copy(deep=True)
        part.columns = part.columns.astype(str)
        emb_part = encoder.encode_dataframe(part, roles)  # [C, 768]
        acc_sum = emb_part.detach() if acc_sum is None else acc_sum + emb_part.detach()
        acc_cnt += 1
emb = acc_sum / max(acc_cnt, 1)

np.savez_compressed(OUT_NPZ, col_names=np.array(df.columns, dtype=object), column_embeddings=emb.detach().cpu().numpy())
with open(OUT_ROLES, "w") as f:
    json.dump({"col_names": list(df.columns), "roles": roles.tolist()}, f, indent=2)

print("Embedding shape:", emb.shape, "device:", emb.device)
print("Saved NPZ:", OUT_NPZ)
print("Saved roles:", OUT_ROLES)