import torch
import torch.nn as nn
import json
from transformers import TapasModel, TapasTokenizer

class TableEmbedding(nn.Module):
    def __init__(self, hf_model_name: str = "google/tapas-base-finetuned-wtq",
                 num_roles: int = 3, role_dim: int = 768, pooling: str = "mean"):
        super().__init__()
        self.model = TapasModel.from_pretrained(hf_model_name)
        hidden = self.model.config.hidden_size
        self.role_emb = nn.Embedding(num_roles, role_dim if role_dim else hidden)
        self.project = None
        if role_dim and role_dim != hidden:
            self.project = nn.Linear(hidden, role_dim)
        self.pooling = pooling

    @staticmethod
    def _column_and_row_ids(token_type_ids: torch.Tensor):
        # token_type_ids: [B, L, 7]; index 1 -> column_ids, index 2 -> row_ids
        column_ids = token_type_ids[..., 1]
        row_ids = token_type_ids[..., 2]
        return column_ids, row_ids

    def _pool_column(self, hidden: torch.Tensor, cols: torch.Tensor, rows: torch.Tensor):
        # hidden: [L, D]; cols/rows: [L]
        valid = (cols > 0) & (rows >= 1)
        if not torch.any(valid):
            return hidden.new_zeros((0, hidden.size(-1)))
        sel_cols = cols[valid]
        sel_vecs = hidden[valid]
        num_cols = int(sel_cols.max().item())
        col_embeds = []
        for c in range(1, num_cols + 1):
            m = sel_cols == c
            if torch.any(m):
                v = sel_vecs[m]
                if self.pooling == "max":
                    col_embeds.append(v.max(dim=0).values)
                elif self.pooling == "lse":
                    # LogSumExp pooling
                    col_embeds.append(torch.logsumexp(v, dim=0))
                else:
                    col_embeds.append(v.mean(dim=0))
        return torch.stack(col_embeds, dim=0) if col_embeds else hidden.new_zeros((0, hidden.size(-1)))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor, roles: torch.Tensor):
        # inputs: [B, L], [B, L], [B, L, 7]; roles: [B, num_cols] (num_cols is the number of columns in the dataframe)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids, output_hidden_states=False)
        last = outputs.last_hidden_state  # [B, L, D]
        B, L, D = last.shape
        out_cols = []
        for b in range(B):
            cols_b, rows_b = self._column_and_row_ids(token_type_ids[b])
            col_embed = self._pool_column(last[b], cols_b, rows_b)  # [C, D]
            if self.project is not None:
                col_embed = self.project(col_embed)
            out_cols.append(col_embed)
        # role embedding added (per sample)
        out_with_roles = []
        for b, col_embed in enumerate(out_cols):
            if col_embed.numel() == 0:
                out_with_roles.append(col_embed)
                continue
            role_vec = self.role_emb(roles[b][: col_embed.size(0)])  # [C, d]
            out_with_roles.append(col_embed + role_vec)
        return out_with_roles  # List[Tensor[C, d]]

class TableEmbeddingWithTokenizer(nn.Module):
    def __init__(self, hf_model_name: str = "google/tapas-base-finetuned-wtq",
                 num_roles: int = 3, role_dim: int = 768, pooling: str = "mean",
                 max_length: int = 512):
        super().__init__()
        self.tokenizer = TapasTokenizer.from_pretrained(hf_model_name)
        self.encoder = TableEmbedding(hf_model_name, num_roles, role_dim, pooling)
        self.max_length = max_length
    def _device(self):
        # infer the current device from encoder parameters
        return next(self.encoder.parameters()).device
    def encode_dataframe(self, df, roles_for_df):
        enc = self.tokenizer(table=df, queries=[""], padding="max_length",
                             truncation=True, return_tensors="pt", max_length=self.max_length)
        device = self._device()
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        token_type_ids = enc["token_type_ids"].to(device)
        roles = roles_for_df.to(device=device, dtype=torch.long).unsqueeze(0)  # [1, C]
        return self.encoder(input_ids, attention_mask, token_type_ids, roles)[0]

    def encode_dataframe_with_roles_json(self, df, analysis_json_path: str, dataset_id: str):
        """
        Convenience: build roles from analysis_results_optimized.json and encode.
        """
        roles_for_df = build_roles_for_df(analysis_json_path, dataset_id, df)
        return self.encode_dataframe(df, roles_for_df)


def build_roles_for_df(analysis_json_path: str, dataset_id: str, df) -> torch.Tensor:
    """
    Build role ids aligned with DataFrame columns.
      0: target, 1: join, 2: attribute
    """
    with open(analysis_json_path, "r") as f:
        data = json.load(f)
    item = None
    for r in data:
        if r.get("dataset") == dataset_id:
            item = r.get("result")
            break
    # Normalize helpers to make matching robust to whitespace/case differences
    def _norm(x):
        return str(x).strip().casefold()
    target = _norm(item.get("target_column", {}).get("name")) if item else None
    joins = {_norm(j) for j in (item.get("join_columns", []) or [])} if item else set()
    roles_list = []
    for c in df.columns:
        cn = _norm(c)
        if target is not None and cn == target:
            roles_list.append(0)
        elif cn in joins:
            roles_list.append(1)
        else:
            roles_list.append(2)
    return torch.tensor(roles_list, dtype=torch.long)