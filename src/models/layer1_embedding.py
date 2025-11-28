import torch
import torch.nn as nn
import json
import pandas as pd
from typing import Dict, List, Optional, Any

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data import Dataset
from torch_frame.data.stats import compute_col_stats, StatType
from torch_frame.nn.encoder import StypeWiseFeatureEncoder
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    LinearBucketEncoder,
    LinearEmbeddingEncoder,
    MultiCategoricalEmbeddingEncoder,
    TimestampEncoder,
)
from torch_frame.utils import infer_df_stype


class TableEmbedding(nn.Module):
    """
    Table embedding model using PyTorch Frame's StypeWiseFeatureEncoder.
    
    This class wraps PyTorch Frame's standard feature encoder and adds
    role-based embeddings (target, join, attribute) on top.
    """
    def __init__(
        self,
        channels: int = 256,
        num_roles: int = 3,
        role_dim: Optional[int] = None,
        stype_encoder_dict: Optional[Dict] = None,
    ):
        super().__init__()
        self.channels = channels
        self.role_dim = role_dim if role_dim else channels
        
        # Role embedding: 0=target, 1=join, 2=attribute
        self.role_emb = nn.Embedding(num_roles, self.role_dim)
        
        # If role_dim and channels differ, need a projection layer
        self.project = None
        if role_dim and role_dim != channels:
            self.project = nn.Linear(channels, role_dim)
        
        # PyTorch Frame's StypeWiseFeatureEncoder (lazy initialization)
        # Will be initialized when col_stats and col_names_dict are available
        self.encoder = None
        self.stype_encoder_dict = stype_encoder_dict or self._default_stype_encoder_dict()
    
    def _default_stype_encoder_dict(self):
        """Default stype encoder configuration using PyTorch Frame's standard encoders
        
        Returns:
            Dict mapping stype to encoder:
            - categorical: EmbeddingEncoder (lookup table based)
            - numerical: LinearBucketEncoder (bucket-based encoding)
            - embedding: LinearEmbeddingEncoder (for pre-computed text embeddings)
            - multicategorical: MultiCategoricalEmbeddingEncoder (for multi-value categorical columns)
            - timestamp: TimestampEncoder (for timestamp columns)
        """
        return {
            stype.categorical: EmbeddingEncoder(),
            stype.numerical: LinearBucketEncoder(),
            stype.embedding: LinearEmbeddingEncoder(),  # For text_embedded columns
            stype.multicategorical: MultiCategoricalEmbeddingEncoder(),  # For multicategorical columns
            stype.timestamp: TimestampEncoder(),  # For timestamp columns
        }

    def _init_encoder(self, col_stats: Dict, col_names_dict: Dict):
        """Initialize PyTorch Frame's StypeWiseFeatureEncoder.
        
        Args:
            col_stats: Column statistics (from Dataset.col_stats or compute_col_stats)
            col_names_dict: Column names dictionary (from TensorFrame.col_names_dict)
        """
        if self.encoder is None:
            # Initialize PyTorch Frame's standard feature encoder
            self.encoder = StypeWiseFeatureEncoder(
                out_channels=self.channels,
                col_stats=col_stats,
                col_names_dict=col_names_dict,
                stype_encoder_dict=self.stype_encoder_dict,
            )
            # Move encoder to the correct device
            device = next(self.parameters()).device
            self.encoder = self.encoder.to(device)
    
    def forward(
        self,
        tensor_frame: TensorFrame,
        roles: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Forward pass using PyTorch Frame's standard encoding pipeline.
        
        Args:
            tensor_frame: PyTorch Frame's TensorFrame [num_rows, ...]
            roles: [batch_size, num_cols] role id for each column 
                   (0=target, 1=join, 2=attribute)
        
        Returns:
            List[Tensor]: Column embeddings for each sample [num_cols, role_dim]
            The embeddings are aggregated over rows (mean pooling) to get
            column-level representations.
        """
        # Ensure encoder is initialized
        if self.encoder is None:
            raise RuntimeError("Encoder not initialized. Call _init_encoder first.")
        
        # Step 1: Encode using PyTorch Frame's StypeWiseFeatureEncoder
        # This applies stype-specific encoders and concatenates results
        # Output: [num_rows, num_cols, channels] (num_rows is the number of rows in DataFrame)
        x, col_names = self.encoder(tensor_frame)  # x: [num_rows, num_cols, channels]
        
        # Step 1.5: Aggregate over rows to get column-level embeddings
        # Mean pooling: [num_rows, num_cols, channels] → [num_cols, channels]
        x = x.mean(dim=0)  # [num_cols, channels]
        
        # Step 2: Project to role_dim (if needed)
        if self.project is not None:
            x = self.project(x)  # [num_cols, role_dim]
        
        # Step 3: Add role embedding to each column
        # roles shape: [batch_size, num_cols] (usually batch_size=1)
        B = roles.shape[0]  # batch_size (usually 1)
        num_cols = x.shape[0]
        out_with_roles = []
        for b in range(B):
            col_embed = x  # [num_cols, role_dim]
            role_vec = self.role_emb(roles[b][:num_cols])  # [num_cols, role_dim]
            out_with_roles.append(col_embed + role_vec)
        
        return out_with_roles  # List[Tensor[num_cols, role_dim]]


class TableEmbeddingWithDataFrame(nn.Module):
    """
    Convenience wrapper class: encode directly from DataFrame using PyTorch Frame.
    
    This class provides a high-level interface that:
    1. Automatically infers semantic types using PyTorch Frame's infer_df_stype()
    2. Creates TensorFrame using PyTorch Frame's Dataset API
    3. Uses PyTorch Frame's StypeWiseFeatureEncoder for encoding
    4. Adds role embeddings (target, join, attribute) on top
    
    Replaces the original TableEmbeddingWithTokenizer (TAPAS-based).
    """
    def __init__(
        self,
        channels: int = 256,
        num_roles: int = 3,
        role_dim: Optional[int] = None,
        col_to_stype: Optional[Dict[str, stype]] = None,
        stype_encoder_dict: Optional[Dict] = None,
    ):
        super().__init__()
        self.encoder_model = TableEmbedding(
            channels=channels,
            num_roles=num_roles,
            role_dim=role_dim,
            stype_encoder_dict=stype_encoder_dict,
        )
        self.col_to_stype = col_to_stype
        self._col_stats = None
        self._col_names_dict = None
    
    def _device(self):
        """infer current device"""
        return next(self.encoder_model.parameters()).device
    
    def _detect_stypes(self, df: pd.DataFrame) -> Dict[str, stype]:
        """Detect column types using PyTorch Frame's automatic inference"""
        if self.col_to_stype is not None:
            return self.col_to_stype
        
        # Use PyTorch Frame's automatic semantic type inference
        col_to_stype = infer_df_stype(df)
        
        # Handle columns that failed inference (return None)
        for col in df.columns:
            if col not in col_to_stype:
                # Fallback for columns that couldn't be inferred
                ser = df[col]
                if pd.api.types.is_numeric_dtype(ser):
                    col_to_stype[col] = stype.numerical
                else:
                    col_to_stype[col] = stype.categorical
        
        return col_to_stype
    
    def _compute_col_stats(self, df: pd.DataFrame, col_to_stype: Dict[str, stype]):
        """Compute column statistics using PyTorch Frame's standard method"""
        col_stats = {}
        for col, stype_val in col_to_stype.items():
            ser = df[col]
            col_stats[col] = compute_col_stats(ser, stype_val)
        return col_stats
    
    def _create_tensor_frame(self, df: pd.DataFrame) -> TensorFrame:
        """Create TensorFrame from DataFrame using PyTorch Frame's standard workflow
        
        Note: For multi-table scenarios (like RelBench), each table should have its own
        encoder instance. This method is designed for processing batches of the same table.
        """
        # Step 1: Detect semantic types (using PyTorch Frame's automatic inference)
        col_to_stype = self._detect_stypes(df)
        
        # Step 2: Create Dataset using PyTorch Frame's standard API
        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            target_col=None,  # No target column for embedding
        )
        
        # Step 3: Materialize to get TensorFrame
        # Let PyTorch Frame compute col_stats automatically (avoids library bugs)
        # This also ensures col_stats is always consistent with the current DataFrame
        dataset.materialize()
        tensor_frame = dataset.tensor_frame
        
        # Step 4: Save col_stats and col_names_dict (for initializing encoder, only once)
        if self._col_stats is None:
            self._col_stats = dataset.col_stats
        if self._col_names_dict is None:
            self._col_names_dict = tensor_frame.col_names_dict
        
        # Step 5: Initialize encoder (if not initialized)
        # StypeWiseFeatureEncoder needs col_stats and col_names_dict
        if self.encoder_model.encoder is None:
            self.encoder_model._init_encoder(
                col_stats=self._col_stats,
                col_names_dict=self._col_names_dict,
            )
        
        return tensor_frame
    
    def encode_dataframe(
        self,
        df: pd.DataFrame,
        roles_for_df: torch.Tensor,
    ) -> torch.Tensor:
        """
        encode DataFrame to get column embeddings
        
        Args:
            df: pandas DataFrame
            roles_for_df: [num_cols] role id for each column
        
        Returns:
            Tensor: [num_cols, channels] column embeddings
        """
        # create TensorFrame
        tensor_frame = self._create_tensor_frame(df)
        
        # add batch dimension
        roles = roles_for_df.unsqueeze(0)  # [1, num_cols]
        
        # encode
        device = self._device()
        tensor_frame = tensor_frame.to(device)
        roles = roles.to(device)
        
        outputs = self.encoder_model(tensor_frame, roles)
        return outputs[0]  # return the first (and only) sample
    
    def encode_dataframe_with_roles_json(
        self,
        df: pd.DataFrame,
        analysis_json_path: str,
        dataset_id: str,
    ) -> torch.Tensor:
        """
        Convenience: build roles from analysis_results_optimized.json and encode.
        """
        roles_for_df = build_roles_for_df(analysis_json_path, dataset_id, df)
        return self.encode_dataframe(df, roles_for_df)


# to keep backward compatibility, keep the original class name as an alias
TableEmbeddingWithTokenizer = TableEmbeddingWithDataFrame


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