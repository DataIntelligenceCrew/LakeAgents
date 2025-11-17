#!/usr/bin/env python3
"""
Data Preprocessor
Load and preprocess raw CSV tables for Layer 1 embedding
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pickle
import json
from collections import Counter
import re

class ColumnTypeDetector:
    """Detect column data type: numerical, categorical, or text"""
    
    def __init__(self, max_unique_for_categorical: int = 100):
        """
        Args:
            max_unique_for_categorical: if unique count is less than this value, treat as categorical, otherwise treat as text
        """
        self.max_unique_for_categorical = max_unique_for_categorical
    
    def detect_type(self, column: pd.Series) -> str:
        """
        Detect the data type of a column
        
        Returns:
            'numerical', 'categorical', or 'text'
        """
        # 1. Check if the column is numerical
        if pd.api.types.is_numeric_dtype(column):
            unique_count = column.nunique()
            # if the unique count is less than 10, treat as categorical
            if unique_count <= 20:
                return 'categorical'
            else:
                return 'numerical'
        
        # 2. Check if the column is categorical
        elif pd.api.types.is_object_dtype(column) or pd.api.types.is_categorical_dtype(column):
            unique_count = column.nunique()
            
            if unique_count <= self.max_unique_for_categorical:
                return 'categorical'
            else:
                return 'text'
        
        # 3. Default to text
        else:
            return 'text'

class NumericalBinner:
    """Convert numerical columns to bins"""
    
    def __init__(self, num_bins: int = 10):
        """
        Args:
            num_bins: 分成多少个区间
        """
        self.num_bins = num_bins
        self.bin_edges = {}  # save the bin edges for each column
    
    def fit(self, column: pd.Series, column_name: str):
        """
        Learn the bin edges
        
        Args:
            column: numerical column
            column_name: column name (for saving the bin edges)
        """
        # Remove NaN
        values = column.dropna()
        
        if len(values) == 0:
            self.bin_edges[column_name] = None
            return
        
        # Calculate the quantiles as the bin edges (more uniform distribution)
        quantiles = np.linspace(0, 1, self.num_bins + 1)
        edges = np.percentile(values, quantiles * 100)
        
        # Ensure the bin edges are unique (handle duplicate values)
        edges = np.unique(edges)
        
        self.bin_edges[column_name] = edges
    
    def transform(self, column: pd.Series, column_name: str) -> pd.Series:
        """
        Convert the numerical values to bin IDs
        
        Returns:
            Series of bin IDs (0, 1, 2, ..., num_bins-1)
        """
        edges = self.bin_edges.get(column_name)
        
        if edges is None or len(edges) < 2:
            # Cannot bin, return all 0
            return pd.Series([0] * len(column), index=column.index)
        
        # Use pd.cut to bin
        binned = pd.cut(column, bins=edges, labels=False, include_lowest=True, duplicates='drop')
        
        # Fill NaN with the special bin (the last bin)
        max_bin = len(edges) - 2  # pd.cut returns 0 to n-2
        binned = binned.fillna(max_bin)
        
        return binned.astype(int)


class CategoricalEncoder:
    """Encode categorical columns to integer IDs"""
    
    def __init__(self, unknown_token: str = '<UNK>'):
        """
        Args:
            unknown_token: token for unknown values
        """
        self.unknown_token = unknown_token
        self.vocab = {}  # {column_name: {value: id}}
        self.id_to_value = {}  # {column_name: {id: value}}
    
    def fit(self, column: pd.Series, column_name: str):
        """
        Build the vocabulary
        
        Args:
            column: categorical column
            column_name: column name
        """
        # Get all unique values (including NaN)
        unique_values = column.unique()
        
        # Create the mapping: value -> id
        # ID 0 is reserved for <UNK>
        vocab = {self.unknown_token: 0}
        
        current_id = 1
        for value in unique_values:
            # Skip NaN
            if pd.isna(value):
                continue
            
            value_str = str(value)
            if value_str not in vocab:
                vocab[value_str] = current_id
                current_id += 1
        
        self.vocab[column_name] = vocab
        
        # Reverse mapping: id -> value
        self.id_to_value[column_name] = {v: k for k, v in vocab.items()}
    
    def transform(self, column: pd.Series, column_name: str) -> pd.Series:
        """
        Convert the categorical values to integer IDs
        
        Returns:
            Series of integer IDs
        """
        vocab = self.vocab.get(column_name, {})
        
        if not vocab:
            # No vocabulary, return all 0
            return pd.Series([0] * len(column), index=column.index)
        
        # Convert the categorical values to integer IDs
        def encode_value(val):
            if pd.isna(val):
                return vocab.get(self.unknown_token, 0)
            val_str = str(val)
            return vocab.get(val_str, vocab.get(self.unknown_token, 0))
        
        return column.apply(encode_value)

class TextTokenizer:
    """Tokenize text columns"""
    
    def __init__(
        self,
        max_tokens: int = 200,
        max_cell_length: int = 200,
        lowercase: bool = True,
        remove_punctuation: bool = False
    ):
        """
        Args:
            max_tokens: maximum number of tokens to keep
            max_cell_length: maximum cell length
            lowercase: whether to convert to lowercase
            remove_punctuation: whether to remove punctuation
        """
        self.max_tokens = max_tokens
        self.max_cell_length = max_cell_length
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.vocab = {}  # global vocabulary
        self.special_tokens = {
            '<PAD>': 0,   # padding
            '<UNK>': 1,   # unknown
            '<NUM>': 2    # number
        }
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text
        
        Returns:
            List of tokens
        """
        if pd.isna(text):
            return []
        
        # Convert to string and truncate
        text = str(text)[:self.max_cell_length]
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation (optional)
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Simple space tokenization
        tokens = text.split()
        
        # Replace numbers with <NUM>
        tokens = [t if not t.isdigit() else '<NUM>' for t in tokens]
        
        # Truncate
        return tokens[:self.max_tokens]
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """
        Build the vocabulary from a list of texts
        
        Args:
            texts: list of texts
            min_freq: minimum frequency (tokens with frequency less than this are mapped to <UNK>)
        """
        # Count the frequency of each token
        token_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_counts.update(tokens)
        
        # Build the vocabulary
        self.vocab = self.special_tokens.copy()
        current_id = len(self.special_tokens)
        
        for token, count in token_counts.most_common():
            if count >= min_freq and token not in self.vocab:
                self.vocab[token] = current_id
                current_id += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Encode the text to a sequence of integer IDs
        
        Returns:
            List of integer IDs
        """
        tokens = self.tokenize(text)
        
        # Convert to IDs
        ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        
        # Pad to max_tokens length
        if len(ids) < self.max_tokens:
            ids += [self.vocab['<PAD>']] * (self.max_tokens - len(ids))
        
        return ids

class TablePreprocessor:
    """
    Main preprocessor class that integrates all components
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: data_config from YAML
        """
        self.config = config
        
        # Initialize all components
        self.type_detector = ColumnTypeDetector(
            max_unique_for_categorical=config['preprocessing']['categorical_encoding']['max_unique_values']
        )
        self.numerical_binner = NumericalBinner(
            num_bins=config['preprocessing']['numerical_binning']['num_bins']
        )
        self.categorical_encoder = CategoricalEncoder(
            unknown_token=config['preprocessing']['categorical_encoding']['unknown_token']
        )
        self.text_tokenizer = TextTokenizer(
            max_tokens=config['preprocessing']['text_processing']['max_tokens'],
            max_cell_length=config['preprocessing']['max_cell_length'],
            lowercase=config['preprocessing']['text_processing']['lowercase'],
            remove_punctuation=config['preprocessing']['text_processing']['remove_punctuation']
        )
        
        # Save the column types information
        self.column_types = {}  # {table_id: {col_name: type}}
        self.column_roles = {}  # {table_id: {col_name: role}}
    
    def load_table(self, table_path: str, dataset_id: str, 
                   max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load and sample the table
        
        Args:
            table_path: CSV file path
            dataset_id: dataset ID
            max_rows: maximum number of rows (sampling)
            
        Returns:
            Sampled DataFrame
        """
        df = pd.read_csv(table_path)
        
        # Sampling (if needed)
        if max_rows and len(df) > max_rows:
            sample_strategy = self.config['preprocessing']['sample_strategy']
            
            if sample_strategy == 'random':
                df = df.sample(n=max_rows, random_state=self.config['random_seed'])
            elif sample_strategy == 'head':
                df = df.head(max_rows)
            # stratified sampling needs to know the target column
        
        return df
    
    def fit(self, tables: Dict[str, pd.DataFrame], 
            dataset_info: Dict[str, Any]):
        """
        Fit all encoders on the training set
        
        Args:
            tables: {table_name: dataframe}
            dataset_info: metadata from analysis_results
        """
        # Iterate over all tables and columns
        for table_name, df in tables.items():
            for col_name in df.columns:
                column = df[col_name]
                
                # Detect the column type
                col_type = self.type_detector.detect_type(column)
                self.column_types.setdefault(table_name, {})[col_name] = col_type
                
                # Fit the appropriate encoder based on the type
                if col_type == 'numerical':
                    self.numerical_binner.fit(column, f"{table_name}.{col_name}")
                
                elif col_type == 'categorical':
                    self.categorical_encoder.fit(column, f"{table_name}.{col_name}")
                
                elif col_type == 'text':
                    # Text columns need to collect all texts to build the vocabulary
                    pass  # will be handled later    
        
        # Build the vocabulary for all text columns
        all_texts = []
        for table_name, df in tables.items():
            for col_name, col_type in self.column_types[table_name].items():
                if col_type == 'text':
                    all_texts.extend(df[col_name].dropna().astype(str).tolist())
        
        if all_texts:
            self.text_tokenizer.build_vocab(all_texts, min_freq=2)
    
    def transform(self, df: pd.DataFrame, table_name: str) -> Dict[str, np.ndarray]:
        """
        Transform the table
        
        Returns:
            {col_name: encoded_values}
        """
        result = {}
        
        for col_name in df.columns:
            col_type = self.column_types[table_name].get(col_name, 'text')
            column = df[col_name]
            
            if col_type == 'numerical':
                encoded = self.numerical_binner.transform(column, f"{table_name}.{col_name}")
            
            elif col_type == 'categorical':
                encoded = self.categorical_encoder.transform(column, f"{table_name}.{col_name}")
            
            elif col_type == 'text':
                # Text needs special handling (returns 2D array)
                encoded = [self.text_tokenizer.encode(text) for text in column]
                encoded = np.array(encoded)
            
            result[col_name] = encoded.values if hasattr(encoded, 'values') else encoded
        
        return result
    
    def save(self, save_path: str):
        """Save the preprocessor state"""
        state = {
            'column_types': self.column_types,
            'column_roles': self.column_roles,
            'numerical_binner': self.numerical_binner.bin_edges,
            'categorical_encoder': {
                'vocab': self.categorical_encoder.vocab,
                'id_to_value': self.categorical_encoder.id_to_value
            },
            'text_tokenizer': {
                'vocab': self.text_tokenizer.vocab
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, load_path: str):
        """Load the preprocessor state"""
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        self.column_types = state['column_types']
        self.column_roles = state['column_roles']
        self.numerical_binner.bin_edges = state['numerical_binner']
        self.categorical_encoder.vocab = state['categorical_encoder']['vocab']
        self.categorical_encoder.id_to_value = state['categorical_encoder']['id_to_value']
        self.text_tokenizer.vocab = state['text_tokenizer']['vocab']