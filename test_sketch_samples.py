"""Test: get original values that form the bottom-k sketch."""
import pandas as pd
import numpy as np
from tools.sketch import (
    bottom_k_sketch_column,
    bottom_k_sketch_column_with_samples,
    hash_value_to_unit,
)

# Small toy column (e.g. like borough names, with duplicates)
df = pd.DataFrame({
    "boro": ["BRONX", "MANHATTAN", "BROOKLYN", "QUEENS", "STATEN ISLAND"] * 4,
    "x": range(20),
})
col = df["boro"]

# Use k=3 so we only get 3 distinct values in the sketch (easy to inspect)
k = 3
sketch_only = bottom_k_sketch_column(col, k=k)
sketch, original_values, column_name = bottom_k_sketch_column_with_samples(col, k=k)

print("Column name:", repr(column_name))
print("Sketch (hashes):", sketch)
print("Original values (bottom-k samples):", original_values)
print()

# Verify: re-hash each original value and check it appears in sketch
print("Verify: each original value hashes into the sketch")
for v in original_values:
    h = hash_value_to_unit(v)
    in_sketch = np.isclose(h, sketch).any()
    print(f"  {repr(v)} -> hash={h:.6f}  in_sketch={in_sketch}")
print("Sketch and sketch_only match:", np.allclose(sketch, sketch_only))
