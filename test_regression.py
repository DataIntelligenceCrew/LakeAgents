import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.text_integration import embed_texts_with_fasttext

path = "/localdisk3/ytang49/opendata/original_query_table/Taxi-Chicago/rows.csv"
df = pd.read_csv(path)
df = df.head(1000)
target_col = "Trip Total"
feature_cols = [ 'Tips']  
X = df[feature_cols]
y = df[target_col]

used_cols = list(X.columns) + [target_col]
df = df.dropna(subset=used_cols)
X = df[feature_cols]
y = df[target_col]

print("X shape:", X.shape)
print("y shape:", y.shape)

model_path = str(Path(__file__).resolve().parent / "fasttext.bin")
parts = []
for col in X.columns:
    if X[col].dtype == object or X[col].dtype.name == "string":
        texts = X[col].fillna("").astype(str).tolist()
        emb = embed_texts_with_fasttext(texts, model_path=model_path)
        parts.append(emb)
    else:
        parts.append(X[col].values.reshape(-1, 1))
X = np.hstack(parts)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("R^2:", model.score(X_test, y_test))