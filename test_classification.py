import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.text_integration import embed_texts_with_fasttext


path = "/localdisk3/ytang49/opendata/original_query_table/Food Inspections-Chicago/rows.csv"
df = pd.read_csv(path)

target_col = "Risk"
X = df[['Facility Type']]
y = df[target_col]

used_cols = list(X.columns) + [target_col]
df = df.dropna(subset=used_cols)

# for col in X.select_dtypes(include="object").columns:
#     X[col] = X[col].astype("category").cat.codes
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

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print(f"Train classes: {len(np.unique(y_train_encoded))}")
print(f"Test classes: {len(np.unique(y_test_encoded))}")

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train_encoded)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {acc:.5f}")