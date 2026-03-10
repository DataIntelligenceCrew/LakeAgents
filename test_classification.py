import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

path = "/localdisk3/ytang49/opendata/query_table/cts7-vksw copy/rows.csv"
df = pd.read_csv(path)

df = df.dropna()

target_col = "HHT"

# X = df[['INTP_adj']]
X = df.drop(columns=[target_col])
y = df[target_col]

# 先编码 X
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category").cat.codes

# 先 split，再编码 y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 在训练集上 fit LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print(f"训练集类别数: {len(np.unique(y_train_encoded))}")
print(f"测试集类别数: {len(np.unique(y_test_encoded))}")

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