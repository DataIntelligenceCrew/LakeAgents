import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

path = "/localdisk3/ytang49/opendata/original_query_table/rvmf-4sg6/rows.csv"
df = pd.read_csv(path)

# 只保留需要的列，并 dropna
df_xy = df[['Overall DV Rate','Shooting Rate']].dropna()

X = df_xy[['Shooting Rate']]
y = df_xy['Overall DV Rate']

print("df_xy shape:", df_xy.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("R^2:", model.score(X_test, y_test))