from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

df = pd.read_csv("full_dataset_30ms_scaled.csv")

print("Fold distribution:")
print(df["fold"].value_counts())
df_files = df.groupby("file")["fold"].first().value_counts()
print("🎧 Files per fold:")
print(df_files)


# Split
train = df[df["fold"] != 5]
test = df[df["fold"] == 5]

X_train = train.iloc[:, :39]
y_train = train["label"]
X_test = test.iloc[:, :39]
y_test = test["label"]

# Train
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Frame-level accuracy: {acc:.4f}")
print(f"✅ Frame-level F1 score: {f1:.4f}")
