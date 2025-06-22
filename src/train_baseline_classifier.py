import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("full_dataset_30000ms.csv")

print("Fold distribution:")
print(df["fold"].value_counts())
print("Files per fold:")
print(df.groupby("file")["fold"].first().value_counts())

# Split into train/test
train = df[df["fold"] != 5]
test = df[df["fold"] == 5]

X_train = train.iloc[:, :39].values
y_train = train["label"].values
X_test = test.iloc[:, :39].values
y_test = test["label"].values

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train label distribution:", pd.Series(y_train).value_counts())
print("Test label distribution:", pd.Series(y_test).value_counts())


# Train
clf = LogisticRegression(solver='liblinear', max_iter=1000)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nFrame-level accuracy: {acc:.4f}")
print(f"Frame-level F1 score: {f1:.4f}")
