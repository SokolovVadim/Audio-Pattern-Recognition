import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Load data
df = pd.read_csv("full_dataset_30000ms_extended.csv")

print("Fold distribution:")
print(df["fold"].value_counts())
print("Files per fold:")
print(df.groupby("file")["fold"].first().value_counts())

# Split into train/test
train = df[df["fold"] != 5]
test = df[df["fold"] == 5]

X_train = train.iloc[:, :42].values
y_train = train["label"].values
X_test = test.iloc[:, :42].values
y_test = test["label"].values

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train label distribution:", pd.Series(y_train).value_counts())
print("Test label distribution:", pd.Series(y_test).value_counts())


classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))


# Train
# clf = LogisticRegression(solver='liblinear', max_iter=1000)
clf = RandomForestClassifier(n_estimators=1, warm_start=True, class_weight=class_weights, random_state=39, n_jobs=12)
n_estimators = 100
clf.set_params(n_estimators=1)
#clf.fit(X_train, y_train)
# Manual training loop with progress bar
for i in tqdm(range(1, n_estimators + 1)):
    clf.set_params(n_estimators=i)
    clf.fit(X_train, y_train)  # Re-fits all trees with added one

# Predict
y_pred = clf.predict(X_test)

# Evaluate frame-level
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Add predictions to test DataFrame
test_df = test.copy()
test_df["pred"] = y_pred

# Majority vote per file
from collections import Counter

file_preds = []
file_truths = []

for file, group in test_df.groupby("file"):
    pred_majority = Counter(group["pred"]).most_common(1)[0][0]
    true_label = group["label"].iloc[0]
    
    file_preds.append(pred_majority)
    file_truths.append(true_label)

# Evaluate file-level
file_acc = accuracy_score(file_truths, file_preds)
file_f1 = f1_score(file_truths, file_preds)

cm_file = confusion_matrix(file_truths, file_preds)
cm_file_percent = cm_file / cm_file.sum() * 100

# Print results
print(f"\nFrame-level accuracy: {acc:.4f}")
print(f"Frame-level F1 score: {f1:.4f}")

print(f"\nFile-level Accuracy (majority vote): {file_acc:.4f}")
print(f"File-level F1 Score: {file_f1:.4f}")
print("Confusion Matrix (in %):")
print(np.round(cm_file_percent, 2))
