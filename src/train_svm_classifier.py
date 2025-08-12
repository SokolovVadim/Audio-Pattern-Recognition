import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("full_dataset_100ms_extended.csv")

# print("Fold distribution:")
# print(df["fold"].value_counts())
# print("Files per fold:")
# print(df.groupby("file")["fold"].first().value_counts())

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

print("Train:", np.bincount(y_train))
print("Test:", np.bincount(y_test))


# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 0.01, 0.1, 1, 10],
#     'kernel': ['rbf'],
#     'class_weight': ['balanced']
# }

# grid_search = GridSearchCV(
#     estimator=SVC(),
#     param_grid=param_grid,
#     scoring='f1',  # because F1 is more important than accuracy here
#     cv=5,
#     verbose=2,
#     n_jobs=-1
# )

# grid_search.fit(X_train, y_train)

# print("Best parameters:", grid_search.best_params_)
# print("Best F1 score (CV):", grid_search.best_score_)

# best_svc = SVC(
#     kernel="rbf",
#     C=0.1,
#     gamma=0.1,
#     class_weight="balanced",
#     random_state=42
# )

classes = [0, 1]
all_labels = np.concatenate([y_train, y_test])  # or better, just from entire original data

svc = SVC(C=0.3, gamma=0.01, class_weight={0: 1.1, 1: 0.8})

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

# Add predictions and file info to DataFrame
test_df = test.copy()
test_df["pred"] = y_pred

from collections import Counter

# Majority vote per file
file_preds = []
file_truths = []

for file, group in test_df.groupby("file"):
    pred_majority = Counter(group["pred"]).most_common(1)[0][0]
    true_label = group["label"].iloc[0]
    
    file_preds.append(pred_majority)
    file_truths.append(true_label)

# File-level metrics
file_acc = accuracy_score(file_truths, file_preds)
file_f1 = f1_score(file_truths, file_preds)
file_cm = confusion_matrix(file_truths, file_preds)
file_cm_percent = file_cm / file_cm.sum() * 100

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm / cm.sum() * 100

# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()

# Frame-level results
print(f"\nFrame-level accuracy: {acc:.4f}")
print(f"Frame-level F1 score: {f1:.4f}")

# File-level results
print(f"\nFile-level Accuracy (majority vote): {file_acc:.4f}")
print(f"File-level F1 Score: {file_f1:.4f}")
print("Confusion Matrix (in %):")
print(np.round(file_cm_percent, 2))

