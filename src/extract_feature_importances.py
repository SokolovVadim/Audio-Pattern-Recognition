import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

feature_names = (
    [f"mfcc_{i+1}" for i in range(13)] +
    [f"delta_{i+1}" for i in range(13)] +
    [f"delta2_{i+1}" for i in range(13)] +
    ["rms"]
)

# Load data
df = pd.read_csv("full_dataset_30000ms.csv")

df.columns = feature_names + list(df.columns[40:])

# Split into train/test
train = df[df["fold"] != 5]
test = df[df["fold"] == 5]

X_train = train.iloc[:, :40].values
y_train = train["label"].values
X_test = test.iloc[:, :40].values
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
clf = RandomForestClassifier(n_estimators=1, warm_start=True, class_weight=class_weights, random_state=39, n_jobs=12)
n_estimators = 100
clf.set_params(n_estimators=1)

# Manual training loop with progress bar
for i in tqdm(range(1, n_estimators + 1)):
    clf.set_params(n_estimators=i)
    clf.fit(X_train, y_train)  # Re-fits all trees with added one

importances = clf.feature_importances_
sorted(zip(feature_names, importances), key=lambda x: -x[1])

# Predict
y_pred = clf.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm / cm.sum() * 100

print(f"\nFrame-level accuracy: {acc:.4f}")
print(f"Frame-level F1 score: {f1:.4f}")
print("Confusion Matrix (in %):")
print(np.round(cm_percent, 2))

# Create DataFrame for easier plotting
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.show()
