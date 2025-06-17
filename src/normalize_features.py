from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
df = pd.read_csv("full_dataset_30ms.csv")

print("Unique files per fold:")
print(df.groupby("fold")["file"].nunique())

print("\nFrame count per fold:")
print(df["fold"].value_counts())

print("\nSample files in fold 5:")
print(df[df["fold"] == 5]["file"].unique()[:5])

# Separate features and metadata
feature_cols = df.columns[:39]  # adjust based on actual index count
meta_cols = df.columns[39:]

# Normalize features (use only training folds if doing CV later)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df[feature_cols])

# Combine back with metadata
df_scaled = pd.DataFrame(features_scaled, columns=feature_cols)
df_scaled[meta_cols] = df[meta_cols]
df_scaled.to_csv("full_dataset_30ms_scaled.csv", index=False)
print("Saved normalized dataset.")
