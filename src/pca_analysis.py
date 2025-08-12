import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# keep 95% variance
N_COMPONENTS = 0.95
OUTPUT = "pca_95_full_dataset_1000ms_extended.csv"
df = pd.read_csv("full_dataset_1000ms_extended.csv")

feature_cols = df.columns[:42]
meta_cols = df.columns[42:]

X = df[feature_cols].values
meta = df[meta_cols].copy()

# Scale and PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

# Combine together

df_pca = pd.DataFrame(X_pca,
                      columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
df_out = pd.concat([df_pca,
                    meta.reset_index(drop=True)], axis=1)

df_out.to_csv(OUTPUT, index=False)

print(f"Saved PCA dataset with shape: {df_out.shape}")

