import os
import pandas as pd

# === Config ===
METADATA_PATH = "metadata_combined.csv"
FOLD_LIST_PATH = "folds_long_format.csv"
OUTPUT_PATH = "androids_metadata_with_folds.csv"

# === Load metadata ===
df_meta = pd.read_csv(METADATA_PATH)
print(f"\n✅ Loaded metadata: {len(df_meta)} rows")
print("🧾 Metadata columns:", df_meta.columns.tolist())
print("🔍 Unique tasks:", df_meta['task'].unique().tolist())

# Extract session_id from filepath (basename without .wav)
df_meta["session_id"] = df_meta["filepath"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])

# Extract speaker_id from session_id
df_meta["speaker_id"] = df_meta["session_id"].str.extract(r"(\d+_[A-Z]{2}\d{2})")

print("\n🔁 Sample speaker/session IDs:")
print(df_meta[["filepath", "session_id", "speaker_id"]].head())

# === Load fold list ===
df_folds = pd.read_csv(FOLD_LIST_PATH)
print(f"\n✅ Loaded folds list: {len(df_folds)} rows")
print("🧾 Fold list columns:", df_folds.columns.tolist())

# === Merge by speaker_id and task ===
df_merged = pd.merge(df_meta, df_folds, on=["speaker_id", "task"], how="left")


# === Check for issues ===
missing = df_merged["fold"].isna().sum()
print(f"\n❗ Missing fold assignments: {missing}")
if missing > 0:
    print("🔍 Sample of missing rows:")
    print(df_merged[df_merged["fold"].isna()][["filepath", "speaker_id", "task"]].head(10))

# === Save result ===
df_merged.to_csv(OUTPUT_PATH, index=False)
print(f"\n📁 Saved merged metadata with folds: {OUTPUT_PATH}")
