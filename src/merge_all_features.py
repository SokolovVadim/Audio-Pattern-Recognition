import pandas as pd
import os
from glob import glob

def merge_feature_files(input_dir):
    all_files = glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(all_files)} feature files.")

    df_list = []
    missing_fold_count = 0
    for file in all_files:
        df = pd.read_csv(file)
        if "fold" not in df.columns:
            print(f"Missing 'fold' in: {file}")
            missing_fold_count += 1
        df_list.append(df)

    print(f"Files missing 'fold': {missing_fold_count}")
    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df

# Example
df = merge_feature_files("features_30ms")
print("Frame count per fold:")
print(df["fold"].value_counts())
print("Unique files per fold:")
print(df.groupby("fold")["file"].nunique())

print(f"Total frames: {len(df)}")
df.to_csv("full_dataset_30ms.csv", index=False)
