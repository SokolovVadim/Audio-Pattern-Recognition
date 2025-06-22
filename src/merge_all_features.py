import pandas as pd
import os
from glob import glob

def merge_feature_files(input_dir):
    all_files = glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(all_files)} feature files in '{input_dir}'")

    df_list = []
    missing_columns = {"fold": 0, "label": 0, "file": 0}
    failed_files = []

    for file in all_files:
        try:
            df = pd.read_csv(file)
            for col in missing_columns:
                if col not in df.columns:
                    print(f"Missing column '{col}' in: {file}")
                    missing_columns[col] += 1
            df_list.append(df)
        except Exception as e:
            failed_files.append((file, str(e)))

    print("\nSummary:")
    for col, count in missing_columns.items():
        print(f" - Files missing '{col}': {count}")
    print(f" - Failed to load: {len(failed_files)}")

    if failed_files:
        for f, reason in failed_files:
            print(f"   {f}: {reason}")

    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df

if __name__ == "__main__":
    input_dir = "features_30000ms"
    output_csv = f"full_dataset_30000ms.csv"

    df = merge_feature_files(input_dir)

    print("\nStats:")
    print("Frame count per fold:")
    print(df["fold"].value_counts())

    print("\nUnique files per fold:")
    print(df.groupby("fold")["file"].nunique())

    print(f"\nTotal frames: {len(df)}")

    df.sort_values(by=["fold", "task", "file"], inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")
