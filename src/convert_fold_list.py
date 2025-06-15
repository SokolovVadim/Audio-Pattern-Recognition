import pandas as pd
import os

def convert_fold_list(wide_csv_path):
    df_raw = pd.read_csv(wide_csv_path, skiprows=1)  # skip "Read,,,Interview" row
    fold_long = []

    # Reading folds: columns 0–4
    for col in df_raw.columns[:5]:
        fold_num = int(col[-1])
        for val in df_raw[col].dropna():
            filename = val.strip("'")  # remove quotes
            fold_long.append({
                "filename": filename,
                "fold": fold_num,
                "task": "Reading"
            })

    # Interview folds: columns 7–11 (index 7 to 11)
    for col in df_raw.columns[7:12]:
        fold_num = int(col[-1])
        for val in df_raw[col].dropna():
            filename = val.strip("'")
            fold_long.append({
                "filename": filename,
                "fold": fold_num,
                "task": "Interview"
            })

    return pd.DataFrame(fold_long)

def merge_dataframes(df_meta, df_folds):

    print(f"✅ Loaded metadata: {len(df_meta)} rows")
    print(f"✅ Loaded fold list: {len(df_folds)} rows")

    df_meta["filename"] = df_meta["filepath"].apply(lambda x: os.path.basename(x).strip())

    df_meta["filename_no_ext"] = df_meta["filename"].apply(lambda x: os.path.splitext(x)[0])
    df_folds["filename"] = df_folds["filename"].apply(lambda x: x.strip().strip("'"))

    # Now merge using filename without extension
    df_merged = df_meta.merge(df_folds, left_on=["filename_no_ext", "task"], right_on=["filename", "task"], how="left")


    # Print a few examples
    print("\n📂 Sample metadata filenames:")
    print(df_meta["filename"].unique()[:5])

    print("\n📂 Sample fold-list filenames:")
    print(df_folds["filename"].unique()[:5])

    print("\n🔍 Unique tasks in metadata:", df_meta["task"].unique())
    print("🔍 Unique tasks in folds:", df_folds["task"].unique())

    print(f"\n✅ Merge complete: {len(df_merged)} rows")

    # Save
    df_merged.to_csv("androids_metadata_with_folds.csv", index=False)
    print("📁 Saved merged file: androids_metadata_with_folds.csv")

    df_merged.drop(columns=["filename_no_ext", "filename_y"], inplace=True)
    df_merged.rename(columns={"filename_x": "filename"}, inplace=True)


    # Check for missing fold values
    missing = df_merged[df_merged["fold"].isna()]
    print(f"\n❗ Files missing fold assignments: {len(missing)}")

    if not missing.empty:
        print("\n🚫 Example missing entries:")
        print(missing[["filename", "task"]].head())

# Example usage
df_folds = convert_fold_list(os.path.join("/home/vadim/ComputerScience/APR/Androids-Corpus", "fold-lists.csv"))
df_folds.to_csv("folds_long_format.csv", index=False)
print(df_folds.head())
df_meta = pd.read_csv("androids_metadata.csv")
merge_dataframes(df_meta, df_folds)
