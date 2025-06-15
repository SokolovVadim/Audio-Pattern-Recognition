import pandas as pd
import os
from glob import glob

def merge_feature_files(input_dir):
    all_files = glob(os.path.join(input_dir, "*.csv"))
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df

# Example
df = merge_feature_files("features_30ms")
print(f"Total frames: {len(df)}")
df.to_csv("full_dataset_30ms.csv", index=False)
