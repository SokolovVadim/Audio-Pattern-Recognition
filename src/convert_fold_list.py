import pandas as pd

# Define paths
FOLD_LIST_PATH = "/home/vadim/ComputerScience/APR/Androids-Corpus/fold-lists.csv"
OUTPUT_PATH = "folds_long_format.csv"

# Load original wide-format fold list
df_raw = pd.read_csv(FOLD_LIST_PATH)

# Print column headers to verify
print("🧾 Columns:", list(df_raw.columns))

# Extract reading and interview fold blocks (skip header row)
df_reading = df_raw.iloc[1:, 0:5].copy()
df_interview = df_raw.iloc[1:, 7:12].copy()

# Rename columns for consistency
df_reading.columns = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
df_interview.columns = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

# Convert to long format
def to_long(df, task_name):
    df_long = df.melt(var_name="fold_col", value_name="filename")
    df_long = df_long.dropna()
    df_long["filename"] = df_long["filename"].str.strip().str.replace("'", "")
    df_long["task"] = task_name
    df_long["fold"] = df_long["fold_col"].str.extract(r'fold(\d)').astype(int)
    df_long["session_id"] = df_long["filename"]
    df_long["speaker_id"] = df_long["filename"].str.extract(r"(\d+_[A-Z]{2}\d{2})")
    return df_long[["session_id", "speaker_id", "fold", "task"]]

df_reading_long = to_long(df_reading, "Reading")
df_interview_long = to_long(df_interview, "Interview")

# After merging df_reading_long and df_interview_long:
df_folds = pd.concat([df_reading_long, df_interview_long], ignore_index=True)

# Extract speaker ID from session_id (first part of ID)
df_folds["speaker_id"] = df_folds["session_id"].str.extract(r"(\d+_[A-Z]{2}\d{2})")

# Keep only one fold assignment per speaker per task
df_folds = df_folds.drop_duplicates(subset=["speaker_id", "task"])


# Final check
print("\n✅ Cleaned fold list (one fold per speaker):")
print(df_folds.groupby(["task", "fold"]).size())

# Save
df_folds.to_csv(OUTPUT_PATH, index=False)

# Check that each speaker is assigned to only one fold per task
speaker_fold_counts = df_folds.groupby(["speaker_id", "task"])["fold"].nunique()
violations = speaker_fold_counts[speaker_fold_counts > 1]
if not violations.empty:
    print("\n❌ Violation: Some speakers appear in multiple folds per task!")
    print(violations)
else:
    print("\n✅ All speakers assigned to one fold per task.")


