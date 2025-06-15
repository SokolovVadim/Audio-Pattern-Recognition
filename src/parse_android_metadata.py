import os
import re
import pandas as pd

def parse_filename(filename):
    """
    Example: 27_PM33_3.wav
    Meaning: speaker 27, P=Patient, M=Male, age=33, education=3
    """
    match = re.match(r"(\d+)_([PCX])([MFX])(\d{2})_(\d)\.wav", filename)
    if not match:
        return None
    speaker_num, condition, gender, age, edu_level = match.groups()
    speaker_id = f"{int(speaker_num)}_{condition}"
    return {
        "speaker_id": speaker_id,
        "condition": "PT" if condition == "P" else "HC",
        "gender": gender,
        "age": int(age),
        "education_level": int(edu_level),
    }

def collect_metadata(base_path):
    records = []

    # Reading-Task
    for group in ["HC", "PT"]:
        group_path = os.path.join(base_path, "Reading-Task", "audio", group)
        for fname in os.listdir(group_path):
            if not fname.endswith(".wav"):
                continue
            info = parse_filename(fname)
            if info:
                info.update({
                    "task": "Reading",
                    "group": group,
                    "filepath": os.path.join(group_path, fname)
                })
                records.append(info)

    # Interview-Task/audio
    for group in ["HC", "PT"]:
        group_path = os.path.join(base_path, "Interview-Task", "audio", group)
        for fname in os.listdir(group_path):
            if not fname.endswith(".wav"):
                continue
            info = parse_filename(fname)
            if info:
                info.update({
                    "task": "Interview",
                    "group": group,
                    "filepath": os.path.join(group_path, fname)
                })
                records.append(info)

    return pd.DataFrame(records)

# Example usage
if __name__ == "__main__":
    base_path = "/home/vadim/ComputerScience/APR/Androids-Corpus"
    df = collect_metadata(base_path)
    print(df.head())
    df.to_csv("androids_metadata.csv", index=False)
