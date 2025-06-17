import librosa
import numpy as np
import pandas as pd
import os

def extract_features_from_file(filepath, sr=16000, win_len_ms=30, step_ms=15, n_mfcc=13):
    y, _ = librosa.load(filepath, sr=sr, mono=True)

    win_length = int(sr * win_len_ms / 1000)
    hop_length = int(sr * step_ms / 1000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=1024, hop_length=hop_length, win_length=win_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    rms = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=win_length)

    # Concatenate: shape = [n_features, n_frames]
    features = np.vstack([mfcc, delta, delta2, rms])
    return features.T  # shape = [n_frames, total_features]

def process_all_files(meta_csv, output_dir, win_len_ms=30, step_ms=15):
    import os
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(meta_csv)

    processed = 0
    failed = []

    for i, row in df.iterrows():
        filepath = row["filepath"]
        speaker_id = row["speaker_id"]
        label = row["condition"]
        fold = row["fold"]
        task = row["task"]

        try:
            features = extract_features_from_file(filepath, win_len_ms=win_len_ms, step_ms=step_ms)
            if features.shape[0] == 0:
                raise ValueError("No frames extracted")

            df_feat = pd.DataFrame(features)
            df_feat["speaker_id"] = speaker_id
            df_feat["label"] = 1 if label == "PT" else 0
            df_feat["fold"] = fold
            df_feat["task"] = task
            df_feat["file"] = os.path.basename(filepath)

            out_name = f"{row['task'].lower()}_{os.path.splitext(os.path.basename(filepath))[0]}.csv"
            out_path = os.path.join(output_dir, out_name)

            df_feat.to_csv(out_path, index=False)
            processed += 1

        except Exception as e:
            failed.append((filepath, str(e)))

    print(f"Processed files: {processed}")
    print(f"Failed files: {len(failed)}")
    for f, reason in failed:
        print(f" - {f}: {reason}")


# Example usage:
process_all_files("androids_metadata_with_folds.csv", "features_30ms", win_len_ms=30, step_ms=15)
