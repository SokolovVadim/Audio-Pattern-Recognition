import librosa
import numpy as np
import pandas as pd
import os

def safe_delta(feat, order=1, width=9):
    """
    Compute delta features safely, skipping if not enough frames.
    """
    if feat.shape[1] < width:
        raise ValueError(f"Too few frames ({feat.shape[1]}) for delta computation with width={width}")
    return librosa.feature.delta(feat, order=order, width=width)

def extract_features_from_file(filepath, sr=16000, win_len_ms=30, step_ms=15, n_mfcc=13):
    y, _ = librosa.load(filepath, sr=sr, mono=True)

    win_length = int(sr * win_len_ms / 1000)
    hop_length = int(sr * step_ms / 1000)

    if len(y) < win_length:
        raise ValueError(f"Audio too short ({len(y)} samples) for win_length={win_length}")

    n_fft = min(2 ** int(np.ceil(np.log2(win_length))), len(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Pad short mfccs
    min_frames = 9
    if mfcc.shape[1] < min_frames:
        pad_width = min_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='edge')

    delta = safe_delta(mfcc, order=1)
    delta2 = safe_delta(mfcc, order=2)
    rms = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=win_length)

    # Pad RMS to match MFCC frames
    target_frames = mfcc.shape[1]
    if rms.shape[1] < target_frames:
        rms = np.pad(rms, ((0, 0), (0, target_frames - rms.shape[1])), mode='edge')

    # Sanity check before stacking
    assert mfcc.shape[1] == delta.shape[1] == delta2.shape[1] == rms.shape[1]

    f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr, frame_length=n_fft, hop_length=hop_length)
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = librosa.feature.rms(y=harmonic, frame_length=win_length, hop_length=hop_length)
    total_energy = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)
    harmonic_ratio = harmonic_energy / (total_energy + 1e-6)

    # Match frame length
    min_len = min(mfcc.shape[1], f0.shape[0], harmonic_ratio.shape[1])
    mfcc = mfcc[:, :min_len]
    delta = delta[:, :min_len]
    delta2 = delta2[:, :min_len]
    rms = rms[:, :min_len]
    f0 = f0[:min_len]
    harmonic_ratio = harmonic_ratio[:, :min_len]


    features = np.vstack([mfcc, delta, delta2, rms, f0[np.newaxis, :], harmonic_ratio])
    return features.T


def process_all_files(meta_csv, output_dir, win_len_ms=30, step_ms=15):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(meta_csv)

    processed, failed = 0, []

    for _, row in df.iterrows():
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
            df_feat["label"] = 1 if label == "P" else 0
            df_feat["fold"] = fold
            df_feat["task"] = task
            df_feat["file"] = os.path.basename(filepath)

            out_name = f"{task.lower()}_{os.path.splitext(os.path.basename(filepath))[0]}.csv"
            out_path = os.path.join(output_dir, out_name)
            df_feat.to_csv(out_path, index=False)
            processed += 1

        except Exception as e:
            failed.append((filepath, str(e)))

    print(f"\nProcessed: {processed} files")
    print(f"Failed: {len(failed)} files")
    for f, reason in failed:
        print(f"  - {f}: {reason}")

# Example usage
if __name__ == "__main__":
    process_all_files(
        meta_csv="androids_metadata_with_folds.csv",
        output_dir="features_5000ms_extended",
        win_len_ms=5000,
        step_ms=2500
    )
