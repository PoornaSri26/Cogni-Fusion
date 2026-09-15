"""
Builds the fused feature table from the REAL dataset (1.5 GB CSV of Muse EEG
+ webcam gaze recorded during saccade/smooth-pursuit tasks at two difficulty
levels, 113 participants).

Streams the CSV in chunks, groups rows by (participant, task, session),
cuts each session into 4 s (1024-sample) epochs, and extracts the SAME
features used by the synthetic pipeline:

  EEG  -> eeg_features.extract_eeg_features  (canonical channel order AF7, AF8, TP9, TP10)
  Eye  -> eye_features.extract_eye_features  (gaze speed, saccade rate,
          fixation error to stimulus, dispersion-based fixation ratio)

Label mapping (defensible from the experimental design):
  level-1-smooth    -> low      (easiest: smooth pursuit, low difficulty)
  level-1-saccades  -> medium   (saccadic tracking at low difficulty)
  level-2-*         -> high     (hardest difficulty level, both task types)

Output: data/processed/features_real.parquet  (~10k epochs, 113 subjects)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd

from eeg_features import extract_eeg_features
from eye_features import extract_eye_features

CSV_PATH = "data/raw/dataset.csv"
OUT_PATH = "data/processed/features_real.parquet"

FS_EEG = 256
EPOCH_SEC = 4
SAMPLES_PER_EPOCH = FS_EEG * EPOCH_SEC  # 1024

# CSV column order -> canonical project order
EEG_COLS_CSV = ["EEG_AF7", "EEG_AF8", "EEG_TP9", "EEG_TP10"]
CHANNEL_NAMES = ["AF7", "AF8", "TP9", "TP10"]

LABEL_MAP = {
    "level-1-smooth": "low",
    "level-1-saccades": "medium",
    "level-2-saccades": "high",
    "level-2-smooth": "high",
}

USECOLS = ["Participant_no", "Task", "Session_no", "Timestamp",
           "EEG_AF7", "EEG_AF8", "EEG_TP9", "EEG_TP10",
           "Gaze_x", "Gaze_y", "Stimulus_x", "Stimulus_y"]


def label_for(task: str) -> str:
    return LABEL_MAP.get(str(task).strip().lower(), None)


def epochize_session(df: pd.DataFrame):
    """Split one session's rows into consecutive 4s epochs."""
    n_epochs = len(df) // SAMPLES_PER_EPOCH
    for i in range(n_epochs):
        sl = df.iloc[i * SAMPLES_PER_EPOCH:(i + 1) * SAMPLES_PER_EPOCH]
        yield sl


def features_for_epoch(chunk: pd.DataFrame):
    eeg = chunk[EEG_COLS_CSV].to_numpy(dtype=np.float64).T  # (4, 1024) canonical order
    eeg_feats = extract_eeg_features(eeg, fs=FS_EEG, channel_names=CHANNEL_NAMES)

    eye_df = pd.DataFrame({
        "t": chunk["Timestamp"].to_numpy(float),
        "gaze_x": chunk["Gaze_x"].to_numpy(float),
        "gaze_y": chunk["Gaze_y"].to_numpy(float),
        "stimulus_x": chunk["Stimulus_x"].to_numpy(float),
        "stimulus_y": chunk["Stimulus_y"].to_numpy(float),
    })
    eye_feats = extract_eye_features(eye_df)
    return {**{f"eeg__{k}": v for k, v in eeg_feats.items()},
            **{f"eye__{k}": v for k, v in eye_feats.items()}}


def build(csv_path=CSV_PATH, out_path=OUT_PATH):
    rows, epoch_id = [], 0
    buf, buf_key = [], None

    def flush(key, buf):
        nonlocal epoch_id
        if key is None or not buf:
            return
        sub = pd.concat(buf, ignore_index=True)
        if len(sub) < SAMPLES_PER_EPOCH:
            return
        participant, task, session = key
        label = label_for(task)
        if label is None:
            return
        for chunk in epochize_session(sub):
            try:
                feats = features_for_epoch(chunk)
            except Exception as e:
                print(f"  skip epoch (feature error): {e}")
                continue
            feats.update({
                "epoch_id": epoch_id, "subject": f"P{int(participant):03d}",
                "label": label, "task": task, "session": int(session),
            })
            rows.append(feats)
            epoch_id += 1

    print(f"Streaming {csv_path} ...")
    last_report = 0
    for chunk in pd.read_csv(csv_path, usecols=USECOLS, chunksize=500_000):
        chunk = chunk.dropna(subset=["Participant_no", "Task", "Session_no"])
        keys = list(zip(chunk["Participant_no"].astype(int),
                        chunk["Task"].astype(str),
                        chunk["Session_no"].astype(int)))
        start = 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or keys[i] != keys[start]:
                if keys[start] == buf_key:
                    buf.append(chunk.iloc[start:i])
                    start = i
                    continue
                flush(buf_key, buf)
                buf_key = keys[start]
                buf = [chunk.iloc[start:i]]
                start = i
        if len(rows) - last_report >= 1000:
            print(f"  ... {len(rows)} epochs so far")
            last_report = len(rows)

    flush(buf_key, buf)

    full = pd.DataFrame(rows)
    feat_cols = [c for c in full.columns if c.startswith(("eeg__", "eye__"))]
    full[feat_cols] = full[feat_cols].astype(float)
    n_nan_before = int(full[feat_cols].isna().sum().sum())
    full[feat_cols] = full[feat_cols].fillna(full[feat_cols].median(numeric_only=True))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)

    print(f"\nBuilt real-data feature table: {full.shape} -> {out_path}")
    print(f"NaNs filled: {n_nan_before}")
    print(f"Subjects: {full['subject'].nunique()}")
    print("Class distribution:")
    print(full["label"].value_counts().to_string())
    return full


if __name__ == "__main__":
    build()
