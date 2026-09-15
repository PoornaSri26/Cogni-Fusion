"""Joins manifest + EEG features + eye features into one fused feature table."""
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from eeg_features import extract_batch as extract_eeg_batch
from eye_features import extract_batch as extract_eye_batch

CHANNEL_NAMES = ["AF7", "AF8", "TP9", "TP10"]


def build(manifest_path="data/raw/manifest.csv", out_path="data/processed/features.parquet"):
    manifest = pd.read_csv(manifest_path)
    eeg_feats = extract_eeg_batch(manifest["eeg_path"].tolist(), channel_names=CHANNEL_NAMES)
    eeg_feats.columns = [f"eeg__{c}" for c in eeg_feats.columns]
    eye_feats = extract_eye_batch(manifest["eye_path"].tolist())
    eye_feats.columns = [f"eye__{c}" for c in eye_feats.columns]
    full = pd.concat([manifest[["epoch_id", "subject", "label"]].reset_index(drop=True),
                       eeg_feats.reset_index(drop=True),
                       eye_feats.reset_index(drop=True)], axis=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    print(f"Built fused feature table: {full.shape} -> {out_path}")
    return full


if __name__ == "__main__":
    build()
