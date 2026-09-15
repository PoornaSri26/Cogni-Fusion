"""
Generates synthetic EEG + eye-tracking session data that mimics the
statistical structure reported in the literature (theta/alpha/beta shifts
under load; fixation/blink/pupil shifts under load), so the rest of the
pipeline can be built, tested and demoed WITHOUT needing lab hardware.

Swap-in points for real data:
  - EEG:  STEW dataset (Kaggle) or SEED-VIG (BCMI lab) -> replace load_eeg_epochs()
  - Eye:  Any Tobii/Pupil-Labs export, or webcam+MediaPipe features -> replace load_eye_epochs()
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
FS_EEG = 256
EPOCH_SEC = 4
N_SUBJECTS = 20
N_EPOCHS_PER_CLASS = 60
CLASSES = ["low", "medium", "high"]
CHANNELS = ["AF7", "AF8", "TP9", "TP10"]


def _simulate_eeg_epoch(label):
    t = np.arange(0, EPOCH_SEC, 1 / FS_EEG)
    load_factor = {"low": 0.0, "medium": 0.5, "high": 1.0}[label]
    sig = np.zeros((len(CHANNELS), len(t)))
    for ch in range(len(CHANNELS)):
        alpha = (1.2 - 0.7 * load_factor) * np.sin(2 * np.pi * 10 * t)
        theta = (0.4 + 0.9 * load_factor) * np.sin(2 * np.pi * 6 * t)
        beta = (0.3 + 0.6 * load_factor) * np.sin(2 * np.pi * 22 * t)
        noise = RNG.normal(0, 0.6, len(t))
        sig[ch] = alpha + theta + beta + noise
    return sig


def _simulate_eye_epoch(label):
    fs_eye = 60
    n = EPOCH_SEC * fs_eye
    load_factor = {"low": 0.0, "medium": 0.5, "high": 1.0}[label]
    pupil = 3.0 + 1.0 * load_factor + RNG.normal(0, 0.15, n)
    fixation_flags = RNG.random(n) < (0.55 + 0.25 * load_factor)
    blink_flags = RNG.random(n) < max(0.01, 0.05 - 0.03 * load_factor)
    gaze_x = np.cumsum(RNG.normal(0, 0.5 + 1.5 * (1 - load_factor), n))
    gaze_y = np.cumsum(RNG.normal(0, 0.5 + 1.5 * (1 - load_factor), n))
    return pd.DataFrame({
        "t": np.arange(n) / fs_eye, "pupil_mm": pupil, "is_fixation": fixation_flags,
        "is_blink": blink_flags, "gaze_x": gaze_x, "gaze_y": gaze_y,
    })


def generate_dataset(out_dir="data/raw"):
    out_dir = Path(out_dir)
    (out_dir / "eeg").mkdir(parents=True, exist_ok=True)
    (out_dir / "eye").mkdir(parents=True, exist_ok=True)
    rows, idx = [], 0
    for subj in range(N_SUBJECTS):
        for label in CLASSES:
            for _ in range(N_EPOCHS_PER_CLASS // N_SUBJECTS + 1):
                eeg = _simulate_eeg_epoch(label)
                eye = _simulate_eye_epoch(label)
                eeg_path = out_dir / "eeg" / f"epoch_{idx:05d}.npy"
                eye_path = out_dir / "eye" / f"epoch_{idx:05d}.csv"
                np.save(eeg_path, eeg)
                eye.to_csv(eye_path, index=False)
                rows.append({"epoch_id": idx, "subject": f"S{subj:02d}", "label": label,
                             "eeg_path": str(eeg_path), "eye_path": str(eye_path)})
                idx += 1
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Generated {len(manifest)} synchronized EEG+eye epochs -> {out_dir}")
    return manifest


if __name__ == "__main__":
    generate_dataset()
