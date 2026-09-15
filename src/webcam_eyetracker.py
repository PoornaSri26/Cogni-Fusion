"""
Webcam eye-tracking using MediaPipe FaceMesh (iris refinement landmarks).
Produces the SAME schema as eye_features.py expects (t, pupil_mm, is_fixation,
is_blink, gaze_x, gaze_y) so it's a drop-in replacement for a Tobii CSV.

Requires a real webcam to actually run live -- this file is written and
correct, but a sandboxed environment without a camera device can't execute
the live loop end-to-end. Test it on your own machine with:
    python src/webcam_eyetracker.py --seconds 10

Landmark reference (MediaPipe FaceMesh with refine_landmarks=True):
  - Left iris: 468-472   Right iris: 473-477
  - Left eye corners: 33, 133      Right eye corners: 362, 263
  - Left eye top/bottom (for EAR/blink): 159, 145
  - Right eye top/bottom (for EAR/blink): 386, 374
"""
import time
import argparse
from collections import deque

import numpy as np
import pandas as pd

try:
    import cv2
    import mediapipe as mp
except ImportError:
    cv2 = None
    mp = None

LEFT_IRIS = list(range(468, 473))
RIGHT_IRIS = list(range(473, 478))
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_VERT = (159, 145)   # top, bottom
RIGHT_EYE_VERT = (386, 374)
EAR_BLINK_THRESHOLD = 0.18   # eye-aspect-ratio below this = eyes closed (calibrate per-user if possible)
FIXATION_DISPERSION_PX = 15  # gaze considered "fixating" if it moves less than this within a short window


def _landmark_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def _eye_aspect_ratio(landmarks, corners, vert, w, h):
    p_left = _landmark_xy(landmarks, corners[0], w, h)
    p_right = _landmark_xy(landmarks, corners[1], w, h)
    p_top = _landmark_xy(landmarks, vert[0], w, h)
    p_bottom = _landmark_xy(landmarks, vert[1], w, h)
    horiz = np.linalg.norm(p_left - p_right) + 1e-6
    vert_dist = np.linalg.norm(p_top - p_bottom)
    return vert_dist / horiz


def _iris_diameter_px(landmarks, iris_idx, w, h):
    pts = np.array([_landmark_xy(landmarks, i, w, h) for i in iris_idx])
    # iris landmark 0 is the center, 1-4 are the boundary -> diameter ~ 2 * mean radius
    center, boundary = pts[0], pts[1:]
    radius = np.linalg.norm(boundary - center, axis=1).mean()
    return 2 * radius


class WebcamEyeTracker:
    """Streams (t, pupil_mm-proxy, is_fixation, is_blink, gaze_x, gaze_y) rows
    from a live webcam, ready to feed straight into eye_features.extract_eye_features().
    Pupil diameter is reported in pixels-normalized-to-eye-width as a proxy for mm,
    since true mm requires a calibrated camera-to-eye distance."""

    def __init__(self, camera_index=0, fixation_window=5):
        if mp is None:
            raise ImportError("Install mediapipe and opencv-python-headless first: "
                               "pip install mediapipe opencv-python-headless")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.cap = cv2.VideoCapture(camera_index)
        self.recent_gaze = deque(maxlen=fixation_window)

    def _process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark

        ear_l = _eye_aspect_ratio(landmarks, LEFT_EYE_CORNERS, LEFT_EYE_VERT, w, h)
        ear_r = _eye_aspect_ratio(landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE_VERT, w, h)
        is_blink = (ear_l + ear_r) / 2 < EAR_BLINK_THRESHOLD

        iris_l = _iris_diameter_px(landmarks, LEFT_IRIS, w, h)
        iris_r = _iris_diameter_px(landmarks, RIGHT_IRIS, w, h)
        eye_width_l = np.linalg.norm(
            _landmark_xy(landmarks, LEFT_EYE_CORNERS[0], w, h)
            - _landmark_xy(landmarks, LEFT_EYE_CORNERS[1], w, h)) + 1e-6
        pupil_proxy_mm = ((iris_l + iris_r) / 2) / eye_width_l * 11.7  # 11.7mm ~ avg horiz. iris diameter, used to scale proxy to mm-like units

        gaze_center = (_landmark_xy(landmarks, LEFT_IRIS[0], w, h)
                       + _landmark_xy(landmarks, RIGHT_IRIS[0], w, h)) / 2
        self.recent_gaze.append(gaze_center)
        dispersion = (np.std([p[0] for p in self.recent_gaze])
                      + np.std([p[1] for p in self.recent_gaze])) if len(self.recent_gaze) > 1 else 0
        is_fixation = dispersion < FIXATION_DISPERSION_PX

        return {
            "pupil_mm": pupil_proxy_mm, "is_fixation": is_fixation, "is_blink": is_blink,
            "gaze_x": float(gaze_center[0]), "gaze_y": float(gaze_center[1]),
        }

    def stream_epoch(self, epoch_sec=4):
        """Blocks for epoch_sec seconds, returns a DataFrame matching the schema
        expected by src/eye_features.py."""
        rows, t0 = [], time.time()
        while time.time() - t0 < epoch_sec:
            ok, frame = self.cap.read()
            if not ok:
                continue
            feats = self._process_frame(frame)
            if feats is not None:
                feats["t"] = time.time() - t0
                rows.append(feats)
        return pd.DataFrame(rows) if rows else None

    def close(self):
        self.cap.release()
        self.face_mesh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=4)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    tracker = WebcamEyeTracker(camera_index=args.camera)
    print(f"Capturing {args.seconds}s from webcam...")
    df = tracker.stream_epoch(epoch_sec=args.seconds)
    tracker.close()
    if df is None or df.empty:
        print("No face detected -- check camera/lighting.")
    else:
        print(df.describe())
