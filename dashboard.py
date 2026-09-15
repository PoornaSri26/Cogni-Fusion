"""
Live cognitive-state dashboard. Polls dashboard_state.json (written by
src/realtime_pipeline.py) and renders:
  1. A big current-state banner (LOW / MEDIUM / HIGH load, or high-load alert)
  2. Live class-probability bars + EEG-trust-weight gauge + confidence metric
  3. Probability-over-time timeline + trust-weight strip chart
  4. Live raw EEG trace of the latest epoch (AF7)
  5. Live gaze scatter (x vs y, latest epoch)
  6. Feature explorer: per-channel EEG band powers + eye-tracking features
  7. Scrolling history table with color-coded predictions
  8. Sidebar: session stats, pipeline heartbeat, model/label info, download

Run alongside the pipeline, in two terminals:
    Terminal 1:  python src/realtime_pipeline.py
    Terminal 2:  streamlit run dashboard.py
"""
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
STATE_PATH = PROJECT_ROOT / "dashboard_state.json"
REFRESH_SEC = 2
PRED_ORDER = ["low", "medium", "high"]
COLOR = {"low": "#2ecc71", "medium": "#f1c40f", "high": "#e74c3c"}
LABEL_DESC = {
    "low": "Relaxed / easy task — alpha-dominant, stable gaze",
    "medium": "Moderate engagement — mixed band activity",
    "high": "High cognitive load — theta/beta rise, alpha drops, gaze locked",
}

st.set_page_config(page_title="Signal Atlas", page_icon="SA", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&display=swap');
:root { --ink: #14212b; --muted: #60717c; --paper: #f6f3ec; --line: #d8d6cf; --teal: #0c8a86; --coral: #e35e43; --gold: #d89120; }
.stApp { background: radial-gradient(circle at 85% 0%, #d9eee8 0, transparent 30%), radial-gradient(circle at 10% 10%, #f6e9d6 0, transparent 27%), var(--paper); color: var(--ink); }
.block-container { max-width: 1420px; padding-top: 1.2rem; padding-bottom: 2.5rem; }
[data-testid="stSidebar"] { background: #172831; }
[data-testid="stSidebar"] * { color: #edf1ed !important; }
[data-testid="stSidebar"] .stDownloadButton button { border-color: #779c99; background: transparent; }
.signal-topline { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding:0 0 .8rem; margin-bottom:1rem; font-family:'DM Mono', monospace; font-size:.72rem; letter-spacing:.12em; color:var(--muted); text-transform:uppercase; }
.signal-mark { color:var(--teal); font-weight:500; }
.atlas-hero { border:1px solid var(--line); border-radius:22px; overflow:hidden; background:rgba(255,255,255,.58); box-shadow:0 18px 55px rgba(24,41,47,.08); margin: .3rem 0 1.3rem; }
.atlas-hero-grid { display:grid; grid-template-columns:1.45fr 1fr; min-height:240px; }
.atlas-main { padding:2rem 2.15rem; position:relative; overflow:hidden; }
.atlas-main:after { content:''; position:absolute; width:360px; height:360px; border:1px solid rgba(12,138,134,.18); border-radius:50%; right:-150px; top:-155px; box-shadow:0 0 0 38px rgba(12,138,134,.045), 0 0 0 76px rgba(12,138,134,.035); }
.atlas-kicker { color:var(--teal); font:500 .73rem 'DM Mono', monospace; letter-spacing:.16em; text-transform:uppercase; }
.atlas-state { font:700 clamp(3.1rem, 7vw, 5.9rem)/.92 'Fraunces', Georgia, serif; letter-spacing:-.065em; margin:.55rem 0 .7rem; color:var(--ink); }
.atlas-copy { max-width:600px; font-size:1.04rem; line-height:1.55; color:#465762; margin:0; }
.atlas-meta { display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:1.35rem; font: .72rem 'DM Mono', monospace; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; }
.atlas-side { background:#172831; color:#eef4f1; padding:1.8rem; display:flex; flex-direction:column; justify-content:space-between; }
.atlas-side-label { font: .7rem 'DM Mono', monospace; letter-spacing:.14em; color:#9fc5bd; text-transform:uppercase; }
.atlas-confidence { font:600 3.25rem/1 'Fraunces', Georgia, serif; margin:.35rem 0; color:#f2d58f; }
.atlas-side-copy { color:#c3d2ce; line-height:1.45; margin:0; }
.atlas-pulse { display:flex; align-items:center; gap:.55rem; font:.7rem 'DM Mono', monospace; color:#9fc5bd; text-transform:uppercase; letter-spacing:.08em; }
.pulse-dot { width:9px; height:9px; border-radius:50%; background:#5ee0b5; box-shadow:0 0 0 5px rgba(94,224,181,.12); }
.metric-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; margin:0 0 1.4rem; }
.metric-note { padding:1rem 1.1rem; border:1px solid var(--line); border-radius:15px; background:rgba(255,255,255,.46); }
.metric-note span { display:block; color:var(--muted); font:.67rem 'DM Mono', monospace; text-transform:uppercase; letter-spacing:.1em; }
.metric-note strong { display:block; font:600 1.35rem 'Fraunces', Georgia, serif; margin-top:.28rem; color:var(--ink); }
.stButton button, .stDownloadButton button { border-radius:9px; }
@media (max-width: 760px) { .atlas-hero-grid { grid-template-columns:1fr; } .atlas-main { padding:1.5rem; } .atlas-side { padding:1.5rem; } .metric-strip { grid-template-columns:1fr; } .signal-topline { gap:.5rem; } }
[data-testid="stToolbar"] { visibility: hidden; }
.section-caption { display:flex; align-items:end; justify-content:space-between; gap:1rem; margin:1.35rem 0 .7rem; }
.section-caption span { color:var(--ink); font:600 1.45rem 'Fraunces', Georgia, serif; letter-spacing:-.03em; }
.section-caption p { max-width:480px; margin:0; color:var(--muted); font-size:.85rem; line-height:1.45; text-align:right; }
.signal-panel { min-height:212px; border:1px solid var(--line); border-radius:17px; padding:1.25rem 1.3rem; background:rgba(255,255,255,.54); }
.signal-panel.dark { background:#21343d; border-color:#21343d; color:#edf3ef; }
.panel-label { color:var(--muted); font:.66rem 'DM Mono', monospace; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.9rem; }
.dark .panel-label { color:#a7c8c1; }
.prob-row { display:grid; grid-template-columns:58px 1fr 42px; align-items:center; gap:.7rem; margin:.64rem 0; font:.72rem 'DM Mono', monospace; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.prob-track { height:8px; background:#e5e5df; border-radius:999px; overflow:hidden; }
.prob-fill { height:100%; border-radius:999px; }
.fusion-number { font:600 2.55rem/1 'Fraunces', Georgia, serif; color:#f3d894; letter-spacing:-.05em; margin:.25rem 0 .45rem; }
.fusion-rail { display:flex; height:12px; overflow:hidden; border-radius:999px; margin:1.1rem 0 .75rem; background:#6b8b87; }
.fusion-eeg { background:#58c1b4; }
.fusion-eye { background:#e7bc6a; }
.fusion-note { color:#c7d6d1; font-size:.88rem; line-height:1.45; margin:0; }
.recovery-note { border-left:3px solid #e3a534; background:#fff2d8; padding:.82rem 1rem; border-radius:0 10px 10px 0; color:#5d481c; font-size:.87rem; line-height:1.45; }
@media (max-width: 760px) { .section-caption { display:block; } .section-caption p { text-align:left; margin-top:.35rem; } }</style>
<div class="signal-topline"><span class="signal-mark">Signal Atlas / Cognitive Telemetry</span><span>EEG + gaze fusion</span></div>
""", unsafe_allow_html=True)


def load_state():
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


state = load_state()
state_age_sec = None
if state:
    try:
        state_age_sec = max(0.0, (datetime.now(timezone.utc) - pd.to_datetime(state["latest"]["timestamp"], utc=True)).total_seconds())
    except (KeyError, TypeError, ValueError):
        state_age_sec = None

if state_age_sec is not None and state_age_sec > REFRESH_SEC * 3:
    st.warning(f"Live pipeline appears paused: latest update is {state_age_sec:.0f}s old.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Session")
    if state:
        hist = state["history"]
        df_all = pd.DataFrame(hist)
        latest = state["latest"]
        pred = latest["prediction"]

        st.metric("Epochs processed", latest["cycle"])
        st.metric("Data freshness", f"{state_age_sec:.0f}s ago" if state_age_sec is not None else "unknown")
        dur = (pd.to_datetime(df_all["timestamp"]).max() - pd.to_datetime(df_all["timestamp"]).min())
        st.metric("Session duration", f"{dur.total_seconds():.0f}s")

        counts = df_all["prediction"].value_counts()
        st.write("**State distribution**")
        for p in PRED_ORDER:
            c = int(counts.get(p, 0))
            st.write(f"<span style='color:{COLOR[p]};font-weight:700'>■</span> {p.title()}: {c}",
                     unsafe_allow_html=True)

        mean_w = float(df_all["eeg_trust_weight"].mean())
        st.metric("Mean EEG trust", f"{mean_w:.2f}",
                  help="Model's average reliance on EEG vs eye-tracking across the session")

        st.divider()
        st.subheader("Download")
        st.download_button("⬇️ Session history (JSON)",
                           data=json.dumps(state, indent=2),
                           file_name="session_history.json",
                           mime="application/json")
        st.download_button("⬇️ Session history (CSV)",
                           data=df_all[["cycle", "timestamp", "prediction", "eeg_trust_weight"]].to_csv(index=False),
                           file_name="session_history.csv",
                           mime="text/csv")
    else:
        st.info("Waiting for the pipeline…")
    st.divider()
    st.caption(f"State file: `{STATE_PATH}`  \nRefresh: every {REFRESH_SEC}s  \n"
               f"Checked: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")

if not state:
    st.warning(
        "No live data yet. Start the pipeline first:\n\n"
        "```bash\npython src/realtime_pipeline.py\n```\n\n"
        "(runs fully on synthetic EEG + synthetic eye data by default — "
        "no webcam or headset required to see the dashboard work)"
    )
    st.stop()

latest = state["latest"]
pred = latest["prediction"]
age_text = f"{state_age_sec:.0f}s ago" if state_age_sec is not None else "awaiting update"
confidence = float(latest.get("confidence", max(latest["probs"].values())))
trust_weight = float(latest["eeg_trust_weight"])
modality = "EEG" if trust_weight >= 0.6 else "gaze" if trust_weight <= 0.4 else "both signals"
freshness = "live" if state_age_sec is not None and state_age_sec <= REFRESH_SEC * 3 else "delayed"
state_story = {
    "low": "The current signal profile is calm and steady. This is a good window for deep, low-pressure work.",
    "medium": "The system sees active engagement without sustained strain. You are in a workable attention zone.",
    "high": "The signal profile suggests elevated cognitive demand. Consider reducing task-switching or taking a short reset.",
}.get(pred, "The system is interpreting the latest multimodal signal window.")

st.markdown(f"""
<div class="atlas-hero">
  <div class="atlas-hero-grid">
    <div class="atlas-main">
      <div class="atlas-kicker">Live cognitive climate / epoch {latest['cycle']:04d}</div>
      <div class="atlas-state">{pred.title()} load</div>
      <p class="atlas-copy">{state_story}</p>
      <div class="atlas-meta"><span>Model confidence {confidence:.0%}</span><span>{modality.title()} is leading</span><span>Feed {freshness}</span></div>
    </div>
    <div class="atlas-side">
      <div><div class="atlas-side-label">Decision certainty</div><div class="atlas-confidence">{confidence:.0%}</div><p class="atlas-side-copy">The fusion model is currently weighting {modality} most heavily for this interpretation.</p></div>
      <div class="atlas-pulse"><span class="pulse-dot"></span> Signal window received</div>
    </div>
  </div>
</div>
<div class="metric-strip">
  <div class="metric-note"><span>EEG contribution</span><strong>{trust_weight:.0%}</strong></div>
  <div class="metric-note"><span>Gaze contribution</span><strong>{1 - trust_weight:.0%}</strong></div>
  <div class="metric-note"><span>Last update</span><strong>{age_text}</strong></div>
</div>
""", unsafe_allow_html=True)

hist_df = pd.DataFrame(hist)
latest = state["latest"]
pred = latest["prediction"]

# ---------------- Signal anatomy ----------------
probs = {label: float(latest["probs"].get(label, 0.0)) for label in PRED_ORDER}
probability_rows = "".join(
    f'<div class="prob-row"><span>{label}</span><div class="prob-track"><div class="prob-fill" style="width:{probs[label] * 100:.1f}%;background:{COLOR[label]}"></div></div><strong>{probs[label]:.0%}</strong></div>'
    for label in PRED_ORDER
)
high_streak = 0
for label in reversed(hist_df["prediction"].tolist()):
    if label != "high":
        break
    high_streak += 1

st.markdown("""
<div class="section-caption">
  <span>Signal anatomy</span>
  <p>Read the model’s current decision without losing the underlying EEG and gaze balance.</p>
</div>
""", unsafe_allow_html=True)

col_probs, col_fusion, col_advice = st.columns([1.12, 1, 1.08])
with col_probs:
    st.markdown(f"""
    <div class="signal-panel">
      <div class="panel-label">State distribution</div>
      {probability_rows}
    </div>
    """, unsafe_allow_html=True)

with col_fusion:
    st.markdown(f"""
    <div class="signal-panel dark">
      <div class="panel-label">Modality balance</div>
      <div class="fusion-number">{trust_weight:.0%} EEG</div>
      <div class="fusion-rail"><div class="fusion-eeg" style="width:{trust_weight * 100:.1f}%"></div><div class="fusion-eye" style="width:{(1 - trust_weight) * 100:.1f}%"></div></div>
      <p class="fusion-note">{modality.title()} is currently carrying the strongest signal contribution.</p>
    </div>
    """, unsafe_allow_html=True)

with col_advice:
    if pred == "high" and high_streak >= 5:
        advice = f"<div class='recovery-note'><strong>Recovery cue.</strong> High load has persisted for {high_streak} windows. A short pause may improve the next focus block.</div>"
    elif pred == "high":
        advice = "<div class='recovery-note'><strong>Intensity rising.</strong> The current window is elevated; keep the next task singular and reduce interruptions.</div>"
    elif pred == "medium":
        advice = "<div class='recovery-note'><strong>Stable engagement.</strong> This is a useful moment for deliberate work. Protect it from unnecessary switching.</div>"
    else:
        advice = "<div class='recovery-note'><strong>Calm baseline.</strong> Use this lower-demand window to plan, read, or recover attention before a demanding block.</div>"
    st.markdown(f"<div class='signal-panel'><div class='panel-label'>Actionable reading</div>{advice}</div>", unsafe_allow_html=True)

# ---------------- Row 2: timelines ----------------
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.subheader("Prediction timeline")
    if len(hist_df) > 1:
        plot_df = pd.DataFrame({
            "cycle": hist_df["cycle"],
            **{p: hist_df["probs"].apply(lambda d, k=p: d.get(k, 0.0)) for p in PRED_ORDER},
        }).set_index("cycle")
        st.line_chart(plot_df, height=220)
        st.caption("Class probabilities over time — watch the state transitions")

with col_t2:
    st.subheader("EEG-trust weight over time")
    if len(hist_df) > 1:
        st.line_chart(hist_df.set_index("cycle")["eeg_trust_weight"], height=220)
        st.caption("How much the fusion model trusted EEG at each epoch — shifts as signal quality/relevance changes")

# ---------------- Row 3: raw EEG + gaze ----------------
col_eeg, col_gaze = st.columns(2)

with col_eeg:
    st.subheader("Live EEG — channel AF7 (latest epoch)")
    trace = latest.get("eeg_trace") or []
    if trace:
        eeg_plot = pd.DataFrame({
            "sample": np.arange(len(trace)) / 256.0,
            "AF7 (µV)": trace,
        }).set_index("sample")
        st.line_chart(eeg_plot, height=200)
        st.caption("First 2 s of the 4 s epoch, 256 Hz — raw signal the features were computed from")
    else:
        st.info("No EEG trace in state file (older pipeline version).")

with col_gaze:
    st.subheader("Live gaze — latest epoch")
    gaze = latest.get("gaze") or {}
    gx, gy = gaze.get("x") or [], gaze.get("y") or []
    if gx and gy and len(gx) == len(gy):
        gaze_df = pd.DataFrame({"gaze_x": gx, "gaze_y": gy})
        st.scatter_chart(gaze_df, x="gaze_x", y="gaze_y", height=200)
        st.caption("Pixel coordinates from the eye tracker — tight cluster = focused fixation, wide scatter = searching")
    else:
        st.info("No gaze data in state file (older pipeline version).")

# ---------------- Row 4: feature explorer ----------------
st.subheader("Latest-epoch features")
feats = latest.get("features") or {}
feat_eeg, feat_eye = feats.get("eeg") or {}, feats.get("eye") or {}

col_fe, col_fy = st.columns(2)
with col_fe:
    st.markdown("**EEG band powers (relative)**")
    if feat_eeg:
        band_df = pd.DataFrame(
            [(ch, float(feat_eeg.get(f"eeg__{ch}_theta_rel", 0)),
              float(feat_eeg.get(f"eeg__{ch}_alpha_rel", 0)),
              float(feat_eeg.get(f"eeg__{ch}_beta_rel", 0)))
             for ch in ["AF7", "AF8", "TP9", "TP10"]],
            columns=["channel", "theta", "alpha", "beta"],
        ).set_index("channel")
        st.dataframe(band_df.style.format("{:.3f}"), width='stretch')
        st.caption("High load → ↑theta, ↑beta, ↓alpha (frontal channels most indicative)")
    else:
        st.info("No EEG features in state file.")

with col_fy:
    st.markdown("**Eye-tracking features**")
    if feat_eye:
        eye_rows = sorted(feat_eye.items())
        eye_df_feat = pd.DataFrame(eye_rows, columns=["feature", "value"])
        eye_df_feat["value"] = eye_df_feat["value"].map(lambda v: f"{v:.4f}")
        st.dataframe(eye_df_feat, width='stretch', hide_index=True)
    else:
        st.info("No eye features in state file.")

# ---------------- Row 5: history table ----------------
st.subheader("Recent history")
show_cols = [c for c in ["cycle", "timestamp", "prediction", "eeg_trust_weight",
                         "confidence"] if c in hist_df.columns]
if "confidence" not in hist_df.columns:
    hist_df["confidence"] = hist_df["probs"].apply(lambda d: max(d.values()) if isinstance(d, dict) else np.nan)
    show_cols = [c for c in ["cycle", "timestamp", "prediction", "eeg_trust_weight", "confidence"]]

tail = hist_df[show_cols].tail(15).iloc[::-1].copy()


def _color_pred(val):
    color = COLOR.get(val, "#888")
    return f"color: {color}; font-weight: 700"


st.dataframe(tail.style.map(_color_pred, subset=["prediction"]).format(
    {"eeg_trust_weight": "{:.2f}", "confidence": "{:.2f}"}),
    width='stretch', hide_index=True)

# ---------------- auto-refresh ----------------
time.sleep(REFRESH_SEC)
st.rerun()









