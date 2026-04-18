"""
app.py - Streamlit ASL Hand Sign Recognition
=============================================
Real-time ASL detection using MediaPipe hand landmarks + RandomForest.
Run with:  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
from pathlib import Path
import threading
import queue

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ASL Hand Sign Recognizer",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(20, 15, 60, 0.85);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #e0e0ff !important;
}

/* Cards */
.pred-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 16px;
    text-align: center;
    transition: all 0.3s ease;
}

.pred-card:hover {
    background: rgba(255,255,255,0.10);
    border-color: rgba(120,80,255,0.5);
    transform: translateY(-2px);
}

/* Big letter */
.big-letter {
    font-size: 120px;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, #a78bfa, #f472b6, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    text-align: center;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.7; }
}

/* Confidence bar */
.conf-bar-wrap {
    background: rgba(255,255,255,0.1);
    border-radius: 99px;
    height: 10px;
    margin: 8px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #7c3aed, #db2777);
    transition: width 0.4s ease;
}

/* Top-k badge */
.topk-badge {
    display: inline-block;
    background: rgba(124,58,237,0.25);
    border: 1px solid rgba(124,58,237,0.5);
    border-radius: 8px;
    padding: 4px 12px;
    margin: 4px;
    font-size: 14px;
    font-weight: 600;
    color: #c4b5fd;
}

/* Status pill */
.status-pill {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 99px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.status-active {
    background: rgba(52,211,153,0.15);
    border: 1px solid #34d399;
    color: #34d399;
}
.status-inactive {
    background: rgba(239,68,68,0.15);
    border: 1px solid #ef4444;
    color: #ef4444;
}

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 8px;
}

/* Sentence box */
.sentence-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 4px;
    color: #e2e8f0;
    min-height: 72px;
    word-break: break-all;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "asl_venv" / "landmark_classifier2.pkl"
CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

# ──────────────────────────────────────────────
# Load model (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

def get_mediapipe():
    """Create a fresh MediaPipe Hands detector each run (NOT cached)."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    return mp_hands, hands, mp.solutions.drawing_utils

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def normalize_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    bx, by, bz = coords[0], coords[1], coords[2]
    for i in range(0, len(coords), 3):
        coords[i]   -= bx
        coords[i+1] -= by
        coords[i+2] -= bz
    max_val = max(abs(c) for c in coords)
    if max_val > 0:
        coords = [c / max_val for c in coords]
    return coords


def get_bbox(hand_landmarks, img_w, img_h, pad=30):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return (
        max(0,     int(min(xs)*img_w) - pad),
        max(0,     int(min(ys)*img_h) - pad),
        min(img_w, int(max(xs)*img_w) + pad),
        min(img_h, int(max(ys)*img_h) + pad),
    )


def confidence_bar_html(label, conf, color="#7c3aed"):
    pct = int(conf * 100)
    return f'<div style="margin:6px 0;"><div style="display:flex;justify-content:space-between;font-size:13px;color:#c4b5fd;margin-bottom:4px;"><span style="font-weight:700;">{label}</span><span>{pct}%</span></div><div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},#db2777);"></div></div></div>'


# ──────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────
def init_state():
    defaults = {
        "running": False,
        "sentence": "",
        "last_letter": "",
        "hold_frames": 0,
        "ema": None,
        "pred_label": "—",
        "pred_conf": 0.0,
        "topk": [],
        "fps": 0.0,
        "hand_detected": False,
        "smoothing": 3,
        "min_conf": 0.60,
        "flip": False,
        "topk_n": 3,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤟 ASL Recognizer")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    st.session_state.flip       = st.toggle("↔️ Flip Camera",      value=st.session_state.flip)
    st.session_state.smoothing  = st.slider("🌊 EMA Smoothing",    1, 10, st.session_state.smoothing)
    st.session_state.min_conf   = st.slider("🎯 Min Confidence",   0.1, 1.0, st.session_state.min_conf, 0.05)
    st.session_state.topk_n     = st.slider("🏆 Top-K Predictions", 1, 5, st.session_state.topk_n)

    st.markdown("---")
    st.markdown("### 📖 ASL Alphabet")
    st.markdown(
        " ".join([f"`{c}`" for c in CLASSES]),
        help="Supported signs"
    )
    st.markdown("---")
    st.markdown("### 🖼️ ASL Reference")
    ref_dir = Path(r"C:\Users\harsh\Downloads\archive\asl_alphabet_test\asl_alphabet_test")
    if ref_dir.exists():
        all_imgs = list(ref_dir.glob("*.jpg"))
        # Sort: A-Z first, then del, nothing, space
        alpha_imgs = sorted([p for p in all_imgs if len(p.stem.replace('_test','')) == 1],
                            key=lambda p: p.stem.replace('_test',''))
        other_imgs = sorted([p for p in all_imgs if len(p.stem.replace('_test','')) > 1],
                            key=lambda p: p.stem.replace('_test',''))
        ref_images = alpha_imgs + other_imgs
        COLS_PER_ROW = 5
        for i in range(0, len(ref_images), COLS_PER_ROW):
            row_imgs = ref_images[i:i + COLS_PER_ROW]
            cols = st.columns(COLS_PER_ROW)
            for j, img_path in enumerate(row_imgs):
                label = img_path.stem.replace("_test", "")
                cols[j].image(str(img_path), caption=label, use_container_width=True)
    else:
        st.warning("Reference images folder not found.")
    st.markdown("---")
    chart_path = Path(__file__).parent / "asl_venv" / "istockphoto-1306688274-612x612.jpg"
    if chart_path.exists():
        st.image(str(chart_path), caption="ASL Alphabet Chart", use_container_width=True)
    st.markdown("---")
    st.caption("Built with MediaPipe + RandomForest · Streamlit")

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 8px 0;">
  <h1 style="font-size:2.8rem;font-weight:800;
             background:linear-gradient(135deg,#a78bfa,#f472b6,#60a5fa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;margin:0;">
    🤟 ASL Hand Sign Recognizer
  </h1>
  <p style="color:#94a3b8;font-size:1rem;margin-top:6px;">
    Real-time American Sign Language detection via MediaPipe landmarks
  </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LAYOUT: video | predictions | sentence
# ──────────────────────────────────────────────
col_video, col_pred = st.columns([3, 2], gap="large")

with col_video:
    st.markdown('<div class="section-header">📷 Live Feed</div>', unsafe_allow_html=True)

    # Start/Stop button
    btn_label = "⏹ Stop Detection" if st.session_state.running else "▶ Start Detection"
    btn_color = "secondary" if st.session_state.running else "primary"
    if st.button(btn_label, use_container_width=True, type=btn_color):
        st.session_state.running = not st.session_state.running
        st.session_state.ema = None
        st.rerun()

    video_placeholder = st.empty()
    fps_placeholder   = st.empty()

with col_pred:
    st.markdown('<div class="section-header">🔮 Prediction</div>', unsafe_allow_html=True)

    pred_placeholder  = st.empty()
    topk_placeholder  = st.empty()
    hand_placeholder  = st.empty()

    st.markdown("---")
    st.markdown('<div class="section-header">📝 Sentence Builder</div>', unsafe_allow_html=True)

    sentence_ph = st.empty()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⌫ Delete Last", use_container_width=True):
            st.session_state.sentence = st.session_state.sentence[:-1]
            st.rerun()
    with c2:
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.sentence = ""
            st.session_state.last_letter = ""
            st.rerun()

    st.info("Hold a sign steady for ~1 second to append it to the sentence. 'space' adds a space, 'del' removes the last character.")

# Always render sentence
sentence_ph.markdown(
    f'<div class="sentence-box">{st.session_state.sentence or "&nbsp;"}</div>',
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────
# Default (stopped) prediction panel
# ──────────────────────────────────────────────
def render_idle():
    pred_placeholder.markdown("""
    <div class="pred-card">
      <span class="big-letter">?</span>
      <p style="color:#94a3b8;margin-top:12px;">Start detection to begin</p>
    </div>""", unsafe_allow_html=True)
    hand_placeholder.markdown(
        '<div style="text-align:center;margin-top:8px;">'
        '<span class="status-pill status-inactive">● Camera Off</span></div>',
        unsafe_allow_html=True
    )

if not st.session_state.running:
    render_idle()
    video_placeholder.markdown("""
    <div style="background:rgba(0,0,0,0.3);border:1px dashed rgba(255,255,255,0.2);
                border-radius:12px;height:380px;display:flex;align-items:center;
                justify-content:center;color:#94a3b8;font-size:1.2rem;">
      📷 Press <strong>&nbsp;▶ Start Detection&nbsp;</strong> to enable camera
    </div>""", unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────
# LIVE DETECTION LOOP
# ──────────────────────────────────────────────
with st.spinner("Loading model & camera..."):
    clf = load_model()
    mp_hands_m, hands_detector, mp_draw = get_mediapipe()

cap = None
try:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        st.error("❌ Could not open camera. Check your webcam.")
        st.session_state.running = False
        st.stop()

    HOLD_FRAMES_NEEDED = 20   # ~1 sec at ~20fps before appending letter
    t_prev = time.time()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Could not read from camera. Check your webcam.")
            break

        if st.session_state.flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        pred_label  = "nothing"
        pred_conf   = 0.0
        topk_preds  = []
        hand_found  = False

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            hand_found = True

            features = np.array(normalize_landmarks(lm)).reshape(1, -1)
            probs    = clf.predict_proba(features)[0]

            # EMA smoothing
            alpha = 1.0 / max(st.session_state.smoothing, 1)
            if st.session_state.ema is None:
                st.session_state.ema = probs
            else:
                st.session_state.ema = (1 - alpha) * st.session_state.ema + alpha * probs

            ema = st.session_state.ema
            top_idx     = int(np.argmax(ema))
            pred_label  = CLASSES[top_idx]
            pred_conf   = float(ema[top_idx])

            # Top-K
            topk_idx   = np.argsort(ema)[-st.session_state.topk_n:][::-1]
            topk_preds = [(CLASSES[i], float(ema[i])) for i in topk_idx]

            # Draw on frame
            mp_draw.draw_landmarks(frame, lm, mp_hands_m.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(180,0,255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2))

            img_h, img_w = frame.shape[:2]
            x1, y1, x2, y2 = get_bbox(lm, img_w, img_h)
            color = (0, 230, 128) if pred_conf >= st.session_state.min_conf else (255, 100, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_str = f"{pred_label}  {pred_conf*100:.0f}%"
            cv2.putText(frame, label_str, (x1, max(30, y1-12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

            # Sentence builder logic
            if pred_conf >= st.session_state.min_conf:
                if pred_label == st.session_state.last_letter:
                    st.session_state.hold_frames += 1
                else:
                    st.session_state.last_letter  = pred_label
                    st.session_state.hold_frames  = 0

                if st.session_state.hold_frames == HOLD_FRAMES_NEEDED:
                    if pred_label == "space":
                        st.session_state.sentence += " "
                    elif pred_label == "del":
                        st.session_state.sentence = st.session_state.sentence[:-1]
                    elif pred_label != "nothing":
                        st.session_state.sentence += pred_label
            else:
                st.session_state.hold_frames = 0
        else:
            st.session_state.ema          = None
            st.session_state.hold_frames  = 0
            st.session_state.last_letter  = ""

        # FPS
        t_now = time.time()
        fps   = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # Progress bar for hold
        hold_pct = min(st.session_state.hold_frames / HOLD_FRAMES_NEEDED, 1.0)
        cv2.rectangle(frame, (0, frame.shape[0]-8), (int(frame.shape[1]*hold_pct), frame.shape[0]),
                      (124, 58, 237), -1)

        # Render video
        video_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )
        fps_placeholder.markdown(
            f'<div style="text-align:right;color:#64748b;font-size:12px;">'
            f'FPS: {fps:.1f} · Smoothing: {st.session_state.smoothing}</div>',
            unsafe_allow_html=True
        )

        # Render predictions
        letter_disp = pred_label if hand_found and pred_conf >= st.session_state.min_conf else "?"
        conf_disp   = pred_conf  if hand_found and pred_conf >= st.session_state.min_conf else 0.0

        pred_placeholder.markdown(f"""
        <div class="pred-card">
          <span class="big-letter">{letter_disp}</span>
          {confidence_bar_html("Confidence", conf_disp)}
          <p style="color:#94a3b8;font-size:13px;margin-top:8px;">
            Hold steady for ~1s to add to sentence
            {'<br><span style="color:#a78bfa;">⏳ ' + str(int(hold_pct*100)) + '%...</span>' if hold_pct > 0.05 else ""}
          """, unsafe_allow_html=True)


        # Top-K
        if topk_preds:
            bars_html  = "".join(
                confidence_bar_html(lbl, c, "#6366f1")
                for lbl, c in topk_preds
            )
            topk_placeholder.markdown(
                f'<div class="pred-card" style="padding:16px;">'
                f'<div class="section-header" style="text-align:left;">Top {st.session_state.topk_n} Predictions</div>'
                f'{bars_html}</div>',
                unsafe_allow_html=True
            )
        else:
            topk_placeholder.markdown(
                '<div class="pred-card" style="padding:16px;color:#64748b;">No hand detected</div>',
                unsafe_allow_html=True
            )

        # Hand status
        status_cls  = "status-active"   if hand_found else "status-inactive"
        status_text = "● Hand Detected"  if hand_found else "● No Hand"
        hand_placeholder.markdown(
            f'<div style="text-align:center;margin:8px 0;">'
            f'<span class="status-pill {status_cls}">{status_text}</span></div>',
            unsafe_allow_html=True
        )

        # Sentence
        sentence_ph.markdown(
            f'<div class="sentence-box">{st.session_state.sentence or "&nbsp;"}</div>',
            unsafe_allow_html=True
        )

except Exception as e:
    st.error(f"Detection error: {e}")

finally:
    # Safely release resources — never let cleanup crash the app
    try:
        if cap is not None and cap.isOpened():
            cap.release()
    except Exception:
        pass
    try:
        hands_detector.close()
    except Exception:
        pass
