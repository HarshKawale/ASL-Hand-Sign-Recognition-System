"""
testing_asl.py - Landmark-Based Real-Time ASL Detection
-------------------------------------------------------
Uses MediaPipe hand landmarks + RandomForest classifier
instead of image-based CNN. This eliminates domain gap issues
(background, lighting, skin color don't affect landmarks).
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib

# ------------ CONFIG ------------
MODEL_PATH = "C:/Kakarotto/Asl hand sign/asl_venv/landmark_classifier2.pkl"
CLASSES = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

CAM_INDEX = 0
MIN_CONF = 0.7
SMOOTHING = 3                # EMA smoothing (1 = disabled)
TOPK = 3
FLIP_FRAME = True            # flip webcam horizontally

# ------------ LOAD MODEL ------------
clf = joblib.load(MODEL_PATH)
print("✅ Landmark classifier loaded:", MODEL_PATH)

# ------------ MEDIAPIPE ------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=MIN_CONF,
    min_tracking_confidence=MIN_CONF
)

# ------------ WEBCAM ------------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ema = None


def normalize_landmarks(hand_landmarks):
    """
    Convert 21 MediaPipe landmarks to normalized 63-element list.
    MUST match the normalization used during training (data_extractor.py).
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
        coords.append(lm.z)

    # Relative to wrist (landmark 0)
    base_x, base_y, base_z = coords[0], coords[1], coords[2]
    for i in range(0, len(coords), 3):
        coords[i] -= base_x
        coords[i + 1] -= base_y
        coords[i + 2] -= base_z

    # Normalize by max absolute value
    max_val = max(abs(c) for c in coords)
    if max_val > 0:
        coords = [c / max_val for c in coords]

    return coords


def get_bbox_from_landmarks(hand_landmarks, img_w, img_h, pad=30):
    """Get bounding box for display purposes only."""
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    xmin = max(0, int(min(xs) * img_w) - pad)
    ymin = max(0, int(min(ys) * img_h) - pad)
    xmax = min(img_w, int(max(xs) * img_w) + pad)
    ymax = min(img_h, int(max(ys) * img_h) + pad)
    return xmin, ymin, xmax, ymax


print("🎥 Live ASL started (Landmark mode)")
print("   'q' = quit | 'f' = toggle flip | 's' = toggle smoothing")
print(f"   Flip: {'ON' if FLIP_FRAME else 'OFF'} | Smoothing: {SMOOTHING}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if FLIP_FRAME:
        frame = cv2.flip(frame, 1)

    # MediaPipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    display_text = "No hand"
    color = (0, 0, 255)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]

        # ---- Normalize landmarks (same as training) ----
        features = normalize_landmarks(lm)
        features_array = np.array(features).reshape(1, -1)

        # ---- Predict ----
        probs = clf.predict_proba(features_array)[0]

        # EMA smoothing
        if ema is None:
            ema = probs
        else:
            alpha = 1.0 / SMOOTHING
            ema = (1 - alpha) * ema + alpha * probs

        pred_idx = int(np.argmax(ema))
        pred_label = CLASSES[pred_idx]
        conf = float(ema[pred_idx])

        # Top-k
        topk_idx = np.argsort(ema)[-TOPK:][::-1]
        topk_text = " | ".join([f"{CLASSES[i]}:{ema[i]*100:.0f}%" for i in topk_idx])

        display_text = f"{pred_label} ({conf*100:.1f}%)"
        color = (0, 255, 0)

        # Draw landmarks and bbox
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        img_h, img_w = frame.shape[:2]
        xmin, ymin, xmax, ymax = get_bbox_from_landmarks(lm, img_w, img_h)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, display_text, (xmin, max(30, ymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(frame, topk_text, (xmin, min(img_h - 10, ymax + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    else:
        ema = None

    # Status bar
    status = f"Mode: LANDMARK | Flip: {'ON' if FLIP_FRAME else 'OFF'} | Smooth: {SMOOTHING}"
    cv2.putText(frame, status, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("ASL Live (Landmark) - q to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        FLIP_FRAME = not FLIP_FRAME
        print(f"🔄 Flip: {'ON' if FLIP_FRAME else 'OFF'}")
    elif key == ord('s'):
        SMOOTHING = 1 if SMOOTHING > 1 else 3
        print(f"🔄 Smoothing: {SMOOTHING}")

cap.release()
cv2.destroyAllWindows()
hands.close()
print("✅ Closed.")
