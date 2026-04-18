import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import os

CLASSES = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

MODEL_PATH = "c:/Kakarotto/Asl hand sign/asl_venv/resnet50v2_production.h5"
CAM_INDEX = 0
MIN_CONF = 0.3  # Lowered threshold
SMOOTHING = 3
TOPK = 3

# Load model
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    exit()

print(f"🔹 Loading ResNet50V2 model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
H, W = model.input_shape[1], model.input_shape[2]
print(f"✅ Model loaded. Input: ({H}, {W})")

# MediaPipe for hand detection (optional - can disable)
USE_HANDS = True  # Set False to predict on entire frame always
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.3,  # Lowered
    min_tracking_confidence=0.3
)

# Webcam
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ema = None
FLIP_FRAME = True

def preprocess_for_resnet(img_bgr, bbox=None):
    """Preprocess for ResNet50V2"""
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        img_crop = img_bgr[ymin:ymax, xmin:xmax]
        img_resized = cv2.resize(img_crop, (W, H))
    else:
        img_resized = cv2.resize(img_bgr, (W, H))
    
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

def get_hand_bbox(hand_landmarks, img_shape):
    h, w = img_shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    pad = 50
    xmin = max(0, int(min(xs)) - pad)
    ymin = max(0, int(min(ys)) - pad)
    xmax = min(w, int(max(xs)) + pad)
    ymax = min(h, int(max(ys)) + pad)
    return xmin, ymin, xmax, ymax

print("🎥 Live ResNet50V2 CNN - ALWAYS PREDICTING!")
print("  'q'=quit | 'f'=flip | 's'=smoothing | 'h'=toggle hand detect")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if FLIP_FRAME:
        frame = cv2.flip(frame, 1)
    
    # ALWAYS predict on full frame OR hand-cropped frame
    if USE_HANDS:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        bbox = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            bbox = get_hand_bbox(lm, frame.shape)
    else:
        results = None
        bbox = None
    
    # ALWAYS PREPROCESS AND PREDICT (no "no hand" case)
    x = preprocess_for_resnet(frame, bbox)
    probs = model.predict(x, verbose=0)[0]
    
    # EMA smoothing
    if ema is None:
        ema = probs
    else:
        alpha = 1.0 / SMOOTHING
        ema = (1 - alpha) * ema + alpha * probs
    
    pred_idx = np.argmax(ema)
    pred_label = CLASSES[pred_idx]
    conf = ema[pred_idx]
    
    # Always show prediction (even low confidence)
    display_text = f"{pred_label} ({conf*100:.1f}%)"
    color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)  # Green if confident, orange if unsure
    
    # Top-K
    topk_idx = np.argsort(ema)[-TOPK:][::-1]
    topk_text = " | ".join([f"{CLASSES[i]}:{ema[i]*100:.0f}%" for i in topk_idx])
    
    # Draw bbox if hand detected
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, display_text, (xmin, max(30, ymin-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, topk_text, (xmin, ymax+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    else:
        # Full frame prediction
        cv2.putText(frame, display_text, (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, topk_text, (30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    
    # Status
    hand_status = "Hand" if bbox else "Full Frame"
    status = f"CNN Always-On | {hand_status} | Flip:{'ON' if FLIP_FRAME else 'OFF'} | Smooth:{SMOOTHING}"
    cv2.putText(frame, status, (10, frame.shape[0]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    
    cv2.imshow("ASL Live - ResNet50V2 CNN (Always Predicting)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        FLIP_FRAME = not FLIP_FRAME
    elif key == ord('s'):
        SMOOTHING = 1 if SMOOTHING > 1 else 3
    elif key == ord('h'):
        USE_HANDS = not USE_HANDS
        print(f"🔄 Hand detection: {'ON' if USE_HANDS else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
hands.close()
print("✅ Closed.")
