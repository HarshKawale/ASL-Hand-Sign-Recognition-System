import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

CLASSES = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

DATA_DIR = "C:/Users/harsh/Downloads/archive/asl_alphabet_test/asl_alphabet_test" 
MODEL_PATH = "c:/Kakarotto/Asl hand sign/asl_venv/resnet50v2_production.h5"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return

    print(f"🔹 Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    H, W = model.input_shape[1], model.input_shape[2]
    print(f"✅ Model loaded. Input Shape: ({H}, {W})")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        return

    correct = 0
    total = 0

    print(f"🔹 Testing images in {DATA_DIR}...")
    files = sorted(os.listdir(DATA_DIR))
    
    for filename in files:
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Derive true label from filename: "A_test.jpg" -> "A"
        try:
            true_label = filename.split('_')[0]
        except IndexError:
            print(f"⚠️ Skipping {filename}: Cannot parse class.")
            continue

        if true_label not in CLASSES:
            # Maybe the mapping is different?
            # Check if there is a close match?
            pass
            # For now assume exact match.
            # print(f"⚠️ Skipping {filename}: '{true_label}' not in class list.")
            # continue 
            # Actually, let's process it and see prediction.
        
        # Load image
        img_path = os.path.join(DATA_DIR, filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"❌ Failed to read {filename}")
            continue
            
        # Resize & Preprocess (Verify correct steps)
        # 1. Resize to model shape
        img_resized = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)
        
        # 2. Convert BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Preprocess
        x = img_rgb.astype(np.float32)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        
        # Predict
        probs = model.predict(x, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_label = CLASSES[pred_idx]
        conf = probs[pred_idx]
        
        is_correct = (pred_label == true_label)
        if is_correct:
            correct += 1
            print(f"✅ {filename:<18} -> {pred_label} ({conf*100:.1f}%)")
        else:
            print(f"❌ {filename:<18} -> {pred_label} ({conf*100:.1f}%) [True: {true_label}]")
        
        total += 1

    if total > 0:
        print(f"\n📊 Final Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    else:
        print("⚠️ No valid images found.")

if __name__ == "__main__":
    main()
