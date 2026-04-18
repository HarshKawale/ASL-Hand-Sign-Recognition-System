"""
data_extractor.py
-----------------
Extract MediaPipe hand landmarks from the Kaggle ASL Alphabet dataset
and save them as a CSV for training a landmark-based classifier.

For each image where a hand is detected:
  - Extract 21 landmarks (x, y, z) from MediaPipe
  - Normalize: make relative to wrist (landmark 0), then scale to [-1, 1]
  - Save as a row in asl_landmarks.csv (63 feature columns + label)
"""

import cv2
import mediapipe as mp
import os
import csv
import time

# ---- CONFIG ----
DATASET_PATH = "C:/Users/harsh/Downloads/archive/asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV = "C:/Kakarotto/Asl hand sign/asl_venv/asl_landmarks2.csv"

CLASSES = sorted(os.listdir(DATASET_PATH))
print(f"Found {len(CLASSES)} classes: {CLASSES}")

# ---- MediaPipe setup (static image mode for dataset images) ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


def normalize_landmarks(hand_landmarks):
    """
    Convert 21 MediaPipe landmarks to a normalized 63-element list.
    
    Steps (matching AkramOM606's approach):
    1. Extract (x, y, z) coords for each of 21 landmarks
    2. Make relative to wrist (landmark 0) - position invariant
    3. Divide by max absolute value - scale invariant
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
        coords.append(lm.z)

    # Make relative to wrist (first landmark)
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


def main():
    data = []
    total_images = 0
    detected_hands = 0
    start_time = time.time()

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_count = 0
        class_detected = 0
        files = os.listdir(class_dir)

        for img_file in files:
            img_path = os.path.join(class_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            total_images += 1
            class_count += 1

            # MediaPipe needs RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = normalize_landmarks(hand_landmarks)
                landmarks.append(class_name)  # label
                data.append(landmarks)
                detected_hands += 1
                class_detected += 1

        elapsed = time.time() - start_time
        rate = total_images / elapsed if elapsed > 0 else 0
        print(f"  [{class_idx+1:2d}/{len(CLASSES)}] {class_name:>8s}: {class_detected}/{class_count} hands detected "
              f"({rate:.0f} img/s, total: {detected_hands})")

    # Write CSV
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    elapsed = time.time() - start_time
    print(f"\n✅ Done! Processed {total_images} images in {elapsed:.1f}s")
    print(f"✅ Hands detected: {detected_hands}/{total_images} ({detected_hands/total_images*100:.1f}%)")
    print(f"✅ Saved to: {OUTPUT_CSV}")
    print(f"✅ CSV shape: {detected_hands} rows x {len(header)} columns")


if __name__ == "__main__":
    main()

