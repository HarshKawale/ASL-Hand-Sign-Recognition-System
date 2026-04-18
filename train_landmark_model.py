"""
train_landmark_model.py
-----------------------
Train a RandomForest classifier on the extracted hand landmarks CSV.
Saves the trained model as landmark_classifier.pkl.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time

# ---- CONFIG ----
CSV_PATH = "C:/Kakarotto/Asl hand sign/asl_venv/asl_landmarks2.csv"
MODEL_PATH = "C:/Kakarotto/Asl hand sign/asl_venv/landmark_classifier2.pkl"
TEST_SPLIT = 0.15
RANDOM_STATE = 42


def main():
    print(f"📂 Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"✅ Classes: {sorted(df['label'].unique())}")
    print(f"✅ Samples per class:")
    print(df['label'].value_counts().sort_index().to_string())

    # Split features and labels
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

    # Train RandomForest
    print("\n🏋️ Training RandomForest classifier...")
    start = time.time()
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,            # use all CPU cores
        random_state=RANDOM_STATE,
        verbose=1
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"✅ Training done in {elapsed:.1f}s")

    # Evaluate
    print("\n📊 Evaluating on test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
