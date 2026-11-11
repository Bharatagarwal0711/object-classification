from pathlib import Path
from collections import Counter, deque

import cv2
import joblib
import numpy as np

from skimage.feature import hog

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "results" / "models"

IMAGE_SIZE = (128, 128)
CAMERA_INDEX = 0

BACKGROUND_LABEL = "background"
CONFIDENCE_THRESHOLD = 60
VOTING_FRAMES = 5


def extract_hog_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)

    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
    )

    return features.astype(np.float32).reshape(1, -1)


def load_models():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            "Models folder nahi mila. Pehle train_models.py run karo."
        )

    loaded_models = []

    for path in sorted(MODEL_DIR.glob("*.joblib")):
        bundle = joblib.load(path)

        if not isinstance(bundle, dict):
            continue

        if "model" not in bundle or "objects" not in bundle:
            continue

        if tuple(bundle.get("image_size", ())) != IMAGE_SIZE:
            continue

        objects = bundle["objects"]

        if BACKGROUND_LABEL not in objects:
            print(f"Skipping old model without background: {path.name}")
            continue

        loaded_models.append({
            "name": bundle.get("model_name", path.stem),
            "model": bundle["model"],
            "objects": objects,
        })

    if not loaded_models:
        raise RuntimeError(
            "Compatible models nahi mile. "
            "Preprocess.py aur train_models.py dobara run karo."
        )

    return loaded_models


def predict_object(model, objects, features):
    prediction = int(model.predict(features)[0])
    label = objects[prediction]

    confidence = 0.0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities)) * 100

    return label, confidence


def main():
    models = load_models()

    print("\nLoaded models:")
    for item in models:
        print("-", item["name"])

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Webcam open nahi hua.")

    # Recent predictions for each model
    history = {
        item["name"]: deque(maxlen=VOTING_FRAMES)
        for item in models
    }

    print("\nObject ko center mein rakho.")
    print("Q dabakar exit karo.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Camera frame read nahi hua.")
                break

            frame = cv2.flip(frame, 1)
            features = extract_hog_features(frame)

            panel_height = 65 + len(models) * 32

            cv2.rectangle(
                frame,
                (10, 10),
                (min(frame.shape[1] - 10, 650), panel_height),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                "OBJECT CLASSIFICATION",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            for index, item in enumerate(models):
                name = item["name"]
                model = item["model"]
                objects = item["objects"]

                label, confidence = predict_object(
                    model, objects, features
                )

                history[name].append((label, confidence))

                predictions = list(history[name])
                labels = [prediction[0] for prediction in predictions]

                most_common_label, count = Counter(labels).most_common(1)[0]

                matching_confidences = [
                    conf for pred_label, conf in predictions
                    if pred_label == most_common_label
                ]

                average_confidence = (
                    sum(matching_confidences) / len(matching_confidences)
                    if matching_confidences else 0
                )

                # Show background only when model predicts background
                # consistently across recent frames.
                if most_common_label == BACKGROUND_LABEL and count >= 3:
                    display_text = f"{name}: No Object Detected"

                elif (
                    most_common_label != BACKGROUND_LABEL
                    and count >= 3
                    and average_confidence >= CONFIDENCE_THRESHOLD
                ):
                    display_text = (
                        f"{name}: {most_common_label} "
                        f"({average_confidence:.1f}%)"
                    )

                else:
                    display_text = f"{name}: Uncertain"

                cv2.putText(
                    frame,
                    display_text,
                    (20, 70 + index * 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "Keep object centered | Q: Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Object Classification", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Webcam closed.")


if __name__ == "__main__":
    main()