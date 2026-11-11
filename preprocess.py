import os
import cv2
import numpy as np

from skimage.feature import hog
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset"
IMAGE_SIZE = (128, 128)

objects = [
    "watch",
    "fork",
    "glasses",
    "pen",
    "stapler",
    "notebook",
    "remote",
    "mouse",
    "cup",
    "headphones",
    "background",
]

X = []
y = []

print("=" * 60)
print("OBJECT CLASSIFICATION - HOG PREPROCESSING")
print("=" * 60)


def extract_hog_features(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
    )
    return features.astype(np.float32)


for label, object_name in enumerate(objects):
    folder_path = os.path.join(DATASET_PATH, object_name)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder nahi mila: {folder_path}")

    image_files = sorted([
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"\nProcessing {object_name}: {len(image_files)} images")

    if len(image_files) != 50:
        raise ValueError(
            f"{object_name} folder mein exactly 50 images honi chahiye."
        )

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Image read nahi hui: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMAGE_SIZE)

        features = extract_hog_features(resized)

        X.append(features)
        y.append(label)


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("\nPreprocessing complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nLabel mapping:")
for label, object_name in enumerate(objects):
    print(f"{label} = {object_name}")

class_counts = np.bincount(y, minlength=len(objects))

print("\nImages per class:")
for label, object_name in enumerate(objects):
    print(f"{object_name}: {class_counts[label]}")

if len(X) != 550:
    raise ValueError("Dataset mein total 550 images honi chahiye.")

if not np.all(class_counts == 50):
    raise ValueError("Har class mein exactly 50 images honi chahiye.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nDataset split complete!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of features:", X.shape[1])
print("Preprocessing successful!")

