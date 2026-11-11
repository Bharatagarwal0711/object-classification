
from pathlib import Path
import time
import csv
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preprocess import X_train, X_test, y_train, y_test, objects, IMAGE_SIZE


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "results" / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    ),
    "SVM": SVC(
        kernel="linear",
        probability=True,
        random_state=42,
    ),
    "Naive Bayes": GaussianNB(),
}


results = []

print("=" * 70)
print("TRAINING SIX SUPERVISED ML MODELS")
print("=" * 70)

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")
    start_time = time.time()

    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2),
        "Training Time (seconds)": round(training_time, 3),
    })

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"Time:      {training_time:.3f} seconds")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=np.arange(len(objects)),
        target_names=objects,
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(objects)),
    ))

    safe_name = model_name.lower().replace(" ", "_")

    bundle = {
        "model": model,
        "model_name": model_name,
        "image_size": IMAGE_SIZE,
        "objects": objects,
    }

    joblib.dump(bundle, MODEL_DIR / f"{safe_name}.joblib")


results.sort(key=lambda item: item["Accuracy"], reverse=True)

csv_path = RESULTS_DIR / "model_comparison.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)


print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

for result in results:
    print(
        f"{result['Model']:20} "
        f"Accuracy: {result['Accuracy']:6.2f}%"
    )

best_model_name = results[0]["Model"]
print(f"\nBest model based on test accuracy: {best_model_name}")
print(f"Comparison saved to: {csv_path}")
print(f"Models saved to: {MODEL_DIR}")

