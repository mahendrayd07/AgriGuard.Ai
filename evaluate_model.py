import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path("data/tomato_raw")
MODEL_PATH = Path("models/tomato_disease.keras")
CLASS_PATH = Path("models/class_names.json")
OUTPUT_DIR = Path("models")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

model = tf.keras.models.load_model(MODEL_PATH)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

true_labels = []
predicted_labels = []

for images, labels in validation_dataset:
    probabilities = model(images, training=False).numpy()
    predictions = probabilities.argmax(axis=1)

    true_labels.extend(labels.numpy())
    predicted_labels.extend(predictions)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    digits=4,
)

print("\nMODEL EVALUATION REPORT\n")
print(report)

with open(OUTPUT_DIR / "evaluation_report.txt", "w", encoding="utf-8") as file:
    file.write(report)

matrix = confusion_matrix(true_labels, predicted_labels)

matrix_text = "Confusion Matrix\n\n" + np.array2string(matrix)

with open(OUTPUT_DIR / "confusion_matrix.txt", "w", encoding="utf-8") as file:
    file.write(matrix_text)

print("\nSaved: models/evaluation_report.txt")
print("Saved: models/confusion_matrix.txt")