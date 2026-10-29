import json
from pathlib import Path

import tensorflow as tf

DATA_DIR = Path("data/rice_raw")
MODEL_DIR = Path("models")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42

MODEL_DIR.mkdir(exist_ok=True)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_dataset.class_names
print("Rice classes:", class_names)

autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(autotune)
validation_dataset = validation_dataset.prefetch(autotune)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    augmentation,
    tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)

model.save(MODEL_DIR / "rice_disease.keras")

with open(MODEL_DIR / "rice_class_names.json", "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)

loss, accuracy = model.evaluate(validation_dataset)
print(f"\nRice validation accuracy: {accuracy * 100:.2f}%")
print("Saved model: models/rice_disease.keras")