import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_DIR / "agri_guard.db"

MODELS = {
    "tomato": tf.keras.models.load_model(
        PROJECT_DIR / "models" / "tomato_disease.keras"
    ),
    "potato": tf.keras.models.load_model(
        PROJECT_DIR / "models" / "potato_disease.keras"
    ),
    "rice": tf.keras.models.load_model(
        PROJECT_DIR / "models" / "rice_disease.keras"
    ),
}

with open(PROJECT_DIR / "models" / "class_names.json", encoding="utf-8") as file:
    tomato_classes = json.load(file)

with open(PROJECT_DIR / "models" / "potato_class_names.json", encoding="utf-8") as file:
    potato_classes = json.load(file)

with open(PROJECT_DIR / "models" / "rice_class_names.json", encoding="utf-8") as file:
    rice_classes = json.load(file)

CLASS_NAMES = {
    "tomato": tomato_classes,
    "potato": potato_classes,
    "rice": rice_classes,
}

display_names = {
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___healthy": "Healthy Tomato Leaf",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
    "Potato___healthy": "Healthy Potato Leaf",
    "Rice___Bacterial_leaf_blight": "Rice Bacterial Leaf Blight",
    "Rice___Blast": "Rice Blast",
    "Rice___healthy": "Healthy Rice Leaf",
    "Rice___Tungro": "Rice Tungro",
}

advisories = {
    "Tomato___Early_blight": "Remove affected leaves, avoid wetting foliage, and follow local agricultural guidance.",
    "Tomato___Late_blight": "This may spread quickly in cool, wet conditions. Seek local agricultural guidance.",
    "Tomato___healthy": "No disease pattern was detected. Continue regular crop monitoring.",
    "Potato___Early_blight": "Remove affected foliage where practical and monitor nearby potato plants. Follow local agricultural guidance.",
    "Potato___Late_blight": "This may spread quickly in cool, wet conditions. Seek local agricultural guidance promptly.",
    "Potato___healthy": "No disease pattern was detected. Continue regular crop monitoring.",
    "Rice___Bacterial_leaf_blight": "Remove severely affected leaves where practical and follow local agricultural guidance.",
    "Rice___Blast": "Monitor nearby plants and use locally recommended disease-management practices.",
    "Rice___healthy": "No disease pattern was detected. Continue regular crop monitoring.",
    "Rice___Tungro": "Seek local agricultural guidance, as Tungro can spread through insect vectors.",
}

# These are high-level Integrated Pest Management options for India / Uttar Pradesh.
# They intentionally do not include doses, tank mixes, or spray schedules: those must
# come from the current product label and a local KVK or agriculture officer.
treatments = {
    "Tomato___Early_blight": {
        "organic": [
            "Remove affected leaves and crop debris; disinfect pruning tools between plants.",
            "Improve spacing and airflow, stake plants, and avoid overhead irrigation.",
            "Use a locally registered Bacillus- or Trichoderma-based biofungicide only as its label directs.",
        ],
        "chemical": [
            "Indian advisories list copper oxychloride, mancozeb, or metalaxyl + mancozeb for blight management.",
            "Use only one product registered for tomato and the diagnosed disease; confirm the current label, dose, interval, and pre-harvest interval with a UP KVK or agriculture officer.",
        ],
    },
    "Tomato___Late_blight": {
        "organic": [
            "Remove heavily affected foliage, avoid wet leaves, and improve drainage and airflow.",
            "Destroy crop residues after harvest and do not compost visibly diseased material near tomato beds.",
            "A locally registered Bacillus- or Trichoderma-based biofungicide may be used only according to its label.",
        ],
        "chemical": [
            "Indian ICAR advisories list cymoxanil + mancozeb and pyraclostrobin + metiram among options used for tomato blight.",
            "Do not mix fungicides without expert advice. Confirm the product registration, label dose, resistance-management rotation, and harvest interval locally.",
        ],
    },
    "Potato___Early_blight": {
        "organic": [
            "Remove infected leaves, control weeds, and keep foliage dry where possible.",
            "Use healthy seed tubers, rotate away from potato and tomato crops, and remove volunteer plants.",
            "Use a locally registered Trichoderma- or Bacillus-based biofungicide only according to its label.",
        ],
        "chemical": [
            "Ask a UP KVK or licensed dealer for a product registered on potato for early blight; commonly discussed active ingredients include mancozeb or chlorothalonil.",
            "Follow the product label exactly for dose, timing, protective equipment, and harvest interval.",
        ],
    },
    "Potato___Late_blight": {
        "organic": [
            "Remove infected foliage promptly, use disease-free seed tubers, and avoid working in fields when leaves are wet.",
            "Improve drainage, earth up tubers, and destroy crop residues after harvest.",
            "Use biological products only when locally registered and labelled for this crop and disease.",
        ],
        "chemical": [
            "ICAR reports list chlorothalonil and fluopicolide + propamocarb hydrochloride in late-blight management programmes.",
            "Confirm the current registration, label, resistance-management rotation, dose, and harvest interval with a UP KVK before use.",
        ],
    },
    "Rice___Bacterial_leaf_blight": {
        "organic": [
            "Use clean seed, balanced nitrogen, good drainage, and avoid moving from infected to healthy fields with wet tools.",
            "Remove severely affected clumps where practical and follow local resistant-variety advice.",
            "Do not expect an organic spray to cure established bacterial leaf blight; get a field diagnosis if symptoms spread quickly.",
        ],
        "chemical": [
            "There is no guaranteed curative chemical treatment for established bacterial leaf blight.",
            "Consult a UP KVK before buying any bactericide. Use only a product currently registered for rice and bacterial disease, with the label dose and safety interval.",
        ],
    },
    "Rice___Blast": {
        "organic": [
            "Remove heavily infected leaves where feasible, avoid excessive nitrogen, and keep field nutrition balanced.",
            "Use locally recommended resistant varieties and clean seed for the next crop cycle.",
            "Use a locally registered Trichoderma- or Pseudomonas-based bio-input only as labelled.",
        ],
        "chemical": [
            "Common active ingredients discussed for rice blast in India include tricyclazole and isoprothiolane.",
            "Confirm current rice registration, label dose, waiting period, and resistance-management advice with a UP KVK before application.",
        ],
    },
    "Rice___Tungro": {
        "organic": [
            "Rogue infected plants early, synchronise planting with nearby farmers where possible, and manage grassy weeds.",
            "Use locally recommended resistant varieties and monitor green leafhopper activity.",
            "There is no organic product that cures a tungro-infected plant.",
        ],
        "chemical": [
            "There is no chemical medicine that cures tungro virus in an infected plant.",
            "For vector management, ask a UP KVK for a currently registered rice product targeting green leafhopper; use it only at the label dose and timing.",
        ],
    },
    "Tomato___healthy": {
        "organic": [
            "Continue crop scouting, sanitation, balanced nutrition, and good airflow.",
        ],
        "chemical": [
            "No chemical treatment is indicated by this screening result.",
        ],
    },
    "Potato___healthy": {
        "organic": [
            "Continue crop scouting, use healthy seed tubers, and remove volunteer plants.",
        ],
        "chemical": [
            "No chemical treatment is indicated by this screening result.",
        ],
    },
    "Rice___healthy": {
        "organic": [
            "Continue field scouting, balanced nutrition, clean seed practices, and weed management.",
        ],
        "chemical": [
            "No chemical treatment is indicated by this screening result.",
        ],
    },
}

def setup_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                advisory TEXT NOT NULL,
                image_name TEXT,
                created_at TEXT NOT NULL
            )
        """)

def save_report(result, image_name):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            INSERT INTO reports
            (crop, prediction, confidence, severity, advisory, image_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result["crop"],
            result["prediction"],
            result["confidence"],
            result["severity"],
            result["advisory"],
            image_name,
            datetime.now(timezone.utc).isoformat(),
        ))

setup_database()

app = FastAPI(title="AgriGuard AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"message": "AgriGuard AI backend is running"}

@app.get("/reports")
def get_reports():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM reports ORDER BY id DESC"
        ).fetchall()

    return [dict(row) for row in rows]

@app.post("/predict")
async def predict_disease(
    crop: str = Form(...),
    image: UploadFile = File(...),
):
    crop_key = crop.lower()

    if crop_key not in MODELS:
        raise HTTPException(
            status_code=400,
            detail="This crop is not supported yet."
        )

    try:
        image_bytes = await image.read()
        leaf_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        leaf_image = leaf_image.resize((224, 224))

        image_array = np.expand_dims(
            np.array(leaf_image, dtype=np.float32),
            axis=0,
        )

        probabilities = MODELS[crop_key].predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[crop_key][predicted_index]
        confidence = round(float(probabilities[predicted_index]) * 100, 2)

        if confidence < 65:
            result = {
                "crop": crop_key.title(),
                "prediction": "Uncertain result",
                "confidence": confidence,
                "severity": "Not estimated",
                "advisory": "The model is not confident. Upload a clear image of the selected crop and seek expert guidance if needed.",
                "treatment": {
                    "organic": [
                        "Retake a clear image of a single affected leaf in natural light.",
                        "Avoid treatment decisions until a local agricultural expert confirms the diagnosis.",
                    ],
                    "chemical": [
                        "Do not apply a chemical treatment from an uncertain AI result.",
                    ],
                },
            }
        else:
            result = {
                "crop": crop_key.title(),
                "prediction": display_names[predicted_class],
                "confidence": confidence,
                "severity": "Not estimated yet",
                "advisory": advisories[predicted_class],
                "treatment": treatments[predicted_class],
            }

        save_report(result, image.filename)
        return result

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process this image: {str(error)}"
        )

app.mount(
    "/app",
    StaticFiles(directory=PROJECT_DIR / "frontend", html=True),
    name="frontend",
)
