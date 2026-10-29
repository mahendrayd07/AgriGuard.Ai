# AgriGuard AI

AI-Based Tomato Leaf Disease Detection and Crop Health Screening System.

## Features

- Upload a tomato leaf image
- Detects Healthy Leaf, Early Blight, or Late Blight
- Displays prediction confidence
- Shows an uncertainty warning below 65% confidence
- Provides safe advisory text
- Saves reports using SQLite
- Shows saved report history in the dashboard

## Technology Stack

- Python
- FastAPI
- TensorFlow and MobileNetV2
- HTML, CSS, JavaScript
- SQLite
- PlantVillage dataset

## Supported Classes

1. Tomato Early Blight
2. Tomato Late Blight
3. Healthy Tomato Leaf

## Model Evaluation

- Validation images: 900
- Validation accuracy: 92.89%
- Weighted F1-score: 92.60%

Evaluation files:

- `models/evaluation_report.txt`
- `models/confusion_matrix.txt`

## Run the Project

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```
