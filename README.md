# ❤️ Heart Disease Prediction ML API

## 📌 Project Overview

This project demonstrates an end-to-end Machine Learning deployment pipeline.
The system predicts the likelihood of heart disease based on patient health metrics.

The model was trained using multiple algorithms, evaluated properly, and deployed
as a production-ready REST API using FastAPI.

---

## 🚀 Features

- Multi-model comparison:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- Proper train-test validation
- ColumnTransformer-based preprocessing
- Missing value handling (SimpleImputer)
- Model retraining on full dataset
- Production-ready FastAPI service
- Strict schema validation using Pydantic
- Structured JSON API responses
- Version-controlled with Git

---

## 🏗 Architecture

Training Phase:
Raw Data → Preprocessing → Model Comparison → Best Model → model.pkl

Deployment Phase:
Client JSON → FastAPI → Data Validation → Preprocessing Pipeline →
Random Forest → Prediction → JSON Response


---

## 📂 Project Structure

Heart_Disease/
│
├── app/
│ ├── main.py # FastAPI application
│ ├── schema.py # Input validation schema
│ └── model_loader.py # Model loading logic
│
├── data/
│ └── heart.csv # Dataset
│
├── model/
│ └── model.pkl # Trained model pipeline
│
├── src/
│ └── train.py # Training script
│
├── requirements.txt
└── README.md


---

## 🧠 Model Details

- Dataset: UCI Heart Disease Dataset (Multiple hospital sources)
- Target: Binary classification (0 = No Disease, 1 = Disease)
- Best Model: Random Forest
- Validation Accuracy: ~85%

---