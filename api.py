from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# =========================
# Config
# =========================
MODEL_PATH = "heart_rf_pipeline.pkl"
THRESHOLD = 0.20

# =========================
# Load artifact
# =========================
artifact = joblib.load(MODEL_PATH)

if isinstance(artifact, dict) and "model" in artifact:
    model = artifact["model"]
    feature_columns = artifact.get("feature_columns", [])
    categorical_columns = artifact.get(
        "categorical_columns",
        ["cp", "restecg", "slope", "ca", "thal"]
    )
else:
    model = artifact
    feature_columns = []
    categorical_columns = ["cp", "restecg", "slope", "ca", "thal"]

# =========================
# App
# =========================
app = FastAPI(
    title="Heart Disease Risk API",
    description="Predict heart disease probability and risk level from patient data.",
    version="1.0.0",
)


# =========================
# Schema
# =========================
class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120, example=63)
    sex: int = Field(..., ge=0, le=1, example=1, description="0 = Female, 1 = Male")
    cp: int = Field(..., ge=0, le=3, example=3)
    trestbps: int = Field(..., ge=50, le=250, example=150)
    chol: int = Field(..., ge=50, le=700, example=300)
    fbs: int = Field(..., ge=0, le=1, example=1)
    thalach: int = Field(..., ge=50, le=250, example=120)
    exang: int = Field(..., ge=0, le=1, example=1)
    oldpeak: float = Field(..., ge=0, le=10, example=3.5)
    restecg: int = Field(..., ge=0, le=2, example=2)
    slope: int = Field(..., ge=0, le=2, example=2)
    ca: int = Field(..., ge=0, le=4, example=3)
    thal: int = Field(..., ge=0, le=3, example=3)


# =========================
# Helpers
# =========================
def make_input_df(payload: PatientInput) -> pd.DataFrame:
    raw_df = pd.DataFrame([{
        "age": payload.age,
        "sex": payload.sex,
        "cp": payload.cp,
        "trestbps": payload.trestbps,
        "chol": payload.chol,
        "fbs": payload.fbs,
        "thalach": payload.thalach,
        "exang": payload.exang,
        "oldpeak": payload.oldpeak,
        "restecg": payload.restecg,
        "slope": payload.slope,
        "ca": payload.ca,
        "thal": payload.thal,
    }])

    if feature_columns:
        df = raw_df.copy()
        df[categorical_columns] = df[categorical_columns].astype(str)
        df = pd.get_dummies(df, columns=categorical_columns)
        df = df.reindex(columns=feature_columns, fill_value=0)
        return df

    return raw_df


def risk_label_from_proba(proba: float) -> Literal["Low Risk", "Medium Risk", "High Risk"]:
    if proba < 0.20:
        return "Low Risk"
    if proba < 0.50:
        return "Medium Risk"
    return "High Risk"


# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {
        "message": "Heart Disease Risk API is running",
        "docs": "/docs",
        "threshold": THRESHOLD,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PatientInput):
    input_df = make_input_df(payload)

    proba = float(model.predict_proba(input_df)[0][1])
    pred_class = int(proba >= THRESHOLD)
    risk_label = risk_label_from_proba(proba)

    return {
        "probability": round(proba, 4),
        "predicted_class": pred_class,
        "threshold": THRESHOLD,
        "risk_level": risk_label,
        "input": payload.model_dump(),
    }


@app.get("/predict_get")
def predict_get(
    age: int,
    sex: int,
    cp: int,
    trestbps: int,
    chol: int,
    fbs: int,
    thalach: int,
    exang: int,
    oldpeak: float,
    restecg: int,
    slope: int,
    ca: int,
    thal: int,
):
    payload = PatientInput(
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        restecg=restecg,
        slope=slope,
        ca=ca,
        thal=thal,
    )

    input_df = make_input_df(payload)
    proba = float(model.predict_proba(input_df)[0][1])
    pred_class = int(proba >= THRESHOLD)
    risk_label = risk_label_from_proba(proba)

    return {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "restecg": restecg,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "probability": round(proba, 4),
        "predicted_class": pred_class,
        "threshold": THRESHOLD,
        "risk_level": risk_label,
    }


@app.get("/sample_predictions")
def sample_predictions():
    samples = [
        {"age": 45, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250, "fbs": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "restecg": 1, "slope": 1, "ca": 0, "thal": 2},
        {"age": 60, "sex": 1, "cp": 3, "trestbps": 150, "chol": 300, "fbs": 1, "thalach": 120, "exang": 1, "oldpeak": 3.5, "restecg": 2, "slope": 2, "ca": 3, "thal": 3},
        {"age": 50, "sex": 0, "cp": 1, "trestbps": 120, "chol": 220, "fbs": 0, "thalach": 160, "exang": 0, "oldpeak": 0.5, "restecg": 0, "slope": 1, "ca": 0, "thal": 2}
    ]

    results = []
    for s in samples:
        payload = PatientInput(**s)
        df = make_input_df(payload)
        proba = float(model.predict_proba(df)[0][1])
        risk = risk_label_from_proba(proba)

        s["probability"] = round(proba, 4)
        s["risk_level"] = risk
        results.append(s)

    return results