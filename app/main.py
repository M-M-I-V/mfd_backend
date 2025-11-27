from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
from model.inference import load_model, predict_from_dataframe
from utils.alert_trigger import batch_alerts
from utils.report_generator import generate_text_report

app = FastAPI(title="Machine Failure Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = load_model()


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.6):
    contents = await file.read()
    # read CSV into DataFrame
    from io import BytesIO
    df = pd.read_csv(BytesIO(contents))

    anomaly_scores, preds = predict_from_dataframe(MODEL, df)

    alerts = batch_alerts(df, anomaly_scores, preds, threshold)
    report = generate_text_report(alerts)

    # return a concise payload for the frontend
    return {
        "n_rows": len(df),
        "n_alerts": len(alerts),
        "alerts": alerts[:20],
        "report": report,
    }