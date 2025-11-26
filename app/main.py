from fastapi import FastAPI, UploadFile
import pandas as pd
from joblib import load

app = FastAPI(
    title="AI Predictive Machine Failure Detector",
    description="Predicts machine failures using sensor CSV uploads",
    version="1.0"
)
