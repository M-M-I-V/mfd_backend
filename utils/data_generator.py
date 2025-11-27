import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "sample_upload.csv"

def make_small_upload(n=200, anomaly_fraction=0.05, seed=123):
    np.random.seed(seed)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="T")
    temperature = np.random.normal(60, 5, size=n)
    vibration = np.random.normal(0.3, 0.1, size=n)
    pressure = np.random.normal(30, 3, size=n)
    rpm = np.random.normal(1500, 100, size=n)

    df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "rpm": rpm,
    })

    k = int(n * anomaly_fraction)
    idx = np.random.choice(n, size=k, replace=False)
    df.loc[idx, "temperature"] += np.random.uniform(12, 35, size=k)
    df.loc[idx, "vibration"] += np.random.uniform(0.4, 0.95, size=k)

    df.to_csv(OUT, index=False)
    print("Created sample_upload.csv at", OUT)

if __name__ == '__main__':
    make_small_upload()