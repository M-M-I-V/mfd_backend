import json
from datetime import datetime

def make_alert(df_row, anomaly_score, index):
    # df_row: pandas Series for the anomalous row
    alert = {
        "id": f"alert-{int(datetime.utcnow().timestamp())}-{index}",
        "timestamp": str(datetime.utcnow()),
        "anomaly_row_ts": str(df_row.get("timestamp")),
        "temperature": float(df_row.get("temperature")),
        "vibration": float(df_row.get("vibration")),
        "pressure": float(df_row.get("pressure")),
        "rpm": float(df_row.get("rpm")),
        "anomaly_score": float(anomaly_score),
        "message": "Anomalous reading detected — recommend inspection",
    }
    return alert

def batch_alerts(df, anomaly_scores, preds, threshold):
    # threshold: anomaly_score above which we create alert
    alerts = []
    for i, s in enumerate(anomaly_scores):
        if s >= threshold or preds[i] == -1:
            alerts.append(make_alert(df.iloc[i], s, i))
    return alerts