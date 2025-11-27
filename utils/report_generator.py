def generate_text_report(alerts):
    if not alerts:
         return "No anomalies detected. System healthy."

    lines = []
    lines.append("Anomaly Report:\n")
    for a in alerts:
        lines.append(f"- {a['id']} at {a['timestamp']} -> {a['message']}")
        lines.append(f" temp={a['temperature']:.1f}, vib={a['vibration']:.2f}, rpm={a['rpm']:.0f}, score={a['anomaly_score']:.3f}")

    return "\n".join(lines)