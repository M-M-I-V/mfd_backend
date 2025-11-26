import numpy as np                      # numerical operations and random sampling
import pandas as pd                     # data manipulation and DataFrame
from sklearn.ensemble import IsolationForest  # unsupervised anomaly detection model
import joblib                           # model serialization utilities
from pathlib import Path                # convenient filesystem path handling

# Path where the trained model will be saved (same directory as this file)
MODEL_PATH = Path(__file__).parent / "model.joblib"

def generate_synthetic_logs(n = 2000, anomaly_fraction = 0.5, seed = 42):
    # Generate synthetic machine logs with some injected anomalies.
    #
    # Args:
    #   n: total number of log rows to generate
    #   anomaly_fraction: fraction of rows to modify as anomalies (0..1)
    #   seed: RNG seed for reproducibility
    #
    # Returns:
    #   pd.DataFrame with columns: timestamp, temperature, vibration, pressure, rpm

    np.random.seed(seed)  # set RNG seed so generated data is reproducible

    # create a sequence of timestamps starting at 2025-01-01 at 1-minute intervals
    timestamps = pd.date_range("2025-01-01", periods = n, freq = "T")

    # generate normal (non-anomalous) sensor readings for each row
    temperature = np.random.normal(loc = 60, scale = 5, size = n)  # degrees
    vibration = np.random.normal(loc = 0.3, scale = 0.1, size = n) # vibration amplitude
    pressure = np.random.normal(loc = 30, scale = 3, size = n)     # pressure units
    rpm = np.random.normal(loc = 1500, scale = 100, size = n)      # rotations per minute

    # assemble the simulated data into a pandas DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "rpm": rpm
    })

    # determine how many rows to mark as anomalies
    k = int(n * anomaly_fraction)

    # choose k distinct row indices uniformly at random to become anomalous
    idx = np.random.choice(n, size = k, replace = False)

    # increase selected rows' sensor values to simulate faults/anomalies
    df.loc[idx, "temperature"] += np.random.uniform(15, 40, size = k)  # raise temperature
    df.loc[idx, "vibration"] += np.random.uniform(0.4, 1.0, size = k)  # raise vibration
    df.loc[idx, "pressure"] += np.random.uniform(10, 25, size = k)     # raise pressure
    df.loc[idx, "rpm"] += np.random.uniform(200, 800, size = k)        # raise rpm

    return df  # return the DataFrame with synthesized normal and anomalous rows

if __name__ == "__main__":
    # Example usage: generate data, train an IsolationForest, and persist model + sample data.

    # generate 3000 rows with 30% anomalies
    df = generate_synthetic_logs(n = 3000, anomaly_fraction = 0.3)

    # feature columns used for training the anomaly detector
    features = ["temperature", "vibration", "pressure", "rpm"]

    # extract feature matrix as a NumPy array (rows x features)
    X = df[features].values

    # configure an IsolationForest for unsupervised anomaly detection
    # n_estimators: number of trees; contamination: expected fraction of anomalies
    model = IsolationForest(n_estimators = 200, contamination = 0.3, random_state = 42)

    # fit the model to the synthetic feature matrix
    model.fit(X)

    # serialize and save the trained model to MODEL_PATH (model.joblib next to this file)
    joblib.dump(model, MODEL_PATH)

    # also save a CSV of the generated sample data two directories up as sample_data.csv
    df.to_csv(Path(__file__).resolve().parent.parent / "sample_data.csv", index = False)