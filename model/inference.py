import joblib                    # load and deserialize models saved with joblib
import numpy as np               # numeric operations library (not used directly here but commonly used with model inputs)
from pathlib import Path         # object-oriented filesystem paths

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"  # absolute path to the serialized model file located next to this script

def load_model():
    model = joblib.load(MODEL_PATH)  # load the model object from the MODEL_PATH file
    return model                      # return the deserialized model to the caller

def predict_from_dataframe(model, df):
    # expects columns: temperature, vibration, pressure, rpm
    features = ["temperature", "vibration", "pressure", "rpm"]  # ordered list of feature column names to use for prediction
    X = df[features].values                                      # extract a NumPy array of shape (n_samples, n_features) from the DataFrame
    # decision_function: higher means more normal; predict: -1 anomaly, 1 normal
    scores = model.decision_function(X)                          # get raw decision scores from the model (higher -> more normal for many anomaly detectors)
    predictions = model.predict(X) # -1 for anomaly, 1 for normal  # get discrete predictions/labels from the model (-1 indicates anomaly, 1 indicates normal)
    # convert to anomaly score where higher -> more anomalous
    anomaly_score = -scores                                       # invert the decision scores so that larger values indicate more anomalous behavior
    return anomaly_score, predictions                             # return a tuple (anomaly scores array, prediction labels array)