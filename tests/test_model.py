import joblib
import numpy as np


def test_model_correctness():
    clf = joblib.load(f"models/diabetes_model.pkl")
    scaler = joblib.load(f"models/scaler.pkl")
    data = [0.0, 120.0, 74.0, 18.0, 63.0, 30.5, 0.285, 26.0]
    x = np.array(data).reshape(-1, 8)
    x_scaled = scaler.transform(x)
    pred = clf.predict(x_scaled)[0]
    assert pred == 0


if __name__ == "__main__":
    test_model_correctness()