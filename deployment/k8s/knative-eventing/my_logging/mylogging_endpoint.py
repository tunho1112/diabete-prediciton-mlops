# Ref: https://github.com/cloudevents/sdk-python#receiving-cloudevents
from fastapi import FastAPI, Request
import pandas as pd 
from scipy.stats import ks_2samp


app = FastAPI()


# Create an endpoint at http://localhost:8000
@app.post("/")
async def on_event(request: Request):
    # Inspect the data
    data = await request.json()
    # Inspect the headers
    headers = request.headers

    print("Received a new event!")
    print("data: ", data)
    print("Is Drift", is_drift(data["input_data"]))
    # Return no content
    return {"message": "Event received successfully."}


async def is_drift(input_data) -> bool:
    features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
        ]
    new_df = pd.DataFrame(input_data, columns=features)
    val_df = pd.read_csv("validation.csv")
    ref_df = val_df[features]
    report = detect_data_drift(ref_df, new_df)

    return overall_drift(report)

async def overall_drift(report, threshold=0.3):
    drift_ratio = report["drift_detected"].mean()
    return drift_ratio > threshold


async def detect_data_drift(reference_df, current_df, p_threshold=0.05) -> pd.DataFrame:
    drift_report = []

    for col in reference_df.columns:

        _, p_value = ks_2samp(reference_df[col], current_df[col])

        drift = p_value < p_threshold

        drift_report.append({
            "feature": col,
            "p_value": p_value,
            "drift_detected": drift
        })

    return pd.DataFrame(drift_report)

    