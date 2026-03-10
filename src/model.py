import logging
import time
import joblib
import pandas as pd
from loguru import logger
from src.schema import Data


class ModelPredictor:
    def __init__(
        self, 
        path_diabetes: str = "models/diabetes_model.pkl", 
        path_scaler: str = "models/scaler.pkl"
    ):
        self.model = joblib.load(path_diabetes)
        self.scaler = joblib.load(path_scaler)

    def predict(self, data: Data):
        start_time = time.time()
        df = pd.DataFrame(data.data, columns=data.columns)
        end_convertdf = time.time()
        logger.info(f"convert df time: {end_convertdf-start_time}")
        df = self.scaler.transform(df)
        end_scaler = time.time()
        logger.info(f"scaler time: {end_scaler-end_convertdf}")
        y_pred = self.model.predict(df)
        end_pred = time.time()
        logger.info(f"predict time: {end_pred-end_scaler}")
        return {
            "id": data.id,
            "predictions": y_pred.tolist(),
        }