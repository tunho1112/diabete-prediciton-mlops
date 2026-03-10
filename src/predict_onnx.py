import pandas as pd 
import numpy as np
import onnx
import onnxruntime as ort
import yaml
import logging
import mlflow
import os 
import time
import joblib
from config import Config


class ModelPredictor:
    def __init__(self, model_path="models/model.onnx"):
        self.model_path=model_path
        self.scaler=joblib.load("models/scaler.pkl")
        logging.info("scaler loaded")
    def predict(self, input_data:np.ndarray):
        start_time = time.time()
        input_data = self.scaler.transform(input_data).astype(np.float32)
        # Load the ONNX model
        onnx_model = onnx.load(self.model_path)  # Change to your model file name
        onnx.checker.check_model(onnx_model)  # Verify the model
        # Create an ONNX runtime session
        session = ort.InferenceSession(self.model_path)  # Change to your model file name
        # Run inference
        inputs = {session.get_inputs()[0].name: input_data}
        predictions = session.run(None, inputs)
        # y_pred=self.model.predict(df)
        end_pred = time.time()
        return predictions
    
if __name__ == "__main__":
    val = pd.read_csv("data/val.csv")
    predictor = ModelPredictor()
    val_X = val.drop(columns=['Outcome'])
    val_y = val['Outcome']
    y_pred = predictor.predict(val_X)
    print(y_pred[0])