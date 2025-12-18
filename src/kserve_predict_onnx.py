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
# kserve
from kserve import Model, ModelServer
from kserve import InferRequest, InferResponse

import json 
from typing import Dict



class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

class ModelPredictor(Model):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.model_path="models/model.onnx"
        self.scaler=joblib.load("models/scaler.pkl")
        self.ready=False
        logging.info("scaler loaded")
    def load(self):
        self.ready = True
        
    def predict(self, input_data: InferRequest, headers: Dict[str, str] = None):
        input_data = input_data["input_data"]
        logging.info("==1===========")
        onnx_model = onnx.load(self.model_path)  # Change to your model file name
        logging.info("==2===")
        onnx.checker.check_model(onnx_model)  # Verify the model
        # Create an ONNX runtime session
        session = ort.InferenceSession(self.model_path)  # Change to your model file name
        # self.model = session
        start_time = time.time()
        input_data = self.scaler.transform(input_data).astype(np.float32)
        # Load the ONNX model

        # Run inference
        inputs = {session.get_inputs()[0].name: input_data}
        predictions = session.run(None, inputs)
        # y_pred=self.model.predict(df)
        end_pred = time.time()
        preds = json.dumps(predictions, cls=NumpyArrayEncoder)
        print(preds)
        return predictions
    
if __name__ == "__main__":
    # val = pd.read_csv("data/val.csv")
    # predictor = ModelPredictor()
    # val_X = val.drop(columns=['Outcome'])
    # val_y = val['Outcome']
    # y_pred = predictor.predict(val_X)
    # print(y_pred[0])

    model = ModelPredictor("diabetest-model")
    model.load()
    ModelServer().start([model])