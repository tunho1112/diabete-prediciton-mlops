import pandas as pd 
import numpy as np
import onnx
import onnxruntime as ort
import time
import joblib
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
    def load(self):
        self.ready = True
        
    def predict(self, input_data: InferRequest, headers: Dict[str, str] = None):
        input_data = input_data["input_data"]
        onnx_model = onnx.load(self.model_path)  # Change to your model file name
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
        return preds
    
if __name__ == "__main__":
    model = ModelPredictor("diabetest-model")
    model.load()
    ModelServer().start([model])