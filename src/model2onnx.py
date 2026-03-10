import mlflow
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import numpy as np


def model_to_onnx(model_name, model_version):
    mlflow.set_tracking_uri("http://0.0.0.0:5000")
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    # num_features = 8  # Example: replace this with your model's input feature size
    # input_data = np.array([[0, 94, 0, 0, 0, 0.0, 0.256, 25]], dtype=np.float32)
    # Specify the initial input types for the model
    initial_types = [('input_data', FloatTensorType((None, 8)))]


    # Convert to ONNX
    onnx_model = onnxmltools.convert.convert_xgboost(model, initial_types=initial_types)

    # Save the ONNX model to disk
    onnxmltools.utils.save_model(onnx_model, "models/model.onnx")


    print("ONNX model saved as model.onnx")

if __name__ == "__main__":
    model_to_onnx(model_name="xgb", model_version="2")