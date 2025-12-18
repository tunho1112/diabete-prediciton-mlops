import joblib
from kfp.components import create_component_from_func

def load_model_from_local(model_path: str):
    """
    Function to load a trained model from the local filesystem.
    """
    try:
        # Load the model using joblib
        model = joblib.load(model_path)
        print("Model loaded successfully from", model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

load_model_op = create_component_from_func(load_model_from_local, base_image='python:3.8-slim')
