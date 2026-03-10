import tritonclient.http as httpclient
import numpy as np
import pandas as pd 
import joblib



def triton_inference(model_name, model_version, input_data):
    """
    Perform inference using Triton Inference Server.
    
    Args:
    - model_name: The name of the model in Triton.
    - model_version: The version of the model in Triton.
    - input_data: The input data to be sent for inference.
    
    Returns:
    - The model output prediction.
    """
    # URL for Triton Inference Server
    url = "localhost:8000"  # Change to your Triton server address if different

    # Initialize the Triton Inference Server client
    client = httpclient.InferenceServerClient(url=url)

    # Prepare the input for the model
    inputs = []
    inputs.append(httpclient.InferInput("input_data", input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)

    # Prepare the output for the model
    outputs = []
    outputs.append(httpclient.InferRequestedOutput("label"))

    # Run the inference
    results = client.infer(model_name=model_name,
                           model_version=model_version,
                           inputs=inputs,
                           outputs=outputs)

    # Extract the prediction result
    output_data = results.as_numpy("label")
    return output_data

if __name__ == "__main__":
    # batch_size = 128
    # val = pd.read_csv("data/val.csv")
    # val_X = val.drop(columns=['Outcome'])
    # # print(val_X)
    scaler=joblib.load("models/scaler.pkl")
    val_X = np.array([
        [6,103,66,0,0,24.3,0.249,29], 
        [3,89,74,16,85,30.4,0.551,38],
        [0,135,94,46,145,40.6,0.284,26],
        [4,116,72,12,87,22.1,0.463,37],
        [3,173,78,39,185,33.8,0.97,31]], dtype=np.float32)

    df = scaler.transform(val_X)
    # Example input data: 1 sample with 8 features
    # input_data = np.array([[1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9], [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8]], dtype=np.float32)
    input_data = np.array(np.array(df[:128]), dtype=np.float32)
    # Triton model details
    model_name = "xgb"  # Name of the model in Triton
    model_version = "1"  # The version of the model

    # Get the prediction from Triton
    prediction = triton_inference(model_name, model_version, input_data)

    # Print the prediction
    print(f"Prediction: {prediction}")
