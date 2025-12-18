# from fastapi import FastAPI

# from predict import ModelPredictor
# from schema import Data



# app = FastAPI()
# predictor = ModelPredictor()


# @app.get("/")
# def read_root():
#     return {"message": "Model prediction API is up."}

# @app.post("/predict")
# def predict(input_data: Data):
#     """
#     Example request:
#     {
#     "id": "123",
#     "data": [[6,148,72,35,0,33.6,0.627,50]],
#     "columns": ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
#     }
#     """
#     result = predictor.predict(input_data)
#     return result