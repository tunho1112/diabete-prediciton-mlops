FROM python:3.9.13

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./models/ ./models
COPY kserve_predict_onnx.py kserve_predict_onnx.py 

EXPOSE 8080

CMD ["python", "kserve_predict_onnx.py"]