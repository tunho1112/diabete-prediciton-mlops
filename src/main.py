from fastapi import FastAPI

from src.model import ModelPredictor
from src.schema import Data
from loguru import logger
import time
# trace log 
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import get_tracer_provider, set_tracer_provider
# prometheus metric
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from prometheus_client import start_http_server


set_tracer_provider(
    TracerProvider(resource=Resource.create({SERVICE_NAME: "ai-serving"}))
)
tracer = get_tracer_provider().get_tracer("service", "0.0.1")
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
span_processor = BatchSpanProcessor(jaeger_exporter)
get_tracer_provider().add_span_processor(span_processor)

# prometheus metric
# Start Prometheus client
start_http_server(port=8099, addr="0.0.0.0")
# Service name is required for most backends
resource = Resource(attributes={SERVICE_NAME: "serving-service"})
# Exporter to export metrics to Prometheus
reader = PrometheusMetricReader()
# Meter is responsible for creating and recording metrics
provider = MeterProvider(resource=resource, metric_readers=[reader])
set_meter_provider(provider)
meter = metrics.get_meter("ai-serving", "0.0.1")

# Create your first counter
counter = meter.create_counter(
    name="ai_request_counter",
    description="Number of AI requests"
)

histogram = meter.create_histogram(
    name="ai_response_histogram",
    description="AI response histogram",
    unit="seconds",
)

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Model prediction API is up new."}

@app.post("/predict")
def predict(input_data: Data):
    """
    Example request:
    {
    "id": "123",
    "data": [[6,148,72,35,0,33.6,0.627,50]],
    "columns": ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    }
    """
    starting_time = time.time()
    with tracer.start_as_current_span("processors") as processors:
        with tracer.start_as_current_span(
            "model-loader", links=[trace.Link(processors.get_span_context())]
        ):
            print("run model-loader ...")
            predictor = ModelPredictor()
        with tracer.start_as_current_span(
            "predictor", links=[trace.Link(processors.get_span_context())]
        ):
            print("run predictor ...")
            result = predictor.predict(input_data)
        
        # Labels for all metrics
        label = {"api": "/predict"}

        # Increase the counter
        counter.add(1, label)

        # Mark the end of the response
        ending_time = time.time()
        elapsed_time = ending_time - starting_time

        # Add histogram
        logger.info("elapsed time: ", elapsed_time)
        logger.info(elapsed_time)
        histogram.record(elapsed_time, label)
        return result