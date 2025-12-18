FROM python:3.9

# Create a folder /app if it doesn't exist,
# the /app folder is the current working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
# Disable pip cache to shrink the image size a little bit,
# since it does not need to be re-installed
RUN pip install -r requirements.txt --no-cache-dir
# Copy models directory
COPY ./models /app/models

# Copy source code
COPY ./src /app/src

LABEL maintainer="minhtu"

EXPOSE 8080

CMD ["opentelemetry-instrument", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
