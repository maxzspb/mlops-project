# Dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# CPU wheels of torch
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Common ML / infra libs
RUN pip install --no-cache-dir \
    mlflow \
    boto3 \
    pandas \
    numpy \
    scikit-learn \
    feast \
    kubernetes \
    prometheus-client

WORKDIR /app
COPY train.py /app/

ENV SIMULATE_GPU=1
ENTRYPOINT ["python", "/app/train.py"]
