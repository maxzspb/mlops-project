FROM python:3.12.3-slim

WORKDIR /app
RUN apt-get update && apt-get install -y build-essential gcc && apt-get clean
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONPATH=/app

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]