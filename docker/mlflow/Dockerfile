FROM python:3.10-slim

WORKDIR /app

# minimal system deps
RUN apt-get update && apt-get install -y build-essential gcc git curl && apt-get clean

# install mlflow and s3 deps
RUN pip install --no-cache-dir mlflow boto3 sqlalchemy psycopg2-binary

# copy a tiny entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 5000
CMD ["/app/entrypoint.sh"]
