#!/usr/bin/env bash
set -e

echo "Waiting for MinIO and Postgres..."
sleep 3

export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
if [ -n "${AWS_ENDPOINT_URL}" ]; then
  export AWS_S3_ENDPOINT_URL=${AWS_ENDPOINT_URL}
  export MLFLOW_S3_ENDPOINT_URL=${AWS_ENDPOINT_URL}
fi

python - <<PY
import os, boto3, time
ep = os.environ.get("AWS_ENDPOINT_URL")
s3 = boto3.client('s3', endpoint_url=ep,
                  aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                  aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
bucket = os.environ.get("MLFLOW_S3_BUCKET", "mlflow")
for _ in range(10):
    try:
        s3.head_bucket(Bucket=bucket)
        print("Bucket exists:", bucket)
        break
    except Exception as e:
        try:
            s3.create_bucket(Bucket=bucket)
            print("Created bucket:", bucket)
            break
        except Exception as e2:
            print("Waiting for S3... retry", e2)
            time.sleep(2)
PY


exec mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" \
  --artifacts-destination "s3://mlflow/" \
  --serve-artifacts \
  --host "${MLFLOW_HOST:-0.0.0.0}" \
  --port "${MLFLOW_PORT:-5000}"