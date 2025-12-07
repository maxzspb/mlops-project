from kfp import dsl
from kfp import compiler

# Твой образ с библиотеками (Feast, MLflow, Sklearn, Boto3)
BASE_IMAGE = "feast-trainer:v3" 

# --- КОМПОНЕНТ 1: ETL & Feast Sync (тот же) ---
@dsl.component(base_image=BASE_IMAGE)
def etl_and_feast_op(
    minio_url: str,
    redis_url: str,
    access_key: str,
    secret_key: str
):
    import os
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris
    import boto3
    from io import BytesIO
    from datetime import datetime
    
    os.environ["AWS_ENDPOINT_URL"] = minio_url
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    os.environ["AWS_REGION"] = "us-east-1"
    
    print("--- [Step 1] Generating Data ---")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df.columns = [c.replace(' ', '_') for c in df.columns]
    df['target'] = iris.target
    df['flower_id'] = np.arange(len(df))
    df['event_timestamp'] = pd.Timestamp.now()

    s3 = boto3.client('s3', endpoint_url=minio_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    try:
        s3.create_bucket(Bucket="feast-data")
    except Exception:
        pass

    out_buffer = BytesIO()
    df.to_parquet(out_buffer, index=False)
    s3.put_object(Bucket="feast-data", Key="iris.parquet", Body=out_buffer.getvalue())
    print("✅ Data saved to MinIO")

    repo_path = "/app/feature_repo"
    yaml_content = f"""
project: iris_project
registry: s3://feast-data/registry.db
provider: local
online_store:
    type: redis
    connection_string: "{redis_url}"
offline_store:
    type: file
"""
    with open(f"{repo_path}/feature_store.yaml", "w") as f:
        f.write(yaml_content)
    
    os.chdir(repo_path)
    
    print("--- [Step 2] Syncing to Redis ---")
    if os.system("feast apply") != 0: raise RuntimeError("Feast Apply Failed")
    if os.system(f"feast materialize-incremental {datetime.now().isoformat()}") != 0:
        raise RuntimeError("Materialize Failed")
    print("✅ Feast synced successfully")


# --- НОВЫЙ КОМПОНЕНТ: запускаем PyTorchJob CRD и ждём завершения ---
# Требует: serviceaccount pipeline'а с правами на CRD pytorchjobs (kubeflow.org)
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["kubernetes", "mlflow", "requests"]
)
def train_with_pytorchjob_op(
    minio_url: str,
    mlflow_url: str,
    access_key: str,
    secret_key: str,
    pytorch_image: str = "pytorch-kfp:latest",
    namespace: str = "kubeflow",
    worker_replicas: int = 1,
    gpu_per_replica: int = 1,
    timeout_seconds: int = 3600
) -> str:
    """
    Создаёт PyTorchJob CRD (kubeflow.org/v1 pytorchjobs), ждёт завершения,
    затем находит последний run в MLflow (эксперимент 'kserve-experiment') и возвращает model_uri.
    Предполагается, что внутри образа pytorch_image есть скрипт train.py, который логирует модель в MLflow.
    """
    import time
    import uuid
    import os
    from kubernetes import client, config
    import mlflow
    from mlflow.tracking import MlflowClient

    # env для доступа к MinIO/MLflow в training контейнерах
    env_vars_for_train = [
        {"name": "AWS_ENDPOINT_URL", "value": minio_url},
        {"name": "AWS_ACCESS_KEY_ID", "value": access_key},
        {"name": "AWS_SECRET_ACCESS_KEY", "value": secret_key},
        {"name": "MLFLOW_S3_ENDPOINT_URL", "value": minio_url},
        {"name": "MLFLOW_TRACKING_URI", "value": mlflow_url},
    ]

    # Имя job
    job_name = f"pytorch-train-{str(uuid.uuid4())[:8]}"

    # PyTorchJob body (примерный, адаптируй под твой образ / команду)
    pytorchjob = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "restartPolicy": "OnFailure",
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "pytorch",
                                    "image": pytorch_image,
                                    # предполагаем, что в образе есть /app/train.py
                                    "command": ["python", "/app/train.py"],
                                    "env": env_vars_for_train,
                                    "resources": {
                                        "limits": {"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": str(gpu_per_replica)},
                                        "requests": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": str(gpu_per_replica)}
                                    }
                                }
                            ]
                        }
                    }
                },
                "Worker": {
                    "replicas": worker_replicas,
                    "restartPolicy": "OnFailure",
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "pytorch",
                                    "image": pytorch_image,
                                    "command": ["python", "/app/train.py", "--worker"],
                                    "env": env_vars_for_train,
                                    "resources": {
                                        "limits": {"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": str(gpu_per_replica)},
                                        "requests": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": str(gpu_per_replica)}
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
    }

    # Создаём CRD объект
    print(f"▶ Creating PyTorchJob {job_name} in namespace {namespace} ...")
    try:
        config.load_incluster_config()
    except Exception:
        # для локального теста (kubectl proxy / kubeconfig)
        config.load_kube_config()

    api = client.CustomObjectsApi()
    group = "kubeflow.org"
    version = "v1"
    plural = "pytorchjobs"

    api.create_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural, body=pytorchjob)
    print("✅ PyTorchJob created, waiting for completion...")

    # wait loop
    start = time.time()
    while True:
        obj = api.get_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural, name=job_name)
        status = obj.get("status", {}) or {}
        conditions = status.get("conditions", []) or []

        # Debug print:
        print("status.conditions:", conditions)

        # check for succeeded
        succeeded = any((c.get("type") == "Succeeded" and c.get("status") == "True") for c in conditions)
        failed = any((c.get("type") == "Failed" and c.get("status") == "True") for c in conditions)

        if succeeded:
            print("✅ PyTorchJob succeeded.")
            break
        if failed:
            raise RuntimeError(f"PyTorchJob {job_name} failed. status: {status}")

        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timeout waiting for PyTorchJob {job_name} to finish (waited {timeout_seconds}s)")

        time.sleep(10)

    # После успешного завершения — получаем последний run модели в MLflow
    print("▶ Querying MLflow for latest run in experiment 'kserve-experiment' ...")
    mlflow.set_tracking_uri(mlflow_url)
    client_ml = MlflowClient(tracking_uri=mlflow_url)

    exp = client_ml.get_experiment_by_name("kserve-experiment")
    if exp is None:
        raise RuntimeError("Experiment 'kserve-experiment' not found in MLflow. Убедись, что train.py логирует туда.")

    exp_id = exp.experiment_id
    runs = client_ml.search_runs([exp_id], order_by=["attribute.start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError("MLflow: no runs found in experiment 'kserve-experiment' after training.")

    latest_run = runs[0]
    artifact_uri = latest_run.info.artifact_uri  # обычно s3://mlflow/0/<run-id>
    model_uri = artifact_uri.rstrip("/") + "/model"  # предполагаем, что train.py логирует артефакт в path "model"

    print(f"✅ Found model at: {model_uri}")
    return model_uri


# --- КОМПОНЕНТ 3: KServe Deploy (тот же) ---
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["kubernetes"]
)
def kserve_deploy_op(
    model_uri: str,
    model_name: str = "iris-classifier",
    namespace: str = "kubeflow"
):
    from kubernetes import client, config
    import json
    
    print(f"🚀 Deploying {model_uri} to KServe...")
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    api = client.CustomObjectsApi()

    isvc = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
            "annotations": {"sidecar.istio.io/inject": "false"}
        },
        "spec": {
            "predictor": {
                "serviceAccountName": "kserve-sa",
                "model": {
                    "modelFormat": {"name": "pytorch"}, # или "sklearn" / "onnx" — укажи по формату модели
                    "storageUri": model_uri
                }
            }
        }
    }

    try:
        api.create_namespaced_custom_object(group="serving.kserve.io", version="v1beta1", namespace=namespace, plural="inferenceservices", body=isvc)
        print("✅ InferenceService created!")
    except client.exceptions.ApiException as e:
        if e.status == 409:
            print("🔄 InferenceService exists, patching...")
            existing = api.get_namespaced_custom_object(group="serving.kserve.io", version="v1beta1", namespace=namespace, plural="inferenceservices", name=model_name)
            isvc["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
            api.replace_namespaced_custom_object(group="serving.kserve.io", version="v1beta1", namespace=namespace, plural="inferenceservices", name=model_name, body=isvc)
            print("✅ InferenceService updated!")
        else:
            raise


# --- ПАЙПЛАЙН ---
@dsl.pipeline(
    name='kserve-mlops-pipeline-pytorch',
    description='Feast -> PyTorchJob -> MLflow -> KServe'
)
def kserve_pipeline_pytorch():
    # Конфигурация (в идеале - через Secrets)
    minio = "http://minio-service.kubeflow.svc.cluster.local:9000"
    mlflow = "http://mlflow.kubeflow.svc.cluster.local:5000"
    redis = "redis-master.kubeflow.svc.cluster.local:6379"
    pytorch_image = "pytorch-kfp:latest"  # образ, который запускает train.py и логирует в MLflow

    # 1. Данные
    etl_task = etl_and_feast_op(
        minio_url=minio,
        redis_url=redis,
        access_key="minio",
        secret_key="minio123"
    )
    
    # 2. Тренировка через PyTorchJob (зависит от ETL)
    train_task = train_with_pytorchjob_op(
        minio_url=minio,
        mlflow_url=mlflow,
        access_key="minio",
        secret_key="minio123",
        pytorch_image=pytorch_image,
        namespace="kubeflow",
        worker_replicas=1,
        gpu_per_replica=1,
        timeout_seconds=3600
    )
    train_task.after(etl_task)

    # 3. Деплой в KServe
    deploy_task = kserve_deploy_op(
        model_uri=train_task.output,
        model_name="iris-model",
        namespace="kubeflow"
    )

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=kserve_pipeline_pytorch,
        package_path='kserve_pipeline_pytorch.yaml'
    )
    print("✅ Compiled!")
