from kfp import dsl
from kfp import compiler

# Твой образ с библиотеками (Feast, MLflow, Sklearn, Boto3)
BASE_IMAGE = "feast-trainer:v3" 

# --- КОМПОНЕНТ 1: ETL & Feast Sync ---
# Объединим подготовку данных и заливку в Redis, чтобы не гонять данные между подами
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
    
    # 1. Настройка окружения
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

    # 2. Сохранение в MinIO (Offline Store)
    s3 = boto3.client('s3', endpoint_url=minio_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    try:
        s3.create_bucket(Bucket="feast-data")
    except Exception:
        pass

    out_buffer = BytesIO()
    df.to_parquet(out_buffer, index=False)
    s3.put_object(Bucket="feast-data", Key="iris.parquet", Body=out_buffer.getvalue())
    print("✅ Data saved to MinIO")

    # 3. Настройка Feast "на лету"
    # Мы перезаписываем конфиг внутри контейнера правильными адресами кластера
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
    
    # 4. Feast Apply & Materialize
    print("--- [Step 2] Syncing to Redis ---")
    if os.system("feast apply") != 0: raise RuntimeError("Feast Apply Failed")
    
    # Материализуем данные (из S3 в Redis)
    if os.system(f"feast materialize-incremental {datetime.now().isoformat()}") != 0: 
        raise RuntimeError("Materialize Failed")
        
    print("✅ Feast synced successfully")


# --- КОМПОНЕНТ 2: Обучение ---
@dsl.component(base_image=BASE_IMAGE)
def train_op(
    minio_url: str,
    mlflow_url: str,
    access_key: str,
    secret_key: str
) -> str: # Возвращает URI модели (s3://...)
    import os
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from mlflow.models.signature import infer_signature
    from feast import FeatureStore
    
    # Env setup
    os.environ["AWS_ENDPOINT_URL"] = minio_url
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_url
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    os.environ["AWS_REGION"] = "us-east-1"
    os.system("pip install scikit-learn==1.3.2 mlflow==2.8.1 'pydantic<2.0.0' numpy<2.0.0")
    # Feast Config (Offline only)
    repo_path = "/app/feature_repo"
    yaml_content = f"""
project: iris_project
registry: s3://feast-data/registry.db
provider: local
offline_store:
    type: file
"""
    with open(f"{repo_path}/feature_store.yaml", "w") as f:
        f.write(yaml_content)
        
    fs = FeatureStore(repo_path=repo_path)
    
    # 1. Получение данных через Feast
    print("--- [Step 3] Fetching Training Data ---")
    entity_df = pd.DataFrame.from_dict({
        "flower_id": np.arange(150),
        "event_timestamp": [pd.Timestamp.now()] * 150
    })
    
    df = fs.get_historical_features(
        entity_df=entity_df,
        features=[
            "iris_stats:sepal_length",
            "iris_stats:sepal_width",
            "iris_stats:petal_length",
            "iris_stats:petal_width",
            "iris_stats:target",
        ],
    ).to_df().dropna()
    
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].astype(np.float32)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. MLflow Training
    print("--- [Step 4] Training Model ---")
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment("kserve-experiment")
    
    model_name = "IrisKServeModel"
    with mlflow.start_run() as run:
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, clf.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        
        signature = infer_signature(X_train, clf.predict(X_train))
        mlflow.sklearn.log_model(clf, artifact_path="model", registered_model_name=model_name, signature=signature)
        
        # Получаем URI модели (s3://mlflow/...)
        model_uri = mlflow.get_artifact_uri("model")
        print(f"✅ Model saved at {model_uri}")
        return model_uri

# --- КОМПОНЕНТ 3: KServe Deploy ---
@dsl.component(
    base_image="python:3.9", # Тут можно легкий образ, главное kubernetes lib
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
    
    config.load_incluster_config()
    api = client.CustomObjectsApi()
    
    # Манифест InferenceService (аналог Seldon Model)
    isvc = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
            "annotations": {
                "sidecar.istio.io/inject": "false" # Иногда нужно отключать, если конфликты
            }
        },
        "spec": {
            "predictor": {
                "serviceAccountName": "kserve-sa", # Наш SA с доступом к MinIO
                "model": {
                    "modelFormat": {
                        "name": "sklearn" # Используем стандартный sklearn сервер KServe
                    },
                    "storageUri": model_uri # s3://mlflow/...
                }
            }
        }
    }
    
    try:
        # Пытаемся создать
        api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            body=isvc
        )
        print("✅ InferenceService created!")
    except client.exceptions.ApiException as e:
        if e.status == 409: # Уже существует -> обновляем
            print("🔄 InferenceService exists, patching...")
            # Получаем resourceVersion для корректного патча (или используем merge-patch)
            existing = api.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=model_name
            )
            isvc["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
            
            api.replace_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=model_name,
                body=isvc
            )
            print("✅ InferenceService updated!")
        else:
            raise e

# --- ПАЙПЛАЙН ---
@dsl.pipeline(
    name='kserve-mlops-pipeline',
    description='Feast -> MLflow -> KServe'
)
def kserve_pipeline():
    # Конфигурация (в идеале - через Secrets)
    minio = "http://minio-service.kubeflow.svc.cluster.local:9000"
    mlflow = "http://mlflow.kubeflow.svc.cluster.local:5000"
    redis = "redis-master.kubeflow.svc.cluster.local:6379"
    
    # 1. Данные
    etl_task = etl_and_feast_op(
        minio_url=minio,
        redis_url=redis,
        access_key="minio",
        secret_key="minio123"
    )
    
    # 2. Обучение (зависит от ETL, хотя KFP может сам понять, если бы мы передавали артефакты)
    train_task = train_op(
        minio_url=minio,
        mlflow_url=mlflow,
        access_key="minio",
        secret_key="minio123"
    )
    train_task.after(etl_task)
    
    # 3. Деплой
    deploy_task = kserve_deploy_op(
        model_uri=train_task.output,
        model_name="iris-model",
        namespace="kubeflow"
    )

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=kserve_pipeline,
        package_path='kserve_pipeline.yaml'
    )
    print("✅ Compiled!")
