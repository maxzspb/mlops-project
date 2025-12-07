import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
from datetime import datetime, timedelta
import boto3
from io import BytesIO
from feast import FeatureStore

import logging
logging.basicConfig(level=logging.INFO)

# --- КОНФИГУРАЦИЯ ---
try:
    MINIO_ENDPOINT = os.environ["AWS_ENDPOINT_URL"]
    MLFLOW_URI = os.environ["MLFLOW_TRACKING_URI"]
except KeyError as e:
    raise RuntimeError(f"❌ Ошибка конфигурации! Не найдена переменная: {e}")

os.environ['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

print(f"🚀 Starting MLOps Pipeline")
print(f"📡 MinIO: {MINIO_ENDPOINT}")
print(f"📡 MLflow: {MLFLOW_URI}")

mlflow.set_tracking_uri(MLFLOW_URI)

# --- 1. ETL: ПОДГОТОВКА ДАННЫХ ---
print("\n--- [1/5] ETL: Preparing Data ---")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df.columns = [c.replace(' ', '_') for c in df.columns]
df['target'] = iris.target
df['flower_id'] = np.arange(len(df))
df['event_timestamp'] = pd.Timestamp.now()

s3 = boto3.client('s3', endpoint_url=MINIO_ENDPOINT)
try:
    s3.create_bucket(Bucket="feast-data")
except Exception:
    pass

out_buffer = BytesIO()
df.to_parquet(out_buffer, index=False)
s3.put_object(Bucket="feast-data", Key="iris.parquet", Body=out_buffer.getvalue())
print("✅ Data saved to MinIO")

# --- 2. FEAST APPLY ---
print("\n--- [2/5] Feast Apply (Registry Update) ---")
os.chdir("feature_repo") 
os.system("feast apply")

# --- 3. MATERIALIZE (Заливка в Redis) ---
print("\n--- [3/5] Materializing to Online Store (Redis) ---")
# Заливаем данные в Redis
exit_code = os.system(f"feast materialize-incremental {datetime.now().isoformat()}")
if exit_code != 0:
    raise RuntimeError("❌ Feast Materialize failed!")
print("✅ Data synced to Redis")

# --- 4. ОБУЧЕНИЕ ---
print("\n--- [4/5] Training Model (Offline Retrieval) ---")
fs = FeatureStore(repo_path=".")

entity_df = pd.DataFrame.from_dict({
    "flower_id": np.arange(150),
    "event_timestamp": [pd.Timestamp.now()] * 150
})

training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_stats:sepal_length",
        "iris_stats:sepal_width",
        "iris_stats:petal_length",
        "iris_stats:petal_width",
        "iris_stats:target",
    ],
).to_df().dropna()

print(f"Loaded {len(training_df)} rows from Offline Store")

X = training_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].astype(np.float32)
y = training_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_name = "IrisFeastModel"
with mlflow.start_run(run_name=model_name):
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    
    signature = infer_signature(X_train, preds)
    mlflow.sklearn.log_model(clf, artifact_path="model", registered_model_name=model_name, signature=signature)
    print(f"✅ Model trained! Accuracy: {acc}")

# --- 5. VERIFY ONLINE STORE ---
print("\n--- [5/5] Verifying Online Store (Smoke Test) ---")
try:
    # Пробуем достать фичи для цветка ID=96, как это будет делать Seldon
    features = fs.get_online_features(
        features=[
            "iris_stats:sepal_length",
            "iris_stats:sepal_width",
        ],
        entity_rows=[{"flower_id": 96}]
    ).to_dict()
    
    print("Retrieved features from Redis:", features)
    
    # Проверка: если список пустой или None - значит Redis пуст
    if not features['flower_id'] or features['sepal_length'][0] is None:
        raise RuntimeError("❌ Redis вернул пустые данные! Инференс не будет работать.")
        
    print("✅ Online Store check passed! Ready for Inference.")
    
except Exception as e:
    print(f"❌ Redis check failed: {e}")
