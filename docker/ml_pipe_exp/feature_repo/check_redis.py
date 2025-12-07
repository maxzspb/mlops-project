from feast import FeatureStore
import os

# Настраиваем, чтобы Feast знал, где искать Redis и MinIO
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
os.environ['AWS_ENDPOINT_URL'] = 'http://localhost:9000'

fs = FeatureStore(repo_path="feature_repo") 

print("--- Запрос в Redis (симуляция Seldon) ---")

# Представь, что это код внутри Seldon Model.
# Пришел запрос: "Дай прогноз для цветка №96"
features = fs.get_online_features(
    features=[
        "iris_stats:sepal_length",
        "iris_stats:sepal_width",
        "iris_stats:petal_length",
        "iris_stats:petal_width",
    ],
    entity_rows=[
        {"flower_id": 96},
        {"flower_id": 97}
    ]
).to_dict()

print("Ответ от Redis:")
print(features)