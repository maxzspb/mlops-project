import os
from datetime import timedelta
from feast import Entity, Field, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64

minio_endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:9000")

# Источник данных
iris_source = FileSource(
    path="s3://feast-data/iris.parquet",
    s3_endpoint_override=minio_endpoint,  # <--- ТЕПЕРЬ ЗДЕСЬ ДИНАМИКА
    timestamp_field="event_timestamp",
)

# ... остальной код без изменений (Entity, FeatureView) ...
flower = Entity(name="flower_id", value_type=ValueType.INT64, description="ID цветка")

iris_view = FeatureView(
    name="iris_stats",
    entities=[flower],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="target", dtype=Int64),
    ],
    source=iris_source,
)