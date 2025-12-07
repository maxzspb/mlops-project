# app/models.py
import os
import logging
import re
from typing import Dict, List
import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

logger = logging.getLogger("models")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URIS = {
    "iris": "models:/IrisRandomForest/Production",
    "wine": "models:/WineRandomForest/Production",
    "cancer": "models:/CancerLogisticRegression/Production"
}

MODELS: Dict[str, object] = {}
EXPECTED_FEATURES: Dict[str, List[str]] = {}

def _norm_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_").lower()
    return s or "f0"

def build_expected_features():
    global EXPECTED_FEATURES
    ds = load_iris()
    EXPECTED_FEATURES["iris"] = [_norm_name(n) for n in ds.feature_names] if hasattr(ds, "feature_names") else [f"f{i}" for i in range(ds.data.shape[1])]

    ds = load_wine()
    EXPECTED_FEATURES["wine"] = [_norm_name(n) for n in ds.feature_names] if hasattr(ds, "feature_names") else [f"f{i}" for i in range(ds.data.shape[1])]

    ds = load_breast_cancer()
    EXPECTED_FEATURES["cancer"] = [_norm_name(n) for n in ds.feature_names] if hasattr(ds, "feature_names") else [f"f{i}" for i in range(ds.data.shape[1])]

def load_models():
    global MODELS
    build_expected_features()
    for key, uri in MODEL_URIS.items():
        try:
            MODELS[key] = mlflow.pyfunc.load_model(uri)
            logger.info(f"Loaded model '{key}' from '{uri}'")
        except Exception:
            MODELS[key] = None
            logger.exception(f"Failed to load model '{key}' from '{uri}'")

def _load_builtin_reference(model_key: str) -> pd.DataFrame:
    if model_key == "iris":
        ds = load_iris()
    elif model_key == "wine":
        ds = load_wine()
    elif model_key == "cancer":
        ds = load_breast_cancer()
    else:
        raise KeyError("no builtin reference")

    feature_names = EXPECTED_FEATURES.get(model_key)
    X = pd.DataFrame(ds.data, columns=feature_names if feature_names else None)
    if hasattr(ds, "target"):
        X["target"] = ds.target
    return X

def map_incoming_to_expected(in_cols: List[str], expected: List[str]) -> Dict[str, str]:
    mapping = {}
    expected_lower = [e.lower() for e in expected]
    exp_norm = [_norm_name(e) for e in expected]
    for inc in in_cols:
        inc_low = inc.lower()
        if inc_low in expected_lower:
            mapping[inc] = expected[expected_lower.index(inc_low)]
            continue
        inc_norm = _norm_name(inc)
        if inc_norm in exp_norm:
            mapping[inc] = expected[exp_norm.index(inc_norm)]
            continue
        candidates = [e for e in expected if (inc_low in e.lower()) or (e.lower() in inc_low)]
        if len(candidates) == 1:
            mapping[inc] = candidates[0]
            continue
    return mapping

# экспортируем полезные вещи
__all__ = [
    "MLFLOW_TRACKING_URI", "MODELS", "EXPECTED_FEATURES",
    "load_models", "_load_builtin_reference", "map_incoming_to_expected", "_norm_name"
]
