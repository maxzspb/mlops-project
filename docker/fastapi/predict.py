# app/predict.py
import logging
from typing import List
from fastapi import APIRouter, HTTPException, Body
import numbers
import pandas as pd
from schemas import PredictRequest, PredictResponse
from models import MODELS, EXPECTED_FEATURES, map_incoming_to_expected
logger = logging.getLogger("predict")
router = APIRouter()

def validate_request(req: PredictRequest):
    if not req.columns or not isinstance(req.columns, list):
        raise HTTPException(status_code=400, detail="`columns` must be a non-empty list of strings")
    if not req.data or not isinstance(req.data, list):
        raise HTTPException(status_code=400, detail="`data` must be a non-empty list of rows")

    n_cols = len(req.columns)
    for i, row in enumerate(req.data):
        if not isinstance(row, list):
            raise HTTPException(status_code=400, detail=f"Row {i} is not a list")
        if len(row) != n_cols:
            raise HTTPException(status_code=400, detail=f"Row {i} has {len(row)} values but expected {n_cols}")
        for j, v in enumerate(row):
            if not (isinstance(v, numbers.Number) or isinstance(v, bool)):
                raise HTTPException(status_code=400, detail=f"Row {i} col {j} is not numeric: {v}")

def _predict(req: PredictRequest, model_key: str):
    if model_key not in MODELS or MODELS[model_key] is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found (not loaded)")
    model = MODELS[model_key]
    validate_request(req)

    try:
        df = pd.DataFrame(req.data, columns=req.columns)
    except Exception as e:
        logger.exception("Bad input for DataFrame creation")
        raise HTTPException(status_code=400, detail=f"Bad input: {e}")

    expected = EXPECTED_FEATURES.get(model_key)
    if expected:
        mapping = map_incoming_to_expected(list(df.columns), expected)
        if mapping:
            df = df.rename(columns=mapping)
            logger.info(f"Renamed incoming columns by heuristic mapping: {mapping}")
        if df.shape[1] == len(expected) and set(df.columns) != set(expected):
            df.columns = expected
        if set(df.columns).issuperset(set(expected)):
            df = df[expected]
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Input features do not match expected features for model '{model_key}'. "
                    f"Expected {len(expected)} features: {expected}. Got {df.shape[1]} features: {list(df.columns)}. "
                    f"Use /models/{model_key}/features or send positional vector with same length."
                )
            )

    try:
        preds = model.predict(df)
    except Exception as e:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    try:
        pred_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
    except Exception:
        pred_list = list(preds)

    return PredictResponse(predictions=pred_list)

# endpoints
@router.post("/predict/iris", response_model=PredictResponse, summary="Predict Iris model")
def predict_iris(req: PredictRequest = Body(...)):
    return _predict(req, "iris")

@router.post("/predict/wine", response_model=PredictResponse, summary="Predict Wine model")
def predict_wine(req: PredictRequest = Body(...)):
    return _predict(req, "wine")

@router.post("/predict/cancer", response_model=PredictResponse, summary="Predict Breast Cancer model")
def predict_cancer(req: PredictRequest = Body(...)):
    return _predict(req, "cancer")

@router.get("/models/{model_key}/features", summary="Get expected features for model")
def get_features(model_key: str):
    if model_key not in EXPECTED_FEATURES:
        raise HTTPException(status_code=404, detail="Unknown model_key")
    return {"model_key": model_key, "expected_features": EXPECTED_FEATURES[model_key]}
