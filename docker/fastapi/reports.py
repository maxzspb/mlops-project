# app/reports.py
import logging
import tempfile
from fastapi import APIRouter, HTTPException, Body, Response
import pandas as pd

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently import BinaryClassification, MulticlassClassification
# reports.py
from schemas import ReportRequest
from models import MODELS, _load_builtin_reference, EXPECTED_FEATURES, map_incoming_to_expected

import mlflow

logger = logging.getLogger("reports")
router = APIRouter()

def _has_real_values(df: pd.DataFrame, col: str) -> bool:
    return (col in df.columns) and df[col].notna().any()

@router.post("/reports/{model_key}", summary="Generate Evidently report for model and log to MLflow")
def generate_report(model_key: str, body: ReportRequest = Body(...)):
    if model_key not in MODELS or MODELS[model_key] is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not loaded")

    # reference
    try:
        ref_df = _load_builtin_reference(model_key)
    except Exception:
        ref_df = pd.DataFrame()

    # current
    if body.data and body.columns:
        try:
            cur_df = pd.DataFrame(body.data, columns=body.columns)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad current data: {e}")
    else:
        if not ref_df.empty:
            cur_df = ref_df.sample(min(len(ref_df), 30), random_state=42).reset_index(drop=True)
        else:
            raise HTTPException(status_code=400, detail="No current data provided and no reference dataset available for demo.")

    expected = EXPECTED_FEATURES.get(model_key, list(cur_df.columns))
    mapping = map_incoming_to_expected(list(cur_df.columns), expected)
    if mapping:
        cur_df = cur_df.rename(columns=mapping)
        logger.info(f"Renamed current columns by heuristic mapping: {mapping}")

    if cur_df.shape[1] == len(expected) and set(cur_df.columns) != set(expected):
        logger.info("Renaming current dataframe columns to expected feature names (positional mapping).")
        cur_df.columns = expected

    if not ref_df.empty:
        inter = set(ref_df.columns).intersection(set(cur_df.columns))
        if not inter:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Columns in current data do not match reference dataset. "
                    f"Reference features: {list(ref_df.columns)}. "
                    f"Current provided: {list(cur_df.columns)}. "
                )
            )

    # predictions (optional)
    model = MODELS[model_key]
    cur_has_pred = False
    ref_has_pred = False
    try:
        if set(expected).issubset(set(cur_df.columns)):
            preds_cur = model.predict(cur_df[expected])
        else:
            preds_cur = model.predict(cur_df.select_dtypes(include=["number"]))
        cur_df = cur_df.copy()
        cur_df["prediction"] = preds_cur
        cur_has_pred = True
    except Exception:
        logger.info("Could not compute predictions for current dataset; continuing without predictions for report.")

    if not ref_df.empty:
        try:
            if set(expected).issubset(set(ref_df.columns)):
                preds_ref = model.predict(ref_df[expected])
            else:
                preds_ref = model.predict(ref_df.select_dtypes(include=["number"]))
            ref_df = ref_df.copy()
            ref_df["prediction"] = preds_ref
            ref_has_pred = True
        except Exception:
            logger.info("Could not compute predictions for reference dataset; will not include predictions in reference.")

    if cur_has_pred and not ref_has_pred:
        cur_df = cur_df.drop(columns=["prediction"], errors="ignore")
        cur_has_pred = False

    if not ref_df.empty:
        for col in ref_df.columns:
            if col not in cur_df.columns:
                cur_df[col] = pd.NA
        for c in list(cur_df.columns):
            if c not in ref_df.columns and c != "prediction":
                logger.info(f"Dropping current-only column '{c}' to match reference (will not be used for drift).")
                cur_df = cur_df.drop(columns=[c], errors="ignore")

    classification_mapping = None
    if not ref_df.empty and _has_real_values(ref_df, "target") and _has_real_values(ref_df, "prediction") and _has_real_values(cur_df, "prediction"):
        unique_targets = pd.Series(ref_df["target"].dropna()).unique().tolist()
        try:
            n_unique = len(unique_targets)
        except Exception:
            n_unique = 0
        if n_unique <= 2:
            classification_mapping = BinaryClassification(target="target", prediction_labels="prediction")
        else:
            classification_mapping = MulticlassClassification(target="target", prediction_labels="prediction")

    data_def = DataDefinition(classification=[classification_mapping]) if classification_mapping is not None else DataDefinition()

    metrics = [DataDriftPreset()]
    if classification_mapping is not None:
        metrics.append(ClassificationPreset())

    report = Report(metrics=metrics)

    ref_dataset = Dataset.from_pandas(ref_df, data_definition=data_def) if not ref_df.empty else None
    cur_dataset = Dataset.from_pandas(cur_df, data_definition=data_def)

    try:
        if ref_dataset is not None:
            evaluation_result = report.run(reference_data=ref_dataset, current_data=cur_dataset)
        else:
            evaluation_result = report.run(current_data=cur_dataset)
    except Exception as e:
        logger.exception("Evidently report run failed")
        raise HTTPException(status_code=500, detail=f"Evidently report run failed: {e}")

    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    tmp_path = tmp.name
    tmp.close()
    evaluation_result.save_html(tmp_path)

    try:
        with mlflow.start_run(run_name=f"{model_key}_drift_report") as run:
            run_id = run.info.run_id
            artifact_path = f"reports/{model_key}"
            mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
            logger.info(f"Logged report to MLflow run_id={run_id} artifact_path={artifact_path}")
    except Exception:
        logger.exception("Failed to log report to MLflow")
        run_id = None

    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()

    headers = {}
    if run_id:
        headers["X-MLFLOW-RUN-ID"] = run_id

    return Response(content=html, media_type="text/html", headers=headers)
