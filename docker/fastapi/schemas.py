# app/schemas.py
from typing import Any, List, Optional
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    columns: List[str] = Field(..., description="List of feature names (order matters)")
    data: List[List[Any]] = Field(..., description="Rows as lists")

class PredictResponse(BaseModel):
    predictions: List

class ReportRequest(BaseModel):
    columns: Optional[List[str]] = Field(None, description="Feature names for current data (optional)")
    data: Optional[List[List[Any]]] = Field(None, description="Current rows (optional). If omitted, demo reference sample will be used.")
    log_to_mlflow: Optional[bool] = Field(True, description="Deprecated, report is always logged to MLflow and run_id returned")
