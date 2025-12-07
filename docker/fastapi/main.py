# app/main.py
import logging
from fastapi import FastAPI
from models import load_models, MODELS
from predict import router as predict_router
from reports import router as reports_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="MLflow Registry Predictor (modular)")

@app.on_event("startup")
def startup():
    load_models()
    logger.info("Models loaded at startup")

# include routers
app.include_router(predict_router)
app.include_router(reports_router)

@app.get("/health")
def health():
    loaded = {k: (v is not None) for k, v in MODELS.items()}
    return {
        "status": "ok",
        "models_loaded": [k for k, v in loaded.items() if v],
        "models_status": loaded
    }
