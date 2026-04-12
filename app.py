import os
import sys
import threading
import mlflow
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, Depends, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from src.vehicle_insurance.constants import (
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_REGISTER_MODEL_NAME,
)
from src.vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from src.vehicle_insurance.logger.logger_config import setup_logger
from src.vehicle_insurance.pipeline.training_pipeline import TrainingPipeline
from src.vehicle_insurance.utils import get_mlflow_client, load_object
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = setup_logger()


class PredictionEngine:
    def __init__(self, model, preprocessor, model_version, recall):
        self.model = model
        self.preprocessor = preprocessor
        self.model_version = model_version
        self.recall = recall

        self.loaded_at = datetime.now()


current_engine = None
is_training = False
_training_lock = threading.Lock()


def _load_engine_from_mlflow():
    try:
        client = get_mlflow_client()
        champion_version_obj = client.get_model_version_by_alias(
            name=MLFLOW_REGISTER_MODEL_NAME, alias=MLFLOW_CHAMPION_ALIAS
        )
        champion_version = champion_version_obj.version
        run_id = champion_version_obj.run_id
        run = client.get_run(run_id=run_id)
        recall = run.data.metrics["test_recall"]
        model_uri = f"models:/{MLFLOW_REGISTER_MODEL_NAME}@{MLFLOW_CHAMPION_ALIAS}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        logger.info(f"Model loaded from URI: {model_uri}")
        local_artifact_dir = client.download_artifacts(
            run_id=run_id, path="data_transformation/preprocessor.pkl"
        )
        preprocessor_path = (
            os.path.join(local_artifact_dir, "preprocessor.pkl")
            if os.path.isdir(local_artifact_dir)
            else local_artifact_dir
        )
        preprocessor = load_object(preprocessor_path)
        logger.info(f"Preprocessor loaded from : {preprocessor_path}")
        return PredictionEngine(
            model=model,
            preprocessor=preprocessor,
            model_version=champion_version,
            recall=recall,
        )
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_engine
    logger.info("App is starting up. Loading champion model from MLflow.")
    try:
        current_engine = _load_engine_from_mlflow()
        logger.info(
            f"Champion model loaded successfully with version: {current_engine.model_version} and recall: {current_engine.recall}"
        )
    except Exception as e:
        logger.error(f"Error occurred while loading champion model: {e}")
        raise VehicleInsuranceException(e, sys) from e

    yield
    logger.info("App is shutting down.")


app = FastAPI(
    title="Vehicle Insurance Prediction API",
    description="API for predicting vehicle insurance claims and managing model training.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def verify_reload_token(x_reload_token: str = Header(...)):
    expected = os.environ.get("RELOAD_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=500, detail="Reload token not configured on server."
        )
    if x_reload_token != expected:
        raise HTTPException(status_code=401, detail="Invalid reload token.")


class PredictRequest(BaseModel):
    Age: int
    Gender: str
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Vehicle_Age: str
    Vehicle_Damage: str
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 35,
                "Gender": "Male",
                "Driving_License": 1,
                "Region_Code": 28.0,
                "Previously_Insured": 0,
                "Vehicle_Age": "1-2 Year",
                "Vehicle_Damage": "Yes",
                "Annual_Premium": 25000.0,
                "Policy_Sales_Channel": 1.0,
                "Vintage": 5,
            }
        }
    }


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    api_timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    recall: float
    loaded_at: str


class TrainResponse(BaseModel):
    status: str
    message: str


class ReloadResponse(BaseModel):
    status: str
    new_model_version: str
    new_recall: float
    reloaded_at: str


@app.get("/", tags=["UI"])
async def root(request: Request):
    """Serves the frontend UI."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
async def health_check():
    engine = current_engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return HealthResponse(
        status="healthy",
        model_version=engine.model_version,
        recall=engine.recall,
        loaded_at=engine.loaded_at.isoformat(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    engine = current_engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        input_df = pd.DataFrame([request.model_dump()])
        transformed_input = engine.preprocessor.transform(input_df)

        prediction_raw = engine.model.predict(transformed_input)
        prediction = int(prediction_raw[0])

        underlying = engine.model._model_impl.lgb_model
        proba = float(underlying.predict_proba(transformed_input)[0][1])

        return PredictResponse(
            prediction=prediction,
            probability=round(proba, 4),
            model_version=engine.model_version,
            api_timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")


def _run_training():
    global is_training
    try:
        logger.info("Background training pipeline started.")
        TrainingPipeline().run_pipeline()
        logger.info("Background training pipeline finished.")
    except Exception as e:
        logger.info(f"Error during training pipeline: {e}")
    finally:
        with _training_lock:
            is_training = False
        logger.info("Training lock released.")


@app.post("/train", response_model=TrainResponse, tags=["Model Training"])
async def train(background_tasks: BackgroundTasks):
    global is_training
    with _training_lock:
        if is_training:
            raise HTTPException(
                status_code=409, detail="Training is already in progress."
            )
        is_training = True
    background_tasks.add_task(_run_training)
    return TrainResponse(
        status="started", message="Training started in the background."
    )


@app.post("/reload", response_model=ReloadResponse, tags=["Model Management"])
async def reload_model(_: None = Depends(verify_reload_token)):
    global current_engine
    try:
        new_engine = _load_engine_from_mlflow()

    except Exception as e:
        logger.info(f"/reload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload model.")

    current_engine = new_engine
    logger.info(
        f"Engine swapped. New version: {current_engine.model_version}, recall: {current_engine.recall}"
    )
    return ReloadResponse(
        status="success",
        new_model_version=current_engine.model_version,
        new_recall=current_engine.recall,
        reloaded_at=datetime.now().isoformat(),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
