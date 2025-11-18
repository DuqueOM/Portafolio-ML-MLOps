"""
FastAPI application para BankChurn Predictor

Ejecutar con: uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000 --reload

Endpoints disponibles:
- POST /predict: Predicción individual
- POST /predict_batch: Predicción en lote
- GET /health: Health check
- GET /metrics: Métricas del modelo
- GET /model_info: Información del modelo
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Agregar path del proyecto
sys.path.append("..")
from main import BankChurnPredictor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="BankChurn Predictor API",
    description="API para predicción de abandono de clientes bancarios",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para el modelo
predictor = None
model_metadata = {}
prediction_cache = {}
request_count = 0
total_prediction_time = 0.0


class CustomerData(BaseModel):
    """Schema para datos de un cliente individual."""

    CreditScore: int = Field(
        ..., ge=300, le=850, description="Puntaje crediticio (300-850)"
    )
    Geography: str = Field(
        ..., description="País de residencia", regex="^(France|Spain|Germany)$"
    )
    Gender: str = Field(..., description="Género", regex="^(Male|Female)$")
    Age: int = Field(..., ge=18, le=100, description="Edad del cliente (18-100)")
    Tenure: int = Field(..., ge=0, le=10, description="Años como cliente (0-10)")
    Balance: float = Field(..., ge=0, description="Saldo de cuenta")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Número de productos (1-4)")
    HasCrCard: int = Field(
        ..., ge=0, le=1, description="Posee tarjeta de crédito (0/1)"
    )
    IsActiveMember: int = Field(..., ge=0, le=1, description="Cliente activo (0/1)")
    EstimatedSalary: float = Field(..., ge=0, description="Salario estimado")

    @validator("Geography")
    def validate_geography(cls, v):
        valid_countries = ["France", "Spain", "Germany"]
        if v not in valid_countries:
            raise ValueError(f"Geography debe ser uno de: {valid_countries}")
        return v

    @validator("Gender")
    def validate_gender(cls, v):
        valid_genders = ["Male", "Female"]
        if v not in valid_genders:
            raise ValueError(f"Gender debe ser uno de: {valid_genders}")
        return v


class BatchCustomerData(BaseModel):
    """Schema para predicción en lote."""

    customers: List[CustomerData] = Field(
        ..., description="Lista de clientes para predicción"
    )

    @validator("customers")
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Máximo 1000 clientes por batch")
        if len(v) == 0:
            raise ValueError("Debe incluir al menos un cliente")
        return v


class PredictionResponse(BaseModel):
    """Schema para respuesta de predicción individual."""

    churn_probability: float = Field(..., description="Probabilidad de churn (0-1)")
    churn_prediction: int = Field(..., description="Predicción binaria (0/1)")
    risk_level: str = Field(..., description="Nivel de riesgo (LOW/MEDIUM/HIGH)")
    confidence: float = Field(..., description="Confianza de la predicción")
    feature_contributions: Dict[str, float] = Field(
        ..., description="Contribución de cada feature"
    )
    model_version: str = Field(..., description="Versión del modelo")
    prediction_timestamp: str = Field(..., description="Timestamp de la predicción")


class BatchPredictionResponse(BaseModel):
    """Schema para respuesta de predicción en lote."""

    predictions: List[PredictionResponse] = Field(
        ..., description="Lista de predicciones"
    )
    batch_id: str = Field(..., description="ID único del batch")
    total_customers: int = Field(..., description="Total de clientes procesados")
    processing_time_seconds: float = Field(..., description="Tiempo de procesamiento")


class HealthResponse(BaseModel):
    """Schema para health check."""

    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    uptime_seconds: float = Field(..., description="Tiempo de actividad en segundos")
    version: str = Field(..., description="Versión de la API")


class ModelMetrics(BaseModel):
    """Schema para métricas del modelo."""

    total_predictions: int = Field(..., description="Total de predicciones realizadas")
    average_prediction_time_ms: float = Field(
        ..., description="Tiempo promedio de predicción (ms)"
    )
    model_accuracy: Optional[float] = Field(None, description="Accuracy del modelo")
    model_f1_score: Optional[float] = Field(None, description="F1-Score del modelo")
    model_auc_roc: Optional[float] = Field(None, description="AUC-ROC del modelo")


# Tiempo de inicio del servidor
start_time = time.time()


def load_model():
    """Carga el modelo y preprocessor al iniciar la aplicación."""
    global predictor, model_metadata

    try:
        logger.info("Cargando modelo...")

        model_path = Path("../models/best_model.pkl")
        preprocessor_path = Path("../models/preprocessor.pkl")
        metadata_path = Path("../models/best_model_metadata.json")

        if not model_path.exists():
            logger.error(f"Modelo no encontrado en: {model_path}")
            return False

        if not preprocessor_path.exists():
            logger.error(f"Preprocessor no encontrado en: {preprocessor_path}")
            return False

        # Cargar modelo
        predictor = BankChurnPredictor()
        predictor.load_model(str(model_path), str(preprocessor_path))

        # Cargar metadatos si existen
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                model_metadata = json.load(f)
        else:
            model_metadata = {
                "model_type": "BankChurnPredictor",
                "version": "1.0.0",
                "training_date": "Unknown",
            }

        logger.info("Modelo cargado exitosamente")
        return True

    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        return False


def calculate_feature_contributions(customer_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calcula contribuciones aproximadas de features usando valores promedio.
    En producción, se usaría SHAP o LIME para explicaciones precisas.
    """
    # Contribuciones simuladas basadas en importancia conocida
    base_contributions = {
        "Age": 0.0,
        "NumOfProducts": 0.0,
        "IsActiveMember": 0.0,
        "Geography": 0.0,
        "Balance": 0.0,
        "CreditScore": 0.0,
        "EstimatedSalary": 0.0,
    }

    # Lógica simplificada para contribuciones
    if customer_data["Age"] > 50:
        base_contributions["Age"] = 0.15
    elif customer_data["Age"] < 30:
        base_contributions["Age"] = -0.05

    if customer_data["NumOfProducts"] == 1:
        base_contributions["NumOfProducts"] = 0.12
    elif customer_data["NumOfProducts"] > 2:
        base_contributions["NumOfProducts"] = -0.08

    if customer_data["IsActiveMember"] == 0:
        base_contributions["IsActiveMember"] = 0.18
    else:
        base_contributions["IsActiveMember"] = -0.10

    if customer_data["Geography"] == "Germany":
        base_contributions["Geography"] = 0.14
    elif customer_data["Geography"] == "France":
        base_contributions["Geography"] = -0.05

    if customer_data["Balance"] == 0:
        base_contributions["Balance"] = 0.08
    elif customer_data["Balance"] > 100000:
        base_contributions["Balance"] = 0.05

    if customer_data["CreditScore"] < 600:
        base_contributions["CreditScore"] = 0.06
    elif customer_data["CreditScore"] > 750:
        base_contributions["CreditScore"] = -0.04

    return base_contributions


def determine_risk_level(probability: float) -> str:
    """Determina el nivel de riesgo basado en la probabilidad."""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


def calculate_confidence(probability: float) -> float:
    """Calcula la confianza de la predicción."""
    # Confianza basada en qué tan lejos está de 0.5 (incertidumbre máxima)
    return abs(probability - 0.5) * 2


@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación."""
    logger.info("Iniciando BankChurn Predictor API...")

    success = load_model()
    if not success:
        logger.error("No se pudo cargar el modelo. La API funcionará en modo limitado.")
    else:
        logger.info("API iniciada exitosamente")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información básica."""
    return {
        "message": "BankChurn Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check del servicio."""
    uptime = time.time() - start_time

    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        uptime_seconds=uptime,
        version="1.0.0",
    )


@app.get("/model_info", response_model=Dict[str, Any])
async def get_model_info():
    """Información del modelo cargado."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    return {
        "model_metadata": model_metadata,
        "model_loaded": True,
        "features_expected": [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
        ],
    }


@app.get("/metrics", response_model=ModelMetrics)
async def get_metrics():
    """Métricas del modelo y API."""
    global request_count, total_prediction_time

    avg_time_ms = (
        (total_prediction_time / request_count * 1000) if request_count > 0 else 0
    )

    # Métricas del modelo desde metadatos (si están disponibles)
    model_accuracy = model_metadata.get("test_accuracy")
    model_f1 = model_metadata.get("test_f1_score")
    model_auc = model_metadata.get("test_auc_roc")

    return ModelMetrics(
        total_predictions=request_count,
        average_prediction_time_ms=avg_time_ms,
        model_accuracy=model_accuracy,
        model_f1_score=model_f1,
        model_auc_roc=model_auc,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predicción de churn para un cliente individual."""
    global request_count, total_prediction_time

    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    start_pred_time = time.time()

    try:
        # Convertir a DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])

        # Preprocesar (usando el preprocessor cargado)
        X_processed = predictor.preprocessor.transform(df)

        # Convertir a DataFrame para compatibilidad
        feature_names = predictor.preprocessor.get_feature_names_out()
        X_processed = pd.DataFrame(X_processed, columns=feature_names)

        # Realizar predicción
        predictions, probabilities = predictor.predict(X_processed)

        probability = float(probabilities[0])
        prediction = int(predictions[0])

        # Calcular métricas adicionales
        risk_level = determine_risk_level(probability)
        confidence = calculate_confidence(probability)
        feature_contributions = calculate_feature_contributions(customer_dict)

        # Actualizar métricas
        pred_time = time.time() - start_pred_time
        request_count += 1
        total_prediction_time += pred_time

        return PredictionResponse(
            churn_probability=probability,
            churn_prediction=prediction,
            risk_level=risk_level,
            confidence=confidence,
            feature_contributions=feature_contributions,
            model_version=model_metadata.get("version", "1.0.0"),
            prediction_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch_data: BatchCustomerData, background_tasks: BackgroundTasks
):
    """Predicción de churn para múltiples clientes."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    start_batch_time = time.time()
    batch_id = f"batch_{int(start_batch_time)}"

    try:
        # Convertir batch a DataFrame
        customers_list = [customer.dict() for customer in batch_data.customers]
        df = pd.DataFrame(customers_list)

        # Preprocesar
        X_processed = predictor.preprocessor.transform(df)
        feature_names = predictor.preprocessor.get_feature_names_out()
        X_processed = pd.DataFrame(X_processed, columns=feature_names)

        # Realizar predicciones
        predictions, probabilities = predictor.predict(X_processed)

        # Crear respuestas individuales
        individual_predictions = []
        for i, (customer_dict, prob, pred) in enumerate(
            zip(customers_list, probabilities, predictions)
        ):
            individual_predictions.append(
                PredictionResponse(
                    churn_probability=float(prob),
                    churn_prediction=int(pred),
                    risk_level=determine_risk_level(float(prob)),
                    confidence=calculate_confidence(float(prob)),
                    feature_contributions=calculate_feature_contributions(
                        customer_dict
                    ),
                    model_version=model_metadata.get("version", "1.0.0"),
                    prediction_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            )

        processing_time = time.time() - start_batch_time

        # Actualizar métricas globales
        global request_count, total_prediction_time
        request_count += len(batch_data.customers)
        total_prediction_time += processing_time

        return BatchPredictionResponse(
            predictions=individual_predictions,
            batch_id=batch_id,
            total_customers=len(batch_data.customers),
            processing_time_seconds=processing_time,
        )

    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error en predicción batch: {str(e)}"
        )


@app.post("/reload_model")
async def reload_model():
    """Recarga el modelo (útil para actualizaciones)."""
    success = load_model()

    if success:
        return {"message": "Modelo recargado exitosamente"}
    else:
        raise HTTPException(status_code=500, detail="Error recargando modelo")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
