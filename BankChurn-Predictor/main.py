#!/usr/bin/env python3
"""
BankChurn Predictor - Sistema de predicción de abandono de clientes bancarios

Uso (CLI unificada):
    # Entrenamiento estándar
    python main.py --mode train \
        --config configs/config.yaml \
        --input data/raw/Churn.csv \
        --model models/best_model.pkl \
        --preprocessor models/preprocessor.pkl

    # Evaluación sobre un CSV etiquetado (alias: eval / evaluate)
    python main.py --mode eval \
        --config configs/config.yaml \
        --input data/raw/Churn.csv \
        --model models/best_model.pkl \
        --preprocessor models/preprocessor.pkl

    # Predicción por lotes sobre nuevos clientes
    python main.py --mode predict \
        --config configs/config.yaml \
        --input data/new_customers.csv \
        --output predictions.csv \
        --model models/best_model.pkl \
        --preprocessor models/preprocessor.pkl

    # Búsqueda de hiperparámetros (modo avanzado)
    python main.py --mode hyperopt \
        --config configs/config.yaml \
        --input data/raw/Churn.csv \
        --n_trials 100 --timeout 3600

Autor: Daniel Duque
Versión: 1.0.0
Fecha: 2024-11-16
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common_utils.seed import set_seed

# Configuración de warnings y logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bankchurn.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Configuración de seeds para reproducibilidad
def set_seeds(seed: int | None = None) -> int:
    """Configura seeds para reproducibilidad completa y devuelve la semilla usada."""
    used = set_seed(seed)
    logger.info(f"Seeds configuradas: {used}")
    return used


class ResampleClassifier:
    """
    Clasificador custom con técnicas de resampling para clases desbalanceadas.

    Implementa oversampling, undersampling y class weighting para mejorar
    el rendimiento en datasets con distribución desigual de clases.
    """

    def __init__(self, estimator=None, strategy: str = "none", random_state: int = 42):
        """
        Inicializa el clasificador con resampling.

        Args:
            estimator: Clasificador base (LogisticRegression, RandomForest, etc.)
            strategy: Estrategia de resampling ('oversample', 'undersample', 'none')
            random_state: Semilla para reproducibilidad
        """
        self.estimator = estimator
        self.strategy = strategy
        self.random_state = random_state
        self._estimator_: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ResampleClassifier":
        """
        Entrena el clasificador aplicando la estrategia de resampling.

        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento

        Returns:
            Self para method chaining
        """
        logger.info(f"Aplicando estrategia de resampling: {self.strategy}")

        df = X.copy()
        df["__target__"] = y.values

        if self.strategy == "oversample":
            # Oversampling de la clase minoritaria
            minority_class = df[df["__target__"] == 1]
            majority_class = df[df["__target__"] == 0]

            minority_oversampled = minority_class.sample(
                n=len(majority_class), replace=True, random_state=self.random_state
            )

            df_resampled = pd.concat([majority_class, minority_oversampled], axis=0)
            df_resampled = df_resampled.sample(frac=1, random_state=self.random_state)

            logger.info(
                f"Oversampling aplicado: {len(minority_class)} -> {len(minority_oversampled)}"
            )

        elif self.strategy == "undersample":
            # Undersampling de la clase mayoritaria
            minority_class = df[df["__target__"] == 1]
            majority_class = df[df["__target__"] == 0]

            majority_undersampled = majority_class.sample(
                n=len(minority_class), replace=False, random_state=self.random_state
            )

            df_resampled = pd.concat([majority_undersampled, minority_class], axis=0)
            df_resampled = df_resampled.sample(frac=1, random_state=self.random_state)

            logger.info(
                f"Undersampling aplicado: {len(majority_class)} -> {len(majority_undersampled)}"
            )

        else:
            df_resampled = df.copy()

        # Separar features y target después del resampling
        y_resampled = df_resampled["__target__"].values
        X_resampled = df_resampled.drop(columns="__target__")

        # Entrenar el estimador base
        from sklearn.base import clone

        self._estimator_ = clone(self.estimator)
        self._estimator_.fit(X_resampled, y_resampled)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones con el modelo entrenado."""
        if self._estimator_ is None:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        return self._estimator_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades de predicción."""
        if self._estimator_ is None:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        return self._estimator_.predict_proba(X)


class BankChurnPredictor:
    """
    Sistema completo de predicción de churn bancario.

    Incluye preprocesamiento, entrenamiento, evaluación y predicción
    con manejo avanzado de clases desbalanceadas.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el predictor con configuración.

        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config = self._load_config(config_path)
        self.model: Any = None
        self.preprocessor: Any = None
        self.is_fitted = False

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Carga configuración desde archivo YAML."""
        default_config = {
            "data": {
                "target_column": "Exited",
                "drop_columns": ["RowNumber", "CustomerId", "Surname"],
                "categorical_features": ["Geography", "Gender"],
                "numerical_features": [
                    "CreditScore",
                    "Age",
                    "Tenure",
                    "Balance",
                    "NumOfProducts",
                    "HasCrCard",
                    "IsActiveMember",
                    "EstimatedSalary",
                ],
            },
            "training": {
                "test_size": 0.2,
                "validation_size": 0.2,
                "random_state": 42,
                "stratify": True,
            },
            "model": {"type": "ensemble", "resampling_strategy": "oversample"},
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            # Merge con configuración por defecto
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
        else:
            logger.warning("Usando configuración por defecto")
            return default_config

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Carga y valida el dataset.

        Args:
            data_path: Ruta al archivo CSV

        Returns:
            DataFrame con los datos cargados
        """
        logger.info(f"Cargando datos desde: {data_path}")

        try:
            df = pd.read_csv(data_path)
            logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

            # Validaciones básicas
            if self.config["data"]["target_column"] not in df.columns:
                raise ValueError(
                    f"Columna target '{self.config['data']['target_column']}' no encontrada"
                )

            # Información sobre distribución de clases
            target_dist = df[self.config["data"]["target_column"]].value_counts()
            logger.info(f"Distribución de clases: {target_dist.to_dict()}")

            return df

        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocesa los datos aplicando transformaciones.

        Args:
            df: DataFrame con datos raw

        Returns:
            Tuple con (X_processed, y)
        """
        logger.info("Iniciando preprocesamiento de datos")

        # Eliminar columnas irrelevantes
        df_clean = df.drop(columns=self.config["data"]["drop_columns"], errors="ignore")

        # Feature engineering simple: indicador de clientes seniors
        if "Age" in df_clean.columns and "Age_over_60" not in df_clean.columns:
            df_clean["Age_over_60"] = (df_clean["Age"] > 60).astype(int)

        # Imputación básica de valores faltantes para evitar NaN en el modelo
        numerical_features_cfg = self.config["data"]["numerical_features"]
        num_cols = [c for c in numerical_features_cfg if c in df_clean.columns]
        if num_cols:
            df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())

        categorical_features_cfg = self.config["data"]["categorical_features"]
        cat_cols = [c for c in categorical_features_cfg if c in df_clean.columns]
        if cat_cols:
            df_clean[cat_cols] = df_clean[cat_cols].fillna("missing")

        # Separar features y target
        X = df_clean.drop(columns=[self.config["data"]["target_column"]])
        y = df_clean[self.config["data"]["target_column"]]

        # Crear pipeline de preprocesamiento
        categorical_features = categorical_features_cfg
        numerical_features = list(numerical_features_cfg)
        if (
            "Age_over_60" in df_clean.columns
            and "Age_over_60" not in numerical_features
        ):
            numerical_features.append("Age_over_60")

        # Transformadores
        categorical_transformer = OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        )
        numerical_transformer = StandardScaler()

        # Pipeline completo
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="passthrough",
        )

        # Aplicar transformaciones
        X_processed = self.preprocessor.fit_transform(X)

        # Convertir a DataFrame para mantener compatibilidad
        feature_names = numerical_features + list(
            self.preprocessor.named_transformers_["cat"].get_feature_names_out(
                categorical_features
            )
        )
        X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

        logger.info(f"Preprocesamiento completado: {X_processed.shape[1]} features")

        return X_processed, y

    def create_model(self) -> Pipeline:
        """
        Crea el modelo ensemble con resampling.

        Returns:
            Pipeline con el modelo configurado
        """
        logger.info("Creando modelo ensemble")

        # Modelos base
        lr = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=self.config["training"]["random_state"],
        )

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=self.config["training"]["random_state"],
        )

        # Ensemble con voting
        ensemble = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf)], voting="soft", weights=[0.4, 0.6]
        )

        # Aplicar resampling si está configurado
        if self.config["model"]["resampling_strategy"] != "none":
            model = ResampleClassifier(
                estimator=ensemble,
                strategy=self.config["model"]["resampling_strategy"],
                random_state=self.config["training"]["random_state"],
            )
        else:
            model = ensemble

        return model

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Entrena el modelo con validación cruzada.

        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento

        Returns:
            Diccionario con métricas de validación cruzada
        """
        logger.info("Iniciando entrenamiento del modelo")

        # Crear modelo
        self.model = self.create_model()

        # Validación cruzada estratificada
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.config["training"]["random_state"],
        )

        cv_scores: Dict[str, List[float]] = {
            "f1": [],
            "roc_auc": [],
            "precision": [],
            "recall": [],
        }

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Entrenando fold {fold + 1}/5")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Entrenar modelo en fold
            model_fold = self.create_model()
            model_fold.fit(X_train_fold, y_train_fold)

            # Evaluar en validación
            y_pred = model_fold.predict(X_val_fold)
            y_pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]

            # Calcular métricas
            from sklearn.metrics import precision_score, recall_score

            cv_scores["f1"].append(f1_score(y_val_fold, y_pred))
            cv_scores["roc_auc"].append(roc_auc_score(y_val_fold, y_pred_proba))
            cv_scores["precision"].append(precision_score(y_val_fold, y_pred))
            cv_scores["recall"].append(recall_score(y_val_fold, y_pred))

        # Entrenar modelo final en todos los datos
        self.model.fit(X, y)
        self.is_fitted = True

        # Calcular estadísticas de CV
        cv_stats: Dict[str, float] = {}
        for metric, scores in cv_scores.items():
            cv_stats[f"{metric}_mean"] = float(np.mean(scores))
            cv_stats[f"{metric}_std"] = float(np.std(scores))

        logger.info("Entrenamiento completado")
        for metric, value in cv_stats.items():
            logger.info(f"{metric}: {value:.4f}")

        return cv_stats

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evalúa el modelo en el conjunto de test.

        Args:
            X_test: Features de test
            y_test: Target de test

        Returns:
            Diccionario con métricas y resultados de evaluación
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")

        logger.info("Evaluando modelo en conjunto de test")

        # Predicciones
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        metrics = {
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Reporte de clasificación
        class_report = classification_report(y_test, y_pred, output_dict=True)

        results = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": class_report,
            "predictions": y_pred,
            "probabilities": y_pred_proba,
        }

        # Log de resultados
        logger.info("Resultados de evaluación:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        return results

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza predicciones en nuevos datos.

        Args:
            X: Features para predicción

        Returns:
            Tuple con (predicciones, probabilidades)
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return predictions, probabilities

    def save_model(self, model_path: str, preprocessor_path: str) -> None:
        """Guarda el modelo y preprocessor entrenados."""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")

        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)

        # Guardar metadatos
        metadata = {
            "model_type": "BankChurnPredictor",
            "version": "1.0.0",
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
        }

        metadata_path = model_path.replace(".pkl", "_metadata.json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Modelo guardado en: {model_path}")
        logger.info(f"Preprocessor guardado en: {preprocessor_path}")

    def load_model(self, model_path: str, preprocessor_path: str) -> None:
        """Carga un modelo y preprocessor previamente entrenados."""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.is_fitted = True

        logger.info(f"Modelo cargado desde: {model_path}")


def hyperparameter_optimization(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 100
) -> Dict[str, Any]:
    """
    Optimización de hiperparámetros con Optuna.

    Args:
        X: Features de entrenamiento
        y: Target de entrenamiento
        n_trials: Número de trials para optimización

    Returns:
        Mejores hiperparámetros encontrados
    """
    logger.info(f"Iniciando optimización de hiperparámetros ({n_trials} trials)")

    def objective(trial):
        # Hiperparámetros a optimizar
        params = {
            "lr_C": trial.suggest_float("lr_C", 0.01, 10.0, log=True),
            "rf_n_estimators": trial.suggest_int("rf_n_estimators", 50, 200),
            "rf_max_depth": trial.suggest_int("rf_max_depth", 5, 20),
            "rf_min_samples_split": trial.suggest_int("rf_min_samples_split", 5, 20),
            "ensemble_lr_weight": trial.suggest_float("ensemble_lr_weight", 0.1, 0.9),
        }

        # Crear modelo con hiperparámetros
        lr = LogisticRegression(
            C=params["lr_C"],
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )

        rf = RandomForestClassifier(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_split=params["rf_min_samples_split"],
            class_weight="balanced_subsample",
            random_state=42,
        )

        ensemble = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf)],
            voting="soft",
            weights=[params["ensemble_lr_weight"], 1 - params["ensemble_lr_weight"]],
        )

        # Validación cruzada
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            ensemble.fit(X_train_fold, y_train_fold)
            y_pred = ensemble.predict(X_val_fold)
            scores.append(f1_score(y_val_fold, y_pred))

        return np.mean(scores)

    # Ejecutar optimización
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Mejor F1-Score: {study.best_value:.4f}")
    logger.info(f"Mejores hiperparámetros: {study.best_params}")

    return study.best_params


def main():
    """Función principal con CLI."""
    parser = argparse.ArgumentParser(
        description="BankChurn Predictor - Sistema de predicción de churn bancario"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "evaluate", "predict", "hyperopt"],
        help="Modo de ejecución (train | eval | predict | hyperopt)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Ruta al archivo de configuración",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/Churn.csv",
        help="Ruta al archivo de datos de entrada",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Ruta al archivo de salida para predicciones",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pkl",
        help="Ruta al modelo guardado",
    )

    parser.add_argument(
        "--preprocessor",
        type=str,
        default="models/preprocessor.pkl",
        help="Ruta al preprocessor guardado",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla opcional para reproducibilidad (CLI > SEED env > 42)",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Número de trials para optimización de hiperparámetros",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout en segundos para optimización",
    )

    args = parser.parse_args()

    # Configurar seeds
    seed_used = set_seeds(args.seed)

    # Crear directorios necesarios
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    try:
        if args.mode == "train":
            logger.info("=== MODO ENTRENAMIENTO ===")

            # Inicializar predictor
            predictor = BankChurnPredictor(args.config)

            # Cargar y preprocesar datos
            df = predictor.load_data(args.input)
            X, y = predictor.preprocess_data(df)

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=predictor.config["training"]["test_size"],
                random_state=predictor.config["training"]["random_state"],
                stratify=y if predictor.config["training"]["stratify"] else None,
            )

            # Entrenar modelo
            cv_results = predictor.train(X_train, y_train)

            # Evaluar en test
            test_results = predictor.evaluate(X_test, y_test)

            # Guardar modelo
            predictor.save_model(args.model, args.preprocessor)

            # Guardar resultados
            results = {
                "cv_results": cv_results,
                "test_results": {
                    k: v
                    for k, v in test_results.items()
                    if k not in ["predictions", "probabilities"]
                },
            }

            import json

            with open("results/training_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Exportar paquete combinado para demo (preprocessor + model)
            try:
                combined_path = Path("models") / "model_v1.0.0.pkl"
                joblib.dump(
                    {
                        "preprocessor": predictor.preprocessor,
                        "model": predictor.model,
                        "version": "1.0.0",
                    },
                    combined_path,
                )
                logger.info(f"Paquete combinado exportado en: {combined_path}")
            except Exception as e:
                logger.warning(f"No se pudo exportar paquete combinado: {e}")

            logger.info("Entrenamiento completado exitosamente")

        elif args.mode in ("evaluate", "eval"):
            logger.info("=== MODO EVALUACIÓN ===")

            # Cargar modelo
            predictor = BankChurnPredictor(args.config)
            predictor.load_model(args.model, args.preprocessor)

            # Cargar datos de test
            df = predictor.load_data(args.input)
            X, y = predictor.preprocess_data(df)

            # Evaluar
            results = predictor.evaluate(X, y)

            # Mostrar resultados
            print("\n=== RESULTADOS DE EVALUACIÓN ===")
            for metric, value in results["metrics"].items():
                print(f"{metric}: {value:.4f}")

            print("\nMatriz de Confusión:")
            print(results["confusion_matrix"])

        elif args.mode == "predict":
            logger.info("=== MODO PREDICCIÓN ===")

            # Cargar modelo
            predictor = BankChurnPredictor(args.config)
            predictor.load_model(args.model, args.preprocessor)

            # Cargar datos para predicción
            df = pd.read_csv(args.input)
            X, _ = predictor.preprocess_data(df)

            # Realizar predicciones
            predictions, probabilities = predictor.predict(X)

            # Guardar resultados
            results_df = df.copy()
            results_df["churn_prediction"] = predictions
            results_df["churn_probability"] = probabilities
            results_df["risk_level"] = pd.cut(
                probabilities, bins=[0, 0.3, 0.7, 1.0], labels=["LOW", "MEDIUM", "HIGH"]
            )

            results_df.to_csv(args.output, index=False)
            logger.info(f"Predicciones guardadas en: {args.output}")

        elif args.mode == "hyperopt":
            logger.info("=== MODO OPTIMIZACIÓN DE HIPERPARÁMETROS ===")

            # Cargar y preprocesar datos
            predictor = BankChurnPredictor(args.config)
            df = predictor.load_data(args.input)
            X, y = predictor.preprocess_data(df)

            # Optimizar hiperparámetros
            best_params = hyperparameter_optimization(X, y, args.n_trials)

            # Guardar mejores hiperparámetros
            with open("results/best_hyperparameters.json", "w") as f:
                json.dump(best_params, f, indent=2)

            logger.info("Optimización completada")

    except Exception as e:
        logger.error(f"Error en ejecución: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
