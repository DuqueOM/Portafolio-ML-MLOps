#!/usr/bin/env python3
"""
GoldRecovery Process Optimizer - Sistema de optimización de procesos metalúrgicos

Uso (CLI unificada v1):
    # Entrenamiento estándar (usa configs/config.yaml si está disponible)
    python main.py --mode train \
        --config configs/config.yaml \
        --input gold_recovery_train.csv \
        --model models/metallurgical_model.pkl

    # Evaluación sobre CSV etiquetado (alias: eval / evaluate)
    python main.py --mode eval \
        --config configs/config.yaml \
        --input gold_recovery_test.csv \
        --model models/metallurgical_model.pkl

    # Predicción por lotes
    python main.py --mode predict \
        --config configs/config.yaml \
        --input gold_recovery_test.csv \
        --output results/predictions.csv \
        --model models/metallurgical_model.pkl

    # Modos avanzados (demo)
    python main.py --mode optimize --config configs/config.yaml --model models/metallurgical_model.pkl
    python main.py --mode monitor --dashboard --port 8501

Autor: Daniel Duque
Versión: 1.0.0
Fecha: 2024-11-16
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common_utils.seed import set_seed

# Configuración de warnings y logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("goldrecovery.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Calcula sMAPE (Symmetric Mean Absolute Percentage Error).

    Métrica especializada para procesos industriales que maneja
    valores cercanos a cero mejor que MAPE tradicional.
    """
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )


class ProcessDataLoader:
    """Cargador y validador de datos de procesos metalúrgicos."""

    def __init__(self):
        self.required_columns = [
            "date",
            "final.output.recovery",
            # columnas típicas del dataset de oro/plata/plomo
            "rougher.output.concentrate_au",
            "rougher.output.concentrate_ag",
            "primary_cleaner.output.concentrate_au",
            "primary_cleaner.output.concentrate_ag",
            "secondary_cleaner.output.concentrate_au",
            "secondary_cleaner.output.concentrate_ag",
        ]

    def load_process_data(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Carga datos de múltiples archivos de proceso.

        Args:
            file_paths: Lista de rutas a archivos CSV

        Returns:
            DataFrame consolidado con datos de proceso
        """
        logger.info(f"Cargando datos de {len(file_paths)} archivos")

        dataframes = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                df["source_file"] = Path(file_path).stem
                dataframes.append(df)
                logger.info(
                    f"Cargado {file_path}: {df.shape[0]} filas, {df.shape[1]} columnas"
                )
            except Exception as e:
                logger.error(f"Error cargando {file_path}: {e}")
                continue

        if not dataframes:
            raise ValueError("No se pudieron cargar datos de ningún archivo")

        # Consolidar datos
        consolidated_df = pd.concat(dataframes, ignore_index=True)
        logger.info(
            f"Datos consolidados: {consolidated_df.shape[0]} filas, {consolidated_df.shape[1]} columnas"
        )

        return consolidated_df

    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida y limpia datos de proceso metalúrgico.

        Args:
            df: DataFrame con datos raw

        Returns:
            DataFrame limpio y validado
        """
        logger.info("Iniciando validación y limpieza de datos")

        df_clean = df.copy()

        # Convertir fecha si existe
        if "date" in df_clean.columns:
            df_clean["date"] = pd.to_datetime(df_clean["date"])
            df_clean = df_clean.sort_values("date").reset_index(drop=True)

        # Filtrar valores válidos de recovery
        if "final.output.recovery" in df_clean.columns:
            initial_count = len(df_clean)
            df_clean = df_clean[
                (df_clean["final.output.recovery"] >= 0)
                & (df_clean["final.output.recovery"] <= 100)
            ]
            logger.info(f"Filtrado recovery: {initial_count} -> {len(df_clean)} filas")

        # Eliminar filas con demasiados valores faltantes
        threshold = 0.7  # Al menos 70% de datos válidos
        df_clean = df_clean.dropna(thresh=int(threshold * len(df_clean.columns)))

        # Crear features derivadas
        df_clean = self._create_derived_features(df_clean)

        logger.info(
            f"Datos después de limpieza: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas"
        )

        return df_clean

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features derivadas específicas para procesos metalúrgicos."""

        # Recovery ratios entre etapas
        if all(
            col in df.columns
            for col in [
                "rougher.output.concentrate_au",
                "primary_cleaner.output.concentrate_au",
            ]
        ):
            df["au_recovery_ratio"] = df["primary_cleaner.output.concentrate_au"] / (
                df["rougher.output.concentrate_au"] + 1e-6
            )

        if all(
            col in df.columns
            for col in [
                "rougher.output.concentrate_ag",
                "primary_cleaner.output.concentrate_ag",
            ]
        ):
            df["ag_recovery_ratio"] = df["primary_cleaner.output.concentrate_ag"] / (
                df["rougher.output.concentrate_ag"] + 1e-6
            )

        # Eficiencia de concentración
        concentrate_cols = [
            col
            for col in df.columns
            if "concentrate" in col and ("au" in col or "ag" in col)
        ]
        if concentrate_cols:
            df["total_concentrate_efficiency"] = df[concentrate_cols].sum(axis=1)

        # Features temporales si hay fecha
        if "date" in df.columns:
            df["hour"] = df["date"].dt.hour
            df["day_of_week"] = df["date"].dt.dayofweek
            df["month"] = df["date"].dt.month

            # Rolling features para capturar tendencias
            if "final.output.recovery" in df.columns:
                df["recovery_rolling_mean_24h"] = (
                    df["final.output.recovery"].rolling(24, min_periods=1).mean()
                )
                df["recovery_rolling_std_24h"] = (
                    df["final.output.recovery"].rolling(24, min_periods=1).std()
                )

        return df


class MetallurgicalPredictor:
    """Predictor especializado para procesos metalúrgicos."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.models: Dict[str, Any] = {}
        self.feature_columns = None
        self.is_fitted = False

    def _get_default_config(self) -> Dict[str, Any]:
        """Configuración por defecto para modelos metalúrgicos."""
        return {
            "models": {
                "xgboost": {
                    "n_estimators": 500,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                },
                "lightgbm": {
                    "n_estimators": 500,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                },
                "random_forest": {
                    "n_estimators": 300,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42,
                },
            },
            "ensemble_weights": {
                "xgboost": 0.4,
                "lightgbm": 0.35,
                "random_forest": 0.25,
            },
        }

    def prepare_features(
        self, df: pd.DataFrame, target_column: str = "final.output.recovery"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara features para entrenamiento/predicción.

        Args:
            df: DataFrame con datos de proceso
            target_column: Nombre de la columna objetivo

        Returns:
            Tuple con (X, y)
        """
        # Separar features y target
        if target_column in df.columns:
            y = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            y = None
            X = df.copy()

        # Eliminar columnas no numéricas o de identificación
        exclude_columns = ["date", "source_file"]
        X = X.select_dtypes(include=[np.number])
        X = X.drop(
            columns=[col for col in exclude_columns if col in X.columns],
            errors="ignore",
        )

        # Manejar valores faltantes
        X = X.fillna(X.median())

        # Guardar columnas de features para consistencia
        if self.feature_columns is None:
            self.feature_columns = X.columns.tolist()
        else:
            # Asegurar consistencia de features
            missing_cols = set(self.feature_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(self.feature_columns)

            if missing_cols:
                logger.warning(f"Columnas faltantes: {missing_cols}")
                for col in missing_cols:
                    X[col] = 0

            if extra_cols:
                logger.warning(f"Columnas extra ignoradas: {extra_cols}")

            X = X[self.feature_columns]

        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Entrena ensemble de modelos metalúrgicos.

        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento

        Returns:
            Métricas de validación cruzada
        """
        logger.info("Iniciando entrenamiento de modelos metalúrgicos")

        # Inicializar modelos
        self.models = {
            "xgboost": xgb.XGBRegressor(**self.config["models"]["xgboost"]),
            "lightgbm": lgb.LGBMRegressor(**self.config["models"]["lightgbm"]),
            "random_forest": RandomForestRegressor(
                **self.config["models"]["random_forest"]
            ),
        }

        # Entrenar cada modelo y evaluar con CV
        cv_results = {}

        for model_name, model in self.models.items():
            logger.info(f"Entrenando {model_name}...")

            # Entrenar modelo
            model.fit(X, y)

            # Validación cruzada
            cv_scores = cross_val_score(
                model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
            )

            cv_results[f"{model_name}_mae"] = -cv_scores.mean()
            cv_results[f"{model_name}_mae_std"] = cv_scores.std()

            logger.info(
                f"{model_name} - MAE CV: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
            )

        self.is_fitted = True
        logger.info("Entrenamiento completado")

        return cv_results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones usando ensemble de modelos.

        Args:
            X: Features para predicción

        Returns:
            Array con predicciones ensemble
        """
        if not self.is_fitted:
            raise ValueError("Modelos no entrenados. Ejecutar train() primero.")

        # Preparar features
        X_prepared, _ = self.prepare_features(X)

        # Predicciones individuales
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X_prepared)

        # Ensemble ponderado
        weights = self.config["ensemble_weights"]
        ensemble_pred = sum(
            [weights[name] * pred for name, pred in predictions.items()]
        )

        return ensemble_pred

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evalúa el modelo en conjunto de test.

        Args:
            X_test: Features de test
            y_test: Target de test

        Returns:
            Diccionario con métricas de evaluación
        """
        if not self.is_fitted:
            raise ValueError("Modelos no entrenados. Ejecutar train() primero.")

        # Predicciones
        y_pred = self.predict(X_test)

        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        smape = symmetric_mean_absolute_percentage_error(y_test, y_pred)

        # R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "smape": smape, "r2_score": r2}

        logger.info("Métricas de evaluación:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        return metrics

    def save_models(self, model_path: str) -> None:
        """Guarda los modelos entrenados."""
        if not self.is_fitted:
            raise ValueError("Modelos no entrenados. Ejecutar train() primero.")

        model_data = {
            "models": self.models,
            "config": self.config,
            "feature_columns": self.feature_columns,
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Modelos guardados en: {model_path}")

    def load_models(self, model_path: str) -> None:
        """Carga modelos previamente entrenados."""
        model_data = joblib.load(model_path)

        self.models = model_data["models"]
        self.config = model_data["config"]
        self.feature_columns = model_data["feature_columns"]
        self.is_fitted = True

        logger.info(f"Modelos cargados desde: {model_path}")


class ProcessOptimizer:
    """Optimizador de procesos metalúrgicos."""

    def __init__(self, predictor: MetallurgicalPredictor):
        self.predictor = predictor

    def optimize_process_parameters(
        self,
        current_conditions: Dict[str, float],
        target_recovery: float = 90.0,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Optimiza parámetros de proceso para alcanzar recovery objetivo.

        Args:
            current_conditions: Condiciones actuales del proceso
            target_recovery: Recovery objetivo (%)
            constraints: Restricciones en parámetros (min, max)

        Returns:
            Parámetros optimizados y predicciones
        """
        logger.info(f"Optimizando proceso para recovery objetivo: {target_recovery}%")

        # Configurar restricciones por defecto
        if constraints is None:
            constraints = {
                "rougher.feed.rate": (100, 1000),
                "primary_cleaner.input": (10, 100),
                "secondary_cleaner.input": (5, 50),
            }

        # Implementación simplificada de optimización
        # En producción se usaría scipy.optimize o algoritmos genéticos

        best_params = current_conditions.copy()
        best_recovery = self._predict_recovery_from_conditions(current_conditions)

        # Grid search simplificado
        for param, (min_val, max_val) in constraints.items():
            if param in current_conditions:
                test_values = np.linspace(min_val, max_val, 20)

                for test_val in test_values:
                    test_conditions = current_conditions.copy()
                    test_conditions[param] = test_val

                    predicted_recovery = self._predict_recovery_from_conditions(
                        test_conditions
                    )

                    if abs(predicted_recovery - target_recovery) < abs(
                        best_recovery - target_recovery
                    ):
                        best_recovery = predicted_recovery
                        best_params[param] = test_val

        optimization_result = {
            "optimized_parameters": best_params,
            "predicted_recovery": best_recovery,
            "improvement": best_recovery
            - self._predict_recovery_from_conditions(current_conditions),
            "target_achieved": abs(best_recovery - target_recovery) < 1.0,
        }

        logger.info(f"Optimización completada. Recovery predicho: {best_recovery:.2f}%")

        return optimization_result

    def _predict_recovery_from_conditions(self, conditions: Dict[str, float]) -> float:
        """Predice recovery basado en condiciones de proceso."""
        # Convertir condiciones a DataFrame
        df_conditions = pd.DataFrame([conditions])

        # Realizar predicción
        try:
            prediction = self.predictor.predict(df_conditions)[0]
            return max(0, min(100, prediction))  # Clamp entre 0-100%
        except Exception as e:
            logger.warning(f"Error en predicción: {e}")
            return 85.0  # Valor por defecto


def main():
    """Función principal con CLI."""
    parser = argparse.ArgumentParser(description="GoldRecovery Process Optimizer")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "evaluate", "predict", "optimize", "monitor"],
        help="Modo de ejecución (train | eval | predict | optimize | monitor)",
    )

    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=["gold_recovery_train.csv"],
        help="Archivos de datos de entrada",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/predictions.csv",
        help="Archivo de salida para predicciones",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/metallurgical_model.pkl",
        help="Ruta al modelo guardado",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="final.output.recovery",
        help="Columna objetivo para entrenamiento",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Archivo de configuración",
    )

    parser.add_argument(
        "--process",
        type=str,
        default="flotation",
        choices=["flotation", "concentration", "full_circuit"],
        help="Proceso a optimizar",
    )

    parser.add_argument(
        "--dashboard", action="store_true", help="Lanzar dashboard de monitoreo"
    )

    parser.add_argument("--port", type=int, default=8501, help="Puerto para dashboard")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla opcional (CLI > SEED env > 42)",
    )

    args = parser.parse_args()

    # Cargar configuración YAML si existe
    cfg: Dict[str, Any] = {}
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, "r") as f:
                raw_cfg = yaml.safe_load(f) or {}
                cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
                logger.info(f"Configuración cargada: {args.config}")
        except Exception as e:
            logger.warning(f"No se pudo cargar config YAML: {e}")

    # Determinar rutas de datos desde config si están disponibles
    train_path_cfg = cfg.get("data", {}).get("train_csv")
    test_path_cfg = cfg.get("data", {}).get("test_csv")
    full_path_cfg = cfg.get("data", {}).get("full_csv")
    if full_path_cfg:
        logger.info(f"Ruta 'full' en config: {full_path_cfg}")

    # Semilla global (CLI > SEED env > 42)
    seed = set_seed(args.seed)
    logger.info("Using seed: %s", seed)

    # Crear directorios necesarios
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    try:
        if args.mode == "train":
            logger.info("=== MODO ENTRENAMIENTO ===")

            # Cargar y procesar datos
            loader = ProcessDataLoader()
            input_paths = [train_path_cfg] if train_path_cfg is not None else args.input
            df = loader.load_process_data(input_paths)
            df_clean = loader.validate_and_clean_data(df)

            # Preparar datos
            predictor_cfg = (
                cfg.get("model") if isinstance(cfg.get("model"), dict) else None
            )
            predictor = MetallurgicalPredictor(config=predictor_cfg)
            X, y = predictor.prepare_features(df_clean, args.target)

            # Split train/test
            test_size = cfg.get("training", {}).get("test_size", 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )

            # Entrenar modelos
            cv_results = predictor.train(X_train, y_train)
            # Guardar resultados de CV
            Path("results").mkdir(exist_ok=True)
            with open(Path("results") / "cv_results.json", "w") as f:
                json.dump(cv_results, f, indent=2)

            # Evaluar en test
            test_metrics = predictor.evaluate(X_test, y_test)

            # Guardar modelo
            predictor.save_models(args.model)
            # Exportar paquete combinado para demo
            try:
                combined = {
                    "models": predictor.models,
                    "config": predictor.config,
                    "feature_columns": predictor.feature_columns,
                    "version": "1.0.0",
                }
                joblib.dump(combined, Path("models") / "model_v1.0.0.pkl")
            except Exception as e:
                logger.warning(f"No se pudo exportar paquete combinado: {e}")

            print("\n=== RESULTADOS DE ENTRENAMIENTO ===")
            print(f"sMAPE en test: {test_metrics['smape']:.2f}%")
            print(f"MAE en test: {test_metrics['mae']:.4f}")
            print(f"R² Score: {test_metrics['r2_score']:.4f}")

        elif args.mode == "predict":
            logger.info("=== MODO PREDICCIÓN ===")

            # Cargar modelo
            predictor = MetallurgicalPredictor()
            predictor.load_models(args.model)

            # Cargar datos para predicción
            loader = ProcessDataLoader()
            df = loader.load_process_data(args.input)
            df_clean = loader.validate_and_clean_data(df)

            # Realizar predicciones
            predictions = predictor.predict(df_clean)

            # Guardar resultados
            results_df = df_clean.copy()
            results_df["predicted_recovery"] = predictions
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(args.output, index=False)

            logger.info(f"Predicciones guardadas en: {args.output}")

        elif args.mode in ("evaluate", "eval"):
            logger.info("=== MODO EVALUACIÓN ===")

            # Cargar modelo
            predictor_cfg = (
                cfg.get("model") if isinstance(cfg.get("model"), dict) else None
            )
            predictor = MetallurgicalPredictor(config=predictor_cfg)
            predictor.load_models(args.model)

            # Cargar datos de test
            loader = ProcessDataLoader()
            eval_paths: List[str]
            if test_path_cfg is not None:
                eval_paths = [test_path_cfg]
            else:
                eval_paths = args.input
            df_test = loader.load_process_data(eval_paths)
            df_test_clean = loader.validate_and_clean_data(df_test)

            # Preparar y evaluar
            X_test, y_test = predictor.prepare_features(df_test_clean, args.target)
            metrics = predictor.evaluate(X_test, y_test)

            # Persistir métricas
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)

            with open(results_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            pd.DataFrame([metrics]).to_csv(results_dir / "metrics.csv", index=False)

            print("\n=== MÉTRICAS DE EVALUACIÓN ===")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

        elif args.mode == "optimize":
            logger.info("=== MODO OPTIMIZACIÓN ===")

            # Cargar modelo
            predictor = MetallurgicalPredictor()
            predictor.load_models(args.model)

            # Condiciones actuales de ejemplo
            current_conditions = {
                "rougher.feed.rate": 300.0,
                "primary_cleaner.input": 50.0,
                "secondary_cleaner.input": 25.0,
            }

            # Optimizar proceso
            optimizer = ProcessOptimizer(predictor)
            optimization_result = optimizer.optimize_process_parameters(
                current_conditions=current_conditions, target_recovery=92.0
            )

            print("\n=== RESULTADOS DE OPTIMIZACIÓN ===")
            print(
                f"Recovery predicho: {optimization_result['predicted_recovery']:.2f}%"
            )
            print(f"Mejora esperada: +{optimization_result['improvement']:.2f}%")
            print(
                f"Objetivo alcanzado: {'Sí' if optimization_result['target_achieved'] else 'No'}"
            )

        elif args.mode == "monitor":
            logger.info("=== MODO MONITOREO ===")

            if args.dashboard:
                # Lanzar dashboard Streamlit
                import subprocess

                subprocess.run(
                    [
                        "streamlit",
                        "run",
                        "app/streamlit_dashboard.py",
                        "--server.port",
                        str(args.port),
                    ]
                )
            else:
                logger.info("Monitoreo en modo consola no implementado")

    except Exception as e:
        logger.error(f"Error en ejecución: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
