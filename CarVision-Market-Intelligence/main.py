#!/usr/bin/env python3
"""
CarVision Market Intelligence - Sistema de análisis de mercado automotriz

Uso:
    python main.py --mode analysis --input data/raw/vehicles_us.csv
    python main.py --mode dashboard --port 8501
    python main.py --mode report --output reports/market_analysis.html
    python main.py --mode export --format excel --output market_data.xlsx

Autor: Daniel Duque
Versión: 1.0.0
Fecha: 2024-11-16
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

# ML utilities from project
from data.preprocess import (
    build_preprocessor,
    infer_feature_types,
    save_split_indices,
    split_data,
)
from data.preprocess import clean_data as ds_clean_data
from data.preprocess import load_data as ds_load_data
from evaluate import evaluate_model as eval_model
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Configuración de warnings y logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("carvision.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class VehicleDataLoader:
    """Cargador y validador de datos de vehículos."""

    def __init__(self):
        self.required_columns = [
            "price",
            "model_year",
            "model",
            "condition",
            "cylinders",
            "fuel",
            "odometer",
            "transmission",
            "drive",
            "size",
            "type",
            "paint_color",
        ]

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga y valida el dataset de vehículos.

        Args:
            file_path: Ruta al archivo CSV

        Returns:
            DataFrame con los datos cargados y validados
        """
        logger.info(f"Cargando datos desde: {file_path}")

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

            # Validar columnas requeridas
            missing_cols = [
                col for col in self.required_columns if col not in df.columns
            ]
            if missing_cols:
                logger.warning(f"Columnas faltantes: {missing_cols}")

            # Información básica del dataset
            logger.info(
                f"Rango de precios: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}"
            )
            logger.info(
                f"Años de modelo: {df['model_year'].min()} - {df['model_year'].max()}"
            )

            return df

        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y preprocesa los datos de vehículos.

        Args:
            df: DataFrame con datos raw

        Returns:
            DataFrame limpio
        """
        logger.info("Iniciando limpieza de datos")

        df_clean = df.copy()

        # Filtrar precios válidos
        df_clean = df_clean[(df_clean["price"] > 1000) & (df_clean["price"] < 500000)]

        # Filtrar años válidos
        current_year = pd.Timestamp.now().year
        df_clean = df_clean[
            (df_clean["model_year"] >= 1990) & (df_clean["model_year"] <= current_year)
        ]

        # Limpiar odometer
        if "odometer" in df_clean.columns:
            df_clean = df_clean[
                (df_clean["odometer"] > 0) & (df_clean["odometer"] < 500000)
            ]

        # Crear features derivadas
        df_clean["vehicle_age"] = current_year - df_clean["model_year"]
        df_clean["price_per_mile"] = df_clean["price"] / (df_clean["odometer"] + 1)

        # Categorizar precios
        df_clean["price_category"] = pd.cut(
            df_clean["price"],
            bins=[0, 10000, 25000, 50000, float("inf")],
            labels=["Budget", "Mid-Range", "Premium", "Luxury"],
        )

        logger.info(f"Datos después de limpieza: {df_clean.shape[0]} filas")

        return df_clean


class MarketAnalyzer:
    """Analizador de mercado automotriz."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analysis_results: Dict[str, Any] = {}

    def analyze_price_distribution(self) -> Dict[str, Any]:
        """Analiza la distribución de precios."""
        logger.info("Analizando distribución de precios")

        price_stats = {
            "mean": self.df["price"].mean(),
            "median": self.df["price"].median(),
            "std": self.df["price"].std(),
            "min": self.df["price"].min(),
            "max": self.df["price"].max(),
            "q25": self.df["price"].quantile(0.25),
            "q75": self.df["price"].quantile(0.75),
        }

        # Distribución por categoría de precio
        price_dist = self.df["price_category"].value_counts()

        self.analysis_results["price_distribution"] = {
            "statistics": price_stats,
            "distribution": price_dist.to_dict(),
        }

        return self.analysis_results["price_distribution"]

    def analyze_market_by_brand(self) -> Dict[str, Any]:
        """Analiza el mercado por marca."""
        logger.info("Analizando mercado por marca")

        # Top marcas por volumen
        brand_volume = self.df["model"].str.split().str[0].value_counts().head(10)

        # Precio promedio por marca
        self.df["brand"] = self.df["model"].str.split().str[0]
        brand_price = (
            self.df.groupby("brand")["price"].agg(["mean", "median", "count"]).round(0)
        )
        brand_price = brand_price[brand_price["count"] >= 100].sort_values(
            "mean", ascending=False
        )

        self.analysis_results["market_by_brand"] = {
            "volume": brand_volume.to_dict(),
            "pricing": brand_price.to_dict(),
        }

        return self.analysis_results["market_by_brand"]

    def analyze_depreciation_patterns(self) -> Dict[str, Any]:
        """Analiza patrones de depreciación."""
        logger.info("Analizando patrones de depreciación")

        # Depreciación por edad
        depreciation = self.df.groupby("vehicle_age")["price"].mean().sort_index()

        # Tasa de depreciación anual
        depreciation_rate = depreciation.pct_change().fillna(0) * -1

        self.analysis_results["depreciation"] = {
            "by_age": depreciation.to_dict(),
            "annual_rate": depreciation_rate.to_dict(),
        }

        return self.analysis_results["depreciation"]

    def find_market_opportunities(self) -> List[Dict[str, Any]]:
        """Identifica oportunidades de mercado."""
        logger.info("Identificando oportunidades de mercado")

        opportunities = []

        # Vehículos subvalorados (precio < percentil 25 para su categoría)
        for category in self.df["price_category"].unique():
            if pd.isna(category):
                continue

            category_data = self.df[self.df["price_category"] == category]
            price_threshold = category_data["price"].quantile(0.25)

            undervalued = category_data[
                (category_data["price"] < price_threshold)
                & (category_data["vehicle_age"] <= 10)
                & (category_data["odometer"] <= 100000)
            ]

            if len(undervalued) > 0:
                opportunities.append(
                    {
                        "category": category,
                        "count": len(undervalued),
                        "avg_price": undervalued["price"].mean(),
                        "potential_value": category_data["price"].median()
                        - undervalued["price"].mean(),
                    }
                )

        self.analysis_results["opportunities"] = opportunities

        return opportunities

    def generate_executive_summary(self) -> Dict[str, Any]:
        """Genera resumen ejecutivo del análisis."""
        logger.info("Generando resumen ejecutivo")

        # Ejecutar todos los análisis si no se han ejecutado
        if "price_distribution" not in self.analysis_results:
            self.analyze_price_distribution()
        if "market_by_brand" not in self.analysis_results:
            self.analyze_market_by_brand()
        if "depreciation" not in self.analysis_results:
            self.analyze_depreciation_patterns()
        if "opportunities" not in self.analysis_results:
            self.find_market_opportunities()

        # KPIs principales
        total_vehicles = len(self.df)
        avg_price = self.df["price"].mean()
        total_market_value = self.df["price"].sum()

        # Oportunidades identificadas
        total_opportunities = sum(
            [opp["count"] for opp in self.analysis_results["opportunities"]]
        )
        potential_value = sum(
            [
                opp["potential_value"] * opp["count"]
                for opp in self.analysis_results["opportunities"]
            ]
        )

        summary = {
            "kpis": {
                "total_vehicles": total_vehicles,
                "average_price": avg_price,
                "total_market_value": total_market_value,
                "total_opportunities": total_opportunities,
                "potential_arbitrage_value": potential_value,
            },
            "insights": {
                "most_popular_brand": list(
                    self.analysis_results["market_by_brand"]["volume"].keys()
                )[0],
                "highest_value_brand": list(
                    self.analysis_results["market_by_brand"]["pricing"]["mean"].keys()
                )[0],
                "avg_depreciation_rate": np.mean(
                    list(self.analysis_results["depreciation"]["annual_rate"].values())
                ),
            },
            "recommendations": [
                f"Focus on {total_opportunities} undervalued vehicles for potential ${potential_value:,.0f} profit",
                (
                    f"Target {list(self.analysis_results['market_by_brand']['volume'].keys())[0]} brand "
                    "for volume opportunities"
                ),
                "Implement dynamic pricing based on vehicle age and market conditions",
            ],
        }

        self.analysis_results["executive_summary"] = summary

        return summary


class VisualizationEngine:
    """Motor de visualizaciones para análisis de mercado."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def create_price_distribution_chart(self) -> go.Figure:
        """Crea gráfico de distribución de precios."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Histograma de Precios",
                "Box Plot por Categoría",
                "Precios por Año",
                "Top 10 Marcas",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Histograma
        fig.add_trace(
            go.Histogram(x=self.df["price"], nbinsx=50, name="Distribución"),
            row=1,
            col=1,
        )

        # Box plot por categoría
        for category in self.df["price_category"].unique():
            if pd.notna(category):
                fig.add_trace(
                    go.Box(
                        y=self.df[self.df["price_category"] == category]["price"],
                        name=str(category),
                    ),
                    row=1,
                    col=2,
                )

        # Precios por año
        yearly_prices = self.df.groupby("model_year")["price"].mean()
        fig.add_trace(
            go.Scatter(
                x=yearly_prices.index,
                y=yearly_prices.values,
                mode="lines+markers",
                name="Precio Promedio",
            ),
            row=2,
            col=1,
        )

        # Top marcas
        self.df["brand"] = self.df["model"].str.split().str[0]
        top_brands = self.df["brand"].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=top_brands.values, y=top_brands.index, orientation="h", name="Volumen"
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800, title_text="Análisis de Precios del Mercado Automotriz"
        )

        return fig

    def create_market_analysis_dashboard(self) -> go.Figure:
        """Crea dashboard completo de análisis de mercado."""
        # Crear análisis
        analyzer = MarketAnalyzer(self.df)
        summary = analyzer.generate_executive_summary()

        # Crear figura con subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "KPIs Principales",
                "Oportunidades por Categoría",
                "Depreciación por Edad",
                "Market Share por Marca",
                "Precio vs Millaje",
                "Distribución por Condición",
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
        )

        # KPIs
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=summary["kpis"]["average_price"],
                title={"text": "Precio Promedio"},
                number={"prefix": "$", "valueformat": ",.0f"},
            ),
            row=1,
            col=1,
        )

        # Oportunidades
        opportunities = analyzer.analysis_results["opportunities"]
        if opportunities:
            categories = [opp["category"] for opp in opportunities]
            values = [opp["potential_value"] for opp in opportunities]
            fig.add_trace(
                go.Bar(x=categories, y=values, name="Valor Potencial"), row=1, col=2
            )

        # Depreciación
        depreciation = analyzer.analysis_results["depreciation"]["by_age"]
        ages = list(depreciation.keys())
        prices = list(depreciation.values())
        fig.add_trace(
            go.Scatter(x=ages, y=prices, mode="lines+markers", name="Depreciación"),
            row=2,
            col=1,
        )

        # Market share
        brand_volume = analyzer.analysis_results["market_by_brand"]["volume"]
        brands = list(brand_volume.keys())[:5]  # Top 5
        volumes = [brand_volume[brand] for brand in brands]
        fig.add_trace(
            go.Pie(labels=brands, values=volumes, name="Market Share"), row=2, col=2
        )

        # Precio vs Millaje
        sample_data = self.df.sample(
            min(1000, len(self.df))
        )  # Muestra para performance
        fig.add_trace(
            go.Scatter(
                x=sample_data["odometer"],
                y=sample_data["price"],
                mode="markers",
                name="Precio vs Millaje",
                opacity=0.6,
            ),
            row=3,
            col=1,
        )

        # Distribución por condición
        condition_counts = self.df["condition"].value_counts()
        fig.add_trace(
            go.Bar(
                x=condition_counts.index,
                y=condition_counts.values,
                name="Por Condición",
            ),
            row=3,
            col=2,
        )

        fig.update_layout(
            height=1200, title_text="CarVision Market Intelligence Dashboard"
        )

        return fig


class ReportGenerator:
    """Generador de reportes automáticos."""

    def __init__(self, analyzer: MarketAnalyzer):
        self.analyzer = analyzer

    def generate_html_report(self, output_path: str) -> None:
        """Genera reporte HTML completo."""
        logger.info(f"Generando reporte HTML: {output_path}")

        # Obtener análisis completo
        summary = self.analyzer.generate_executive_summary()

        # Template HTML básico
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CarVision Market Intelligence Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .kpi {{ background: #f0f0f0; padding: 20px; margin: 10px 0; border-radius: 5px; }}
                .insight {{ background: #e8f4fd; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>🚗 CarVision Market Intelligence Report</h1>
            <p><strong>Fecha de generación:</strong> {timestamp}</p>
            
            <h2>📊 KPIs Principales</h2>
            <div class="kpi">
                <h3>Métricas del Mercado</h3>
                <ul>
                    <li><strong>Total de Vehículos:</strong> {total_vehicles:,}</li>
                    <li><strong>Precio Promedio:</strong> ${avg_price:,.0f}</li>
                    <li><strong>Valor Total del Mercado:</strong> ${total_value:,.0f}</li>
                    <li><strong>Oportunidades Identificadas:</strong> {opportunities}</li>
                    <li><strong>Valor Potencial de Arbitraje:</strong> ${arbitrage_value:,.0f}</li>
                </ul>
            </div>
            
            <h2>💡 Insights Clave</h2>
            <div class="insight">
                <h3>Análisis de Mercado</h3>
                <ul>
                    <li><strong>Marca Más Popular:</strong> {popular_brand}</li>
                    <li><strong>Marca de Mayor Valor:</strong> {valuable_brand}</li>
                    <li><strong>Tasa de Depreciación Promedio:</strong> {depreciation_rate:.1%}</li>
                </ul>
            </div>
            
            <h2>🎯 Recomendaciones</h2>
            <ul>
        """.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_vehicles=summary["kpis"]["total_vehicles"],
            avg_price=summary["kpis"]["average_price"],
            total_value=summary["kpis"]["total_market_value"],
            opportunities=summary["kpis"]["total_opportunities"],
            arbitrage_value=summary["kpis"]["potential_arbitrage_value"],
            popular_brand=summary["insights"]["most_popular_brand"],
            valuable_brand=summary["insights"]["highest_value_brand"],
            depreciation_rate=summary["insights"]["avg_depreciation_rate"],
        )

        # Agregar recomendaciones
        for rec in summary["recommendations"]:
            html_template += f"<li>{rec}</li>"

        html_template += """
            </ul>
            
            <h2>📈 Oportunidades de Mercado</h2>
            <table>
                <tr>
                    <th>Categoría</th>
                    <th>Cantidad</th>
                    <th>Precio Promedio</th>
                    <th>Valor Potencial</th>
                </tr>
        """

        # Agregar oportunidades
        for opp in self.analyzer.analysis_results["opportunities"]:
            html_template += f"""
                <tr>
                    <td>{opp['category']}</td>
                    <td>{opp['count']}</td>
                    <td>${opp['avg_price']:,.0f}</td>
                    <td>${opp['potential_value']:,.0f}</td>
                </tr>
            """

        html_template += """
            </table>
            
            <footer>
                <p><em>Reporte generado por CarVision Market Intelligence v1.0.0</em></p>
            </footer>
        </body>
        </html>
        """

        # Guardar reporte
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_template)

        logger.info(f"Reporte HTML guardado en: {output_path}")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths = cfg["paths"]
    tr = cfg["training"]
    prep = cfg["preprocessing"]

    Path(paths["artifacts_dir"]).mkdir(parents=True, exist_ok=True)

    # Load & clean data
    df = ds_clean_data(ds_load_data(paths["data_path"]))

    # Infer features
    num_cols, cat_cols = infer_feature_types(
        df,
        target=tr["target"],
        numeric_features=prep.get("numeric_features") or None,
        categorical_features=prep.get("categorical_features") or None,
        drop_columns=prep.get("drop_columns") or None,
    )

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test, split_indices = split_data(
        df, tr["target"], tr["test_size"], tr["val_size"], cfg["seed"], tr["shuffle"]
    )
    save_split_indices(split_indices, paths["split_indices_path"])

    # Preprocessor
    pre = build_preprocessor(
        num_cols,
        cat_cols,
        numeric_imputer=prep.get("numeric_imputer", "median"),
        categorical_imputer=prep.get("categorical_imputer", "most_frequent"),
        scale_numeric=prep.get("scale_numeric", True),
        handle_unknown=prep.get("handle_unknown_category", "ignore"),
    )

    # Model
    if tr.get("model") == "random_forest":
        rf_params = tr.get("random_forest_params", {})
        # ensure reproducibility
        if "random_state" not in rf_params:
            rf_params["random_state"] = cfg["seed"]
        model = RandomForestRegressor(**rf_params)
    else:
        raise NotImplementedError(f"Modelo no soportado: {tr.get('model')}")

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    logger.info("Entrenando modelo...")
    pipe.fit(X_train, y_train)

    # Validation metrics
    yv = pipe.predict(X_val)
    val_metrics = {
        "rmse": rmse(y_val, yv),
        "mae": float(mean_absolute_error(y_val, yv)),
        "mape": float(
            np.mean(np.abs((np.array(y_val) - yv) / (np.array(y_val) + 1e-8))) * 100
        ),
        "r2": float(r2_score(y_val, yv)),
    }
    logger.info(f"Métricas de validación: {val_metrics}")

    # Persist artifacts
    joblib.dump(pipe, paths["model_path"])
    # Export a copy for demo loading
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/model_v1.0.0.pkl")
    feature_columns = sorted(num_cols + cat_cols)
    with open(Path(paths["artifacts_dir"]) / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)
    with open(Path(paths["artifacts_dir"]) / "metrics_val.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    return {
        "val_metrics": val_metrics,
        "model_path": paths["model_path"],
        "feature_columns": feature_columns,
    }


def main():
    """Función principal con CLI."""
    parser = argparse.ArgumentParser(
        description="CarVision Market Intelligence - Análisis de mercado automotriz"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "analysis",
            "dashboard",
            "report",
            "export",
            "train",
            "eval",
            "predict",
        ],
        help="Modo de ejecución",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/vehicles_us.csv",
        help="Ruta al archivo de datos de entrada",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Ruta de salida para reportes/exports",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="html",
        choices=["html", "excel", "json"],
        help="Formato de salida",
    )

    parser.add_argument(
        "--port", type=int, default=8501, help="Puerto para dashboard Streamlit"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Archivo de configuración",
    )

    parser.add_argument(
        "--input_json",
        type=str,
        default=None,
        help="Ruta a JSON con payload para modo predict",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla para reproducibilidad (sobrescribe config)",
    )

    args = parser.parse_args()

    # Crear directorios necesarios
    Path("reports").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    try:
        if args.mode == "analysis":
            logger.info("=== MODO ANÁLISIS ===")

            # Cargar y limpiar datos
            loader = VehicleDataLoader()
            df = loader.load_data(args.input)
            df_clean = loader.clean_data(df)

            # Realizar análisis completo
            analyzer = MarketAnalyzer(df_clean)
            summary = analyzer.generate_executive_summary()

            # Mostrar resultados
            print("\n=== RESUMEN EJECUTIVO ===")
            print(
                f"Total de vehículos analizados: {summary['kpis']['total_vehicles']:,}"
            )
            print(f"Precio promedio: ${summary['kpis']['average_price']:,.0f}")
            print(
                f"Oportunidades identificadas: {summary['kpis']['total_opportunities']}"
            )
            print(
                f"Valor potencial de arbitraje: ${summary['kpis']['potential_arbitrage_value']:,.0f}"
            )

            print("\n=== INSIGHTS CLAVE ===")
            for key, value in summary["insights"].items():
                print(f"{key}: {value}")

            print("\n=== RECOMENDACIONES ===")
            for i, rec in enumerate(summary["recommendations"], 1):
                print(f"{i}. {rec}")

        elif args.mode == "dashboard":
            logger.info("=== MODO DASHBOARD ===")

            # Importar y ejecutar dashboard Streamlit
            import subprocess

            subprocess.run(
                [
                    "streamlit",
                    "run",
                    "app/streamlit_app.py",
                    "--server.port",
                    str(args.port),
                ]
            )

        elif args.mode == "report":
            logger.info("=== MODO REPORTE ===")

            # Cargar datos y generar análisis
            loader = VehicleDataLoader()
            df = loader.load_data(args.input)
            df_clean = loader.clean_data(df)

            analyzer = MarketAnalyzer(df_clean)

            # Generar reporte
            report_gen = ReportGenerator(analyzer)

            if args.format == "html":
                output_file = (
                    f"{args.output}.html"
                    if not args.output.endswith(".html")
                    else args.output
                )
                report_gen.generate_html_report(output_file)

            logger.info(f"Reporte generado: {output_file}")

        elif args.mode == "export":
            logger.info("=== MODO EXPORT ===")

            # Cargar y procesar datos
            loader = VehicleDataLoader()
            df = loader.load_data(args.input)
            df_clean = loader.clean_data(df)

            # Exportar según formato
            if args.format == "excel":
                output_file = (
                    f"{args.output}.xlsx"
                    if not args.output.endswith(".xlsx")
                    else args.output
                )
                df_clean.to_excel(output_file, index=False)
            elif args.format == "json":
                output_file = (
                    f"{args.output}.json"
                    if not args.output.endswith(".json")
                    else args.output
                )
                df_clean.to_json(output_file, orient="records", indent=2)

            logger.info(f"Datos exportados: {output_file}")

        elif args.mode == "train":
            logger.info("=== MODO TRAIN ===")
            cfg = load_config(args.config)
            if args.seed is not None:
                cfg["seed"] = int(args.seed)
            result = train_model(cfg)
            logger.info(f"Modelo guardado en: {result['model_path']}")
            print(json.dumps(result["val_metrics"], indent=2))

        elif args.mode == "eval":
            logger.info("=== MODO EVAL ===")
            cfg = load_config(args.config)
            if args.seed is not None:
                cfg["seed"] = int(args.seed)
            results = eval_model(cfg)
            print(json.dumps(results, indent=2))

        elif args.mode == "predict":
            logger.info("=== MODO PREDICT ===")
            if not args.input_json or not Path(args.input_json).exists():
                raise FileNotFoundError("Debe especificar --input_json con ruta válida")
            cfg = load_config(args.config)
            if args.seed is not None:
                cfg["seed"] = int(args.seed)
            paths = cfg["paths"]
            model = joblib.load(paths["model_path"])
            # load feature columns
            feat_path = Path(paths["artifacts_dir"]) / "feature_columns.json"
            if feat_path.exists():
                feature_columns = json.loads(Path(feat_path).read_text())
            else:
                # fallback: try to introspect ColumnTransformer
                try:
                    pre = model.named_steps["pre"]
                    feature_columns = list(pre.transformers_[0][2]) + list(
                        pre.transformers_[1][2]
                    )
                except Exception:
                    raise RuntimeError(
                        "No se pudo determinar columnas de features. Falta artifacts/feature_columns.json"
                    )

            payload = json.loads(Path(args.input_json).read_text())
            df_in = pd.DataFrame([payload])
            # align columns
            for col in feature_columns:
                if col not in df_in.columns:
                    df_in[col] = np.nan
            df_in = df_in[feature_columns]
            pred = model.predict(df_in)
            out = {"prediction": float(pred[0])}
            print(json.dumps(out, indent=2))

    except Exception as e:
        logger.error(f"Error en ejecución: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
