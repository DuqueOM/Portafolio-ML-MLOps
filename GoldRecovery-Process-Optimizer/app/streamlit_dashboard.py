from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RESULTS_PATH = Path("results/predictions.csv")
DEFAULT_TRAIN = Path("gold_recovery_train.csv")

st.set_page_config(page_title="GoldRecovery Dashboard", layout="wide")

st.title("GoldRecovery - Monitoreo de Recovery")

if RESULTS_PATH.exists():
    df = pd.read_csv(RESULTS_PATH)
    st.success(f"Cargando resultados: {RESULTS_PATH}")
elif DEFAULT_TRAIN.exists():
    df = pd.read_csv(DEFAULT_TRAIN)
    st.info("Mostrando datos de entrenamiento (sin resultados de prediccion)")
else:
    st.error("No hay datos disponibles")
    st.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox(
        "Columna objetivo",
        options=[c for c in df.columns if "recovery" in c],
        index=0 if any("final.output.recovery" == c for c in df.columns) else 0,
    )
with col2:
    var = st.selectbox(
        "Variable a inspeccionar", options=[c for c in numeric_cols if c != target_col]
    )

st.line_chart(df[[target_col]].rename(columns={target_col: "target"}))

if var:
    st.area_chart(df[[var]].rename(columns={var: "feature"}))
