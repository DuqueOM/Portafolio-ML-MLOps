from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="CarVision Dashboard", layout="wide")

DATA_PATH = "vehicles_us.csv"
if not Path(DATA_PATH).exists():
    st.error(f"Dataset {DATA_PATH} no encontrado.")
else:
    df = pd.read_csv(DATA_PATH)
    st.title("CarVision Market Intelligence Dashboard")
    st.caption("Exploración rápida del dataset")

    with st.expander("Muestra aleatoria", expanded=True):
        st.write(df.sample(min(10, len(df)), random_state=42))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de Precios")
        st.plotly_chart(px.histogram(df, x="price", nbins=50), use_container_width=True)
    with col2:
        st.subheader("Precio vs Millaje")
        if "odometer" in df.columns:
            st.plotly_chart(
                px.scatter(
                    df.sample(min(2000, len(df))), x="odometer", y="price", opacity=0.4
                ),
                use_container_width=True,
            )

    st.subheader("Precio por Año del Modelo")
    if "model_year" in df.columns:
        st.plotly_chart(
            px.line(
                df.groupby("model_year")["price"].mean().reset_index(),
                x="model_year",
                y="price",
            ),
            use_container_width=True,
        )
