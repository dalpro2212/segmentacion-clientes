import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime as dt
import plotly.express as px

st.set_page_config(page_title="Segmentación de Clientes", page_icon="🛒", layout="wide")
st.title("🛒 Segmentación de Clientes en Tiempo Real")

st.sidebar.header("⚙️ Configuración")
k = st.sidebar.slider("Número de Clusters (K)", min_value=2, max_value=10, value=3)

uploaded_file = st.file_uploader("online_retail_lite.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,)
    st.success(f"Archivo cargado: {df.shape[0]} filas")
    st.dataframe(df.head())

    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    fecha_ref = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recencia=("InvoiceDate", lambda x: (fecha_ref - x.max()).days),
        Frecuencia=("InvoiceNo", "nunique"),
        Monto=("TotalPrice", "sum")
    ).reset_index()

    if st.button("🚀 Ejecutar K-Means"):
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[["Recencia", "Frecuencia", "Monto"]])

        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        rfm["Cluster"] = modelo.fit_predict(rfm_scaled)

        st.success(f"¡Listo! {k} clusters encontrados.")
        st.dataframe(rfm.head())

        # Gráfico de dispersión interactivo
        fig = px.scatter(
            rfm, x="Recencia", y="Frecuencia",
            color=rfm["Cluster"].astype(str),
            title="Clusters de Clientes",
            labels={"color": "Cluster"}
        )

        # Centroides
        centroides = scaler.inverse_transform(modelo.cluster_centers_)
        for i, centro in enumerate(centroides):
            fig.add_scatter(
                x=[centro[0]], y=[centro[1]],
                mode="markers",
                marker=dict(symbol="x", size=15, color="black"),
                name=f"Centroide {i}"
            )

        st.plotly_chart(fig, use_container_width=True)

        # Métricas por segmento
        st.markdown("### 📊 Métricas por Segmento")
        cluster_sel = st.selectbox("Selecciona un Cluster:", sorted(rfm["Cluster"].unique()))

        subset = rfm[rfm["Cluster"] == cluster_sel][["Recencia", "Frecuencia", "Monto"]]

        col1, col2 = st.columns(2)
        col1.metric("Varianza promedio", round(subset.var().mean(), 2))
        col2.metric("Desviación Estándar promedio", round(subset.std().mean(), 2))

        st.dataframe(subset.describe().round(2))
