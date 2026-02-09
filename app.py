import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Celestica IA", layout="wide")

st.title("🛡️ Celestica IA: Smart-Trace Analyzer")
st.write("Sube el archivo Excel de trazabilidad para un análisis con Machine Learning.")

uploaded_file = st.file_uploader("Arrastra tu Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        # Leer Excel
        df = pd.read_excel(uploaded_file)
        
        # Limpieza de fechas blindada
        df['In DateTime'] = pd.to_datetime(df['In DateTime'], errors='coerce')
        mask_short_year = df['In DateTime'].dt.year < 100
        df.loc[mask_short_year, 'In DateTime'] += pd.offsets.DateOffset(years=2000)
        df = df.dropna(subset=['In DateTime']).sort_values('In DateTime')

        # Cálculo de intervalos
        df['gap_mins'] = df.groupby('Station')['In DateTime'].diff().dt.total_seconds() / 60
        df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

        # MACHINE LEARNING: Detección de anomalías
        model = IsolationForest(contamination=0.05, random_state=42)
        df['IA_Status'] = model.fit_predict(df[['gap_mins']])
        
        # Filtrado para el Dashboard
        df_normal = df[df['IA_Status'] == 1]
        q1, q3 = df_normal['gap_mins'].quantile([0.25, 0.75])
        df_clean = df_normal[(df_normal['gap_mins'] >= q1) & (df_normal['gap_mins'] <= q3)]
        media_ct = df_clean['gap_mins'].mean()

        # KPIs Visuales
        c1, c2, c3 = st.columns(3)
        c1.metric("Cycle Time Real", f"{media_ct:.2f} min")
        c2.metric("Salud del Dato", f"{(len(df_clean)/len(df)*100):.1f}%")
        c3.metric("Capacidad Real (8h)", f"{int((415/media_ct)*0.75)} uds")

        # Gráfico Interactivo
        st.subheader("📊 Mapa de Producción (Verde: OK | Rojo: Anomalía)")
        fig = px.scatter(df, x='In DateTime', y='gap_mins', color=df['IA_Status'].astype(str),
                         color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'},
                         labels={'IA_Status': 'Estado IA', 'gap_mins': 'Minutos/Pieza'})
        st.plotly_chart(fig, use_container_width=True)

        # Botón Descarga
        st.download_button("📥 Descargar Reporte Limpio", df_clean.to_csv(index=False).encode('utf-8'), "reporte_ia.csv")
        
    except Exception as e:
        st.error(f"Error: {e}")
