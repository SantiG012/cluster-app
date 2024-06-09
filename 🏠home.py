import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./datasets/df.csv')
columns = df.drop(['CUST_ID', 'Cluster'], axis=1).columns

st.markdown(f"""
    # Instrucciones de Uso
    1. Dirigirse a la sección de **Clustering** en el *menú*.
    2. Cargar un dataset en formato CSV con las siguientes columnas: 
""")

st.write(columns)

st.markdown("""
    **NOTAS**:
    """)
st.markdown("""
        - ***TODAS*** las columnas deben ser numéricas.
        - El dataset debe contener ***TODAS*** las columnas antes mencionadas. Tampoco acepta columnas adicionales.
        - Los resultados se encuentran en botones plegables debido a la cantidad de información.
    """)
