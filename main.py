import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

DF_PATH = 'datasets/df.csv'
df = pd.read_csv(DF_PATH)

st.title('Clustering para el uso de tarjetas de crédito')

st.write(
    """
    Se ha encontrado un conjunto de datos con mas de 8,000 registros pertenecientes a compras con tarjetas de crédito.
    Se desea realizar un análisis de clustering para identificar patrones de comportamiento en los clientes. 
    Para conocer el dataset, puede consultarlo [aquí](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).
    """
)	

st.title('Exploración de datos')
st.write('Algunos gráficos para visualizar los datos. El proceso completo lo puede encontrar [aquí](https://colab.research.google.com/drive/1CceuYFzsgEIK9lRHF-XPzUAteXCwN_KK?usp=sharing).')

fig, ax = plt.subplots()

df.hist(figsize=(50,50), ax=ax)

st.pyplot(fig=fig)