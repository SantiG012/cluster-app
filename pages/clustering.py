import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

expected_df = pd.read_csv('./datasets/df.csv')
expected_columns = expected_df.drop(['CUST_ID', 'Cluster'], axis=1).columns
k = 3
is_k_selected = False

def convert_uploaded_file(uploaded_file) -> pd.DataFrame:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        return uploaded_df
    except Exception as e:
        raise TypeError(f"El archivo no es un CSV válido: {e}.")


def validate(uploaded_df) -> None:

    for expected_column in expected_columns:
        if expected_column not in expected_df.columns:
           raise KeyError(f"La columna {expected_column} no se encuentra en el dataset.")
    
    if len(uploaded_df.columns) != len(expected_columns):
        raise KeyError("El dataset contiene columnas adicionales.")
    
    # Validate if all columns are numeric
    for column in uploaded_df.columns:
        if not pd.api.types.is_numeric_dtype(uploaded_df[column]):
            raise ValueError(f"La columna {column} no es numérica.")
        

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

def reduce_dimensionality(df: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(df)
    return pd.DataFrame(reduced_data, columns=['PC1', 'PC2','PC3'])

def plot_elbow_curve(df: pd.DataFrame) -> None:
    global k
    toggle_elbow = st.toggle("Gráfica del Codo (Métrica Whitin-Cluster Sum of Squares)", value=False)
    model = KMeans()
    wcss = []

    for i in range(1, 11):
        model = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        model.fit(df)
        wcss.append(model.inertia_)

    if toggle_elbow:

        
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss)
        ax.set_xlabel('Número de Clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)
    
    k = ask_for_k()

def ask_for_k() -> int:
    global is_k_selected
    k = st.number_input("Número de Clusters (3 por defecto)", min_value=2, max_value=11, value=3)

    if k is None:
        raise ValueError("El número de clusters no puede ser nulo.")
    if not isinstance(k, int):
        raise ValueError("El número de clusters debe ser un entero.")
    
    is_k_selected = True
    return k

def train_model(df: pd.DataFrame) -> KMeans:
    model = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    model.fit(df)
    return model

def plot_clusters(df: pd.DataFrame, model: KMeans) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.scatter(df['PC1'], df['PC2'],df['PC3'], c=model.labels_, cmap='rainbow')
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], model.cluster_centers_[:, 2], s=300, c='yellow', label='Centroids')
    ax.legend()

    toggle_clusters = st.toggle("Ver Clusters", value=False)

    if toggle_clusters:
        st.pyplot(fig)

def plot_pairplot(df: pd.DataFrame) -> None:
    pallete = sns.color_palette('tab10')[:k]
    toggle_pairplot = st.toggle("Ver Pairplot (Toma bastante tiempo. Se usa para identificar los scatter plots más relevantes)", value=False)
    if toggle_pairplot:
        st.pyplot(sns.pairplot(df, hue='Cluster', palette=pallete))

def plot_scatter_plot(df: pd.DataFrame) -> None:
    column_a = ""
    column_b = ""
    are_columns_different = False

    toggle_scatter = st.toggle("Ver Scatter Plot")

    if toggle_scatter:
        column_a = st.selectbox("Seleccionar columna A", df.columns)
        column_b = st.selectbox("Seleccionar columna B", df.columns)

        if column_a == column_b:
            st.warning("Seleccione columnas diferentes")
        
        if column_a != column_b:
            are_columns_different = True

            
        
        if are_columns_different:
            fig, ax = plt.subplots()
            ax.scatter(df[column_a], df[column_b], c=df['Cluster'])
            ax.set_xlabel(column_a)
            ax.set_ylabel(column_b)
            st.pyplot(fig)

st.title("Clustering")
st.markdown("""
    **NOTA:** Ser ***PACIENTE*** con el tiempo de carga de los gráficos. El dataset es ***GRANDE***, los cálculos pueden tardar. Adicional a ello, el servidor host tiene ***LIMITACIONES*** de recursos.
""")
uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])

if uploaded_file:
    try:
        uploaded_df = convert_uploaded_file(uploaded_file)
        validate(uploaded_df)
    except TypeError as e:
        error_message = e.args[0] if len(e.args) > 0 else "Error"
        st.warning(f"{error_message} Por favor, suba un archivo CSV.")
    except KeyError as e:
        error_message = e.args[0] if len(e.args) > 0 else "Error"
        st.warning(f"{error_message} Por favor, suba un que contenga cada una de las siguientes columnas: {','.join(expected_columns)}")
    except ValueError as e:
        error_message = e.args[0] if len(e.args) > 0 else "Error"
        st.warning(f"{error_message} Todas las columnas deben ser numéricas")
    else:
        scaled_df = scale_features(uploaded_df)
        reduced_df = reduce_dimensionality(scaled_df)
        plot_elbow_curve(reduced_df)
        model = train_model(reduced_df)
        plot_clusters(reduced_df, model)
        uploaded_df['Cluster'] = model.labels_
        plot_pairplot(uploaded_df)
        plot_scatter_plot(uploaded_df)
