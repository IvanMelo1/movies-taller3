import pandas as pd
import streamlit as st
import umap
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import ast  # Para convertir las cadenas de embeddings a listas


def generar_visualizaciones():
    st.header("Visualizaciones Interactivas de Embeddings")

    # Cargar los datos procesados con los embeddings
    st.write("Cargando datos procesados...")
    try:
        df = pd.read_csv("data/processed/embedded_dataset.csv")
        df['embedding'] = df['embedding'].apply(ast.literal_eval)  # Convertir cadenas a listas
        embeddings = np.array(df['embedding'].tolist())  # Convertir a numpy array
        st.success("Datos cargados correctamente.")
    except Exception as e:
        st.error(f"Error cargando los datos: {e}")
        return

    # Mostrar información básica del dataset
    st.write("Vista previa del dataset procesado:")
    st.dataframe(df[['Title', 'Genre', 'IMDb Rating', 'Year']].head())

    try:
        st.write("Distribución de películas por género:")
        distribucion_clases = df['Genre'].value_counts()  # Calcular la cantidad por género
        st.bar_chart(distribucion_clases)  # Mostrar el gráfico de barras en Streamlit

        # Tabla adicional con la distribución para mayor claridad
        st.write("Distribución detallada:")
        st.dataframe(distribucion_clases.reset_index().rename(columns={'index': 'Genre', 'Genre': 'Count'}))
    except Exception as e:
        st.error(f"Error mostrando la distribución por género: {e}")

    # Verificar que el número de muestras sea suficiente para t-SNE
    if embeddings.shape[0] < 30:
        st.warning("El dataset tiene menos de 30 muestras. Añade más datos para visualizar con t-SNE.")
        return

    # Reducir dimensiones usando UMAP
    try:
        st.write("Reduciendo dimensiones con UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_result = reducer.fit_transform(embeddings)
        df['UMAP_1'] = umap_result[:, 0]
        df['UMAP_2'] = umap_result[:, 1]
        st.success("UMAP completado.")
    except Exception as e:
        st.error(f"Error en UMAP: {e}")

    # Reducir dimensiones usando PCA
    try:
        st.write("Reduciendo dimensiones con PCA...")
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(embeddings)
        df['PCA_1'] = pca_result[:, 0]
        df['PCA_2'] = pca_result[:, 1]
        st.success("PCA completado.")
    except Exception as e:
        st.error(f"Error en PCA: {e}")

    # Reducir dimensiones usando t-SNE
    try:
        st.write("Reduciendo dimensiones con t-SNE...")
        tsne = TSNE(n_components=2, perplexity=20, random_state=42)
        tsne_result = tsne.fit_transform(embeddings)
        df['TSNE_1'] = tsne_result[:, 0]
        df['TSNE_2'] = tsne_result[:, 1]
        st.success("t-SNE completado.")
    except Exception as e:
        st.error(f"Error en t-SNE: {e}")

    # Visualización UMAP
    try:
        st.write("Visualización con UMAP:")
        fig_umap = px.scatter(
            df, x='UMAP_1', y='UMAP_2',
            color='Genre',
            hover_data=['Title', 'Year', 'IMDb Rating', 'Director'],
            title="Visualización de Embeddings con UMAP"
        )
        st.plotly_chart(fig_umap)
    except Exception as e:
        st.error(f"Error visualizando UMAP: {e}")

    # Visualización PCA
    try:
        st.write("Visualización con PCA:")
        fig_pca = px.scatter(
            df, x='PCA_1', y='PCA_2',
            color='Genre',
            hover_data=['Title', 'Year', 'IMDb Rating', 'Director'],
            title="Visualización de Embeddings con PCA"
        )
        st.plotly_chart(fig_pca)
    except Exception as e:
        st.error(f"Error visualizando PCA: {e}")

    # Visualización t-SNE
    try:
        st.write("Visualización con t-SNE:")
        fig_tsne = px.scatter(
            df, x='TSNE_1', y='TSNE_2',
            color='Genre',
            hover_data=['Title', 'Year', 'IMDb Rating', 'Director'],
            title="Visualización de Embeddings con t-SNE"
        )
        st.plotly_chart(fig_tsne)
    except Exception as e:
        st.error(f"Error visualizando t-SNE: {e}")

    # Interpretaciones finales
    st.info(
        "Estas visualizaciones permiten identificar patrones y similitudes entre películas. "
        "Las técnicas utilizadas, como UMAP, PCA y t-SNE, ayudan a representar datos de alta dimensionalidad "
        "en un espacio más manejable, facilitando la detección de clústeres y relaciones interesantes."
    )
