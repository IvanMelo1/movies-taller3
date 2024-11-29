import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

def sistema_recomendacion():
    # Cargar el modelo de embeddings
    modelo = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("Modelo cargado: **all-MiniLM-L6-v2**")
    
    # Cargar datos del dataset
    st.write("Cargando datos...")
    df = pd.read_csv("dataset.csv")  # Asegúrate de que el archivo dataset.csv está en la misma carpeta
    st.write("Primeras filas del dataset:")
    st.dataframe(df.head())
    
    # Generar la columna 'texto'
    st.write("Generando columna 'texto' para embeddings...")
    df['texto'] = (
        df['Title'].astype(str) + " " +
        df['IMDb Rating'].astype(str) + " " +
        df['Year'].astype(str) + " " +
        df['Genre'].astype(str) + " " +
        df['Director'].astype(str) + " " +
        df['Star Cast'].astype(str) + " " +
        df['MetaScore'].astype(str)
    )
    st.write("Columna 'texto' generada:")
    st.dataframe(df['texto'].head())
    
    # Calcular los embeddings con una barra de progreso
    st.write("Calculando embeddings...")
    progress_bar = st.progress(0)
    embeddings = []
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        embeddings.append(modelo.encode(row['texto']).tolist())
        progress_bar.progress((idx + 1) / total_rows)  # Actualizar barra de progreso
    
    df['embedding'] = embeddings
    st.success("Embeddings generados exitosamente.")
    
    # Más lógica de recomendación aquí
    st.write("Recomendaciones listas.")
    st.info("¡Ya puedes pasar al chatbot para explorar las recomendaciones personalizadas!")
    
    # Guardar el dataset con embeddings procesados
    df.to_csv("data/processed/embedded_dataset.csv", index=False)
