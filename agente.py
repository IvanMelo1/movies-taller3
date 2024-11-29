from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import streamlit as st
from config import Config
import requests  # Para usar APIs como TMDb

def obtener_poster_tmdb(titulo):
    """Busca el póster de una película usando la API de TMDb."""
    api_key = "TU_API_KEY_DE_TMDB"  # Sustituir por tu API Key de TMDb
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={titulo}"
    response = requests.get(url).json()
    if response.get("results"):
        poster_path = response["results"][0].get("poster_path")
        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
    return None

def chatbot():
    st.write("🤖 **Chatbot de Recomendaciones de Películas**")

    # Cargar embeddings y datos
    try:
        df = pd.read_csv(Config.DATA_PROCESSED_PATH + "embedded_dataset.csv")
        df['embedding'] = df['embedding'].apply(eval)  # Convertir los embeddings de string a lista
        modelo_embeddings = SentenceTransformer(Config.EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Error cargando datos o modelo: {e}")
        return

    # Entrada del usuario
    input_usuario = st.text_input("🎥 Describe qué tipo de película te interesa (género, actores, director, etc.):")

    if input_usuario:
        # Obtener el embedding del usuario
        embedding_usuario = modelo_embeddings.encode(input_usuario, convert_to_tensor=True).to(dtype=torch.float32)
        
        # Convertir los embeddings de las películas a tensores de PyTorch
        embeddings_peliculas = torch.tensor(df['embedding'].tolist(), dtype=torch.float32)

        # Calcular similitudes entre el embedding del usuario y las películas
        similitudes = util.cos_sim(embedding_usuario, embeddings_peliculas)[0]
        top_indices = torch.topk(similitudes, k=20).indices  # Obtenemos un top más amplio (e.g., 20)

        # Crear un DataFrame con los resultados de similitud
        df_resultados = df.iloc[top_indices.tolist()].copy()
        df_resultados['similaridad'] = similitudes[top_indices].tolist()

        # Eliminar duplicados por título (mantener la película con mayor similitud)
        df_resultados = df_resultados.sort_values('similaridad', ascending=False)
        df_unicos = df_resultados.drop_duplicates(subset='Title')

        # Seleccionar solo los top 5 únicos
        top_unicos = df_unicos.head(5)

        # Mostrar las recomendaciones
        st.subheader("🎬 **Tus Recomendaciones:**")
        for _, row in top_unicos.iterrows():
            titulo = row['Title']
            año = row['Year']
            rating = row['IMDb Rating']
            director = row['Director']
            genero = row['Genre']
            poster_url = obtener_poster_tmdb(titulo)

            # Mostrar detalles de la película
            st.markdown(f"**🎥 {titulo}** ({año})")
            st.write(f"- ⭐ **Rating IMDb:** {rating}")
            st.write(f"- 🎭 **Género:** {genero}")
            st.write(f"- 🎬 **Director:** {director}")
            if poster_url:
                st.image(poster_url, width=150)
            st.write("---")
