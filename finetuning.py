import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import os

def realizar_finetuning():
    st.header("Fine-tuning del Modelo de Embeddings")

    # Cargar el dataset
    st.write("Cargando el dataset...")
    try:
        dataset_path = st.file_uploader("Sube tu archivo CSV con el dataset para fine-tuning", type=["csv"])
        if dataset_path is None:
            st.warning("Por favor, sube un archivo CSV para continuar.")
            return

        df = pd.read_csv(dataset_path)
        st.success("Dataset cargado correctamente.")
        st.write("Vista previa del dataset:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return

    # Verificar columnas necesarias
    required_columns = ["sentence1", "sentence2", "label"]
    if not all(col in df.columns for col in required_columns):
        st.error(f"El dataset debe contener las columnas: {required_columns}")
        return

    # Selección del modelo base
    st.write("Seleccionando el modelo base...")
    modelo_base = st.selectbox(
        "Elige un modelo preentrenado para ajustar:",
        ["all-MiniLM-L6-v2", "distilbert-base-nli-stsb-mean-tokens", "paraphrase-MiniLM-L12-v2"]
    )
    model = SentenceTransformer(modelo_base)
    st.success(f"Modelo '{modelo_base}' cargado correctamente.")

    # Preparar los datos para el fine-tuning
    st.write("Preparando los datos para el fine-tuning...")
    try:
        train_examples = [
            InputExample(texts=[row["sentence1"], row["sentence2"]], label=float(row["label"]))
            for _, row in df.iterrows()
        ]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model=model)
        st.success("Datos preparados para el fine-tuning.")
    except Exception as e:
        st.error(f"Error al preparar los datos: {e}")
        return

    # Configuración de parámetros de entrenamiento
    st.write("Configuración del entrenamiento...")
    num_epochs = st.slider("Selecciona el número de épocas:", min_value=1, max_value=10, value=3)
    output_path = st.text_input("Ruta para guardar el modelo ajustado:", "models/finetuned_model")

    # Iniciar el proceso de fine-tuning
    if st.button("Iniciar Fine-tuning"):
        try:
            st.write("Entrenando el modelo...")
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                warmup_steps=int(0.1 * len(train_dataloader) * num_epochs),
                output_path=output_path
            )
            st.success(f"Fine-tuning completado. Modelo guardado en '{output_path}'.")
        except Exception as e:
            st.error(f"Error durante el fine-tuning: {e}")

    # Evaluación del modelo ajustado (opcional)
    st.write("Opcional: Evalúa el modelo ajustado con un nuevo conjunto de datos.")
    eval_dataset_path = st.file_uploader("Sube un archivo CSV con datos de evaluación (opcional)", type=["csv"])
    if eval_dataset_path:
        try:
            eval_df = pd.read_csv(eval_dataset_path)
            st.write("Vista previa de datos de evaluación:")
            st.dataframe(eval_df.head())

            st.write("Calculando similitudes...")
            eval_df["similarity"] = eval_df.apply(
                lambda row: model.similarity([row["sentence1"], row["sentence2"]])[0],
                axis=1
            )
            st.success("Evaluación completada. Resultados:")
            st.dataframe(eval_df[["sentence1", "sentence2", "similarity"]].head())
        except Exception as e:
            st.error(f"Error al evaluar el modelo ajustado: {e}")
