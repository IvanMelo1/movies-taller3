import streamlit as st
from recomendacion import sistema_recomendacion
from visualizaciones import generar_visualizaciones
from agente import chatbot

st.title("Sistema de Recomendación con Modelos de Lenguaje")

opciones = ["Recomendación", "Visualizaciones", "Chatbot"]
seleccion = st.sidebar.selectbox("Selecciona una funcionalidad:", opciones)

if seleccion == "Recomendación":
    st.header("Sistema de Recomendación")
    sistema_recomendacion()
elif seleccion == "Visualizaciones":
    st.header("Visualizaciones")
    generar_visualizaciones()
elif seleccion == "Chatbot":
    st.header("Agente Inteligente")
    chatbot()
